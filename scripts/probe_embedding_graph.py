"""Compare eager and NPUGraph encoder execution without changing live serving."""
import argparse
import json
from pathlib import Path
import statistics
import time

from moqe_serving.config import Settings
from moqe_serving.routing.ascend import AscendRouterRuntime


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--labels', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    runtime = AscendRouterRuntime(Settings.load(args.config).router)
    torch = runtime.torch
    import torch_npu
    rows = sorted([r for r in map(json.loads, Path(args.labels).open()) if r['split'] == 'test'],
                  key=lambda r: len(r['input_ids']))
    report = {'passed': False, 'samples': []}
    try:
        for index in [0, len(rows)//2, len(rows)*9//10, len(rows)-1]:
            row = rows[index]
            text = runtime.tokenizer.decode(row['input_ids'], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            encoded = runtime.encoder_tokenizer(text, return_tensors='pt')
            static = {k: v.to(runtime.device) for k, v in encoded.items()}
            length = static['input_ids'].shape[1]
            positions = torch.arange(length, device=runtime.device).unsqueeze(0)
            def forward():
                # Keep original causal SDPA, bypassing mask-generation scalar synchronizations.
                return runtime.encoder(input_ids=static['input_ids'], attention_mask={'full_attention': None},
                    position_ids=positions, use_cache=False).last_hidden_state
            with torch.inference_mode():
                baseline = runtime.encoder(**static, use_cache=False).last_hidden_state
                for _ in range(3):
                    reference = forward()
                torch.npu.synchronize()
                graph = torch_npu.npu.NPUGraph()
                with torch_npu.npu.graph(graph):
                    output = forward()
                graph.replay()
                torch.npu.synchronize()
                error = (output.float()-reference.float()).abs().max().item()
                mask_error = (reference[:, -1].float()-baseline[:, -1].float()).abs().max().item()
                timings = {}
                for name in ['eager', 'graph']:
                    values = []
                    for _ in range(10):
                        torch.npu.synchronize()
                        start = time.perf_counter()
                        if name == 'graph':
                            graph.replay()
                        else:
                            forward()
                        torch.npu.synchronize()
                        values.append((time.perf_counter()-start)*1000)
                    timings[name] = statistics.median(values)
                sample = {'tokens': length, 'timings_ms': timings, 'graph_error': error,
                          'causal_path_last_token_error': mask_error}
                assert error == 0 and mask_error == 0, 'Capture changed encoder outputs'
                report['samples'].append(sample)
                Path(args.output).write_text(json.dumps(report, indent=2))
                print(json.dumps(sample), flush=True)
                graph.reset()
        report['passed'] = True
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        Path(args.output).write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
