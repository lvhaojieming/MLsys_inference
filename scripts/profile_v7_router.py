"""Synchronised warm timing breakdown of contextual routing on the serving device."""
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
    times = {}

    def wrap(obj, method, name, device=False):
        original = getattr(obj, method)
        def measured(*a, **kw):
            if device:
                torch.npu.synchronize()
            start = time.perf_counter()
            result = original(*a, **kw)
            if device:
                torch.npu.synchronize()
            times[name] = (time.perf_counter()-start)*1000
            return result
        setattr(obj, method, measured)

    wrap(runtime.encoder, 'forward', 'embedding_forward_ms', device=True)
    wrap(runtime.router, 'forward', 'mlp_forward_cpu_ms')
    wrap(runtime.tokenizer, 'apply_chat_template', 'chat_tokenize_ms')
    wrap(runtime.tokenizer, 'decode', 'decode_ms')
    original = runtime.encoder_tokenizer
    class TokenizerProxy:
        def __call__(self, *a, **kw):
            start = time.perf_counter()
            result = original(*a, **kw)
            times['embedding_tokenize_ms'] = (time.perf_counter()-start)*1000
            return result
    runtime.encoder_tokenizer = TokenizerProxy()
    rows = sorted([r for r in map(json.loads, Path(args.labels).open()) if r['split'] == 'test'],
                  key=lambda r: len(r['input_ids']))
    report = {'scope': 'isolated warm component profile; NPU synchronization brackets embedding forward', 'samples': []}
    for index in [0, len(rows)//2, len(rows)*9//10, len(rows)-1]:
        row = rows[index]
        text = runtime.tokenizer.decode(row['input_ids'], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        content = text[len('<|im_start|>user\n'):].split('<|im_end|>', 1)[0]
        messages = [{'role': 'user', 'content': content}]
        for _ in range(2):
            runtime.route(messages, row['max_new_tokens'], {'enable_thinking': False})
        records = []
        for _ in range(5):
            times.clear()
            decision = runtime.route(messages, row['max_new_tokens'], {'enable_thinking': False})
            records.append({**times, 'total_ms': decision.elapsed_ms,
                            'other_ms': decision.elapsed_ms-sum(times.values())})
        sample = {'id': row['id'], 'prompt_tokens': len(row['input_ids']), 'repetitions': records,
                  'median_ms': {k: statistics.median(r[k] for r in records) for k in records[0]}}
        report['samples'].append(sample)
        print(json.dumps(sample['median_ms'] | {'prompt_tokens': sample['prompt_tokens']}), flush=True)
        Path(args.output).write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
