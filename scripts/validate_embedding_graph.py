"""Check graph routing against saved V7 predictions on every held-out sample."""
import argparse
import json
from pathlib import Path
import statistics

from moqe_serving.config import Settings
from moqe_serving.routing.ascend import AscendRouterRuntime


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--labels', required=True)
    p.add_argument('--predictions', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    runtime = AscendRouterRuntime(Settings.load(args.config).router)
    assert runtime.embedding_graphs
    rows = [r for r in map(json.loads, Path(args.labels).open()) if r['split'] in ('valid', 'test')]
    prior = {r['id']: r for r in map(json.loads, Path(args.predictions).open())}
    report = {'passed': False, 'samples': [], 'mismatches': []}
    graphs = runtime.embedding_graphs
    graphs.hits = graphs.fallbacks = graphs.oversize = 0
    for row in rows:
        text = runtime.tokenizer.decode(row['input_ids'], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        prefix = '<|im_start|>user\n'
        assert text.startswith(prefix)
        messages = [{'role': 'user', 'content': text[len(prefix):].split('<|im_end|>', 1)[0]}]
        assert runtime.tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
            add_generation_prompt=True, enable_thinking=False) == row['input_ids']
        decision = runtime.route(messages, row['max_new_tokens'], {'enable_thinking': False})
        expected = ('awq', 'gptq')[prior[row['id']]['selected_calibrated']]
        error = max(abs(decision.probabilities[e]-prior[row['id']]['expert_probabilities'][i])
                    for i,e in enumerate(('awq', 'gptq')))
        sample = {'id': row['id'], 'split': row['split'], 'tokens': decision.input_tokens,
                  'route_ms': decision.elapsed_ms, 'probability_error': error,
                  'expected': expected, 'actual': decision.expert}
        report['samples'].append(sample)
        if decision.expert != expected:
            report['mismatches'].append(sample)
        if len(report['samples']) % 100 == 0:
            print(json.dumps({'completed': len(report['samples']), 'total': len(rows),
                              'mismatches': len(report['mismatches']), 'fallbacks': graphs.fallbacks}), flush=True)
            Path(args.output).write_text(json.dumps(report, indent=2))
    latencies = sorted(s['route_ms'] for s in report['samples'])
    report['summary'] = {'samples': len(rows), 'mismatches': len(report['mismatches']),
                         'graph_hits': graphs.hits, 'boundary_fallbacks': graphs.fallbacks, 'oversize': graphs.oversize,
                         'route_ms_median': statistics.median(latencies),
                         'route_ms_p95': latencies[round((len(latencies)-1)*.95)],
                         'max_probability_error': max(s['probability_error'] for s in report['samples'])}
    report['passed'] = not report['mismatches']
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report['summary']), flush=True)
    if not report['passed']:
        raise RuntimeError('Graph changed held-out routing decisions')


if __name__ == '__main__':
    main()
