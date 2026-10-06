"""Sequential warm TTFT for direct backend, fixed gateway, and V7 auto routing."""
import argparse
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time

import httpx


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--labels', required=True)
    p.add_argument('--predictions', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--samples', type=int, default=24)
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = sorted([json.loads(l) for l in Path(args.labels).open() if json.loads(l)['split'] == 'test'],
                  key=lambda r: len(r['input_ids']))
    selected = [rows[round(i * (len(rows)-1)/(args.samples-1))] for i in range(args.samples)]
    predictions = {r['id']: r for r in map(json.loads, Path(args.predictions).open())}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg['router']['tokenizer'], local_files_only=True)
    template = tokenizer.get_chat_template()
    base = 'http://127.0.0.1:' + str(cfg['gateway']['port'])
    replicas = {r['expert']: r for r in cfg['replicas']}
    report = {'passed': False, 'started_at': time.time(), 'samples': [],
              'scope': 'warm, sequential concurrency=1; TTFT is first nonempty content delta; excludes startup and WAN client',
              'checkpoint': cfg['router']['checkpoint'], 'selection': 'validation-best seed; test prompts evenly spaced by token length'}

    def save():
        (out/'report.json').write_text(json.dumps(report, indent=2))

    def prompt(row):
        text = tokenizer.decode(row['input_ids'], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        prefix = '<|im_start|>user\n'
        assert text.startswith(prefix), 'Only single-user prompts supported by this benchmark'
        content = text[len(prefix):].split('<|im_end|>', 1)[0]
        messages = [{'role': 'user', 'content': content}]
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           enable_thinking=False, return_dict=False)
        assert ids == row['input_ids'], 'Benchmark must preserve original serialized prompt'
        return messages

    def request(client, url, payload):
        start = time.perf_counter()
        first = None
        parts = []
        done = False
        with client.stream('POST', url, json=payload) as response:
            response.raise_for_status()
            headers = dict(response.headers)
            for line in response.iter_lines():
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    done = True
                    break
                event = json.loads(data)
                if event.get('error'):
                    raise RuntimeError(event['error'])
                for c in event.get('choices', []):
                    content = c.get('delta', {}).get('content')
                    if content:
                        first = first if first is not None else (time.perf_counter()-start)*1000
                        parts.append(content)
        assert first is not None and done, 'No content or incomplete stream'
        return {'ttft_ms': first, 'e2e_ms': (time.perf_counter()-start)*1000,
                'router_ms': float(headers['x-moqe-router-ms']) if 'x-moqe-router-ms' in headers else None,
                'expert': headers.get('x-moqe-expert'), 'replica': headers.get('x-moqe-replica'),
                'content': ''.join(parts)}

    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1', cfg['gateway']['port'])) != 0, 'Test Gateway port occupied'
    log = (out/'gateway.log').open('wb')
    process = subprocess.Popen([sys.executable, '-m', 'moqe_serving', '--config', args.config], stdout=log, stderr=log)
    try:
        with httpx.Client(timeout=180) as client:
            deadline = time.monotonic()+180
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError('Gateway exited; inspect gateway.log')
                try:
                    ready = client.get(base+'/ready')
                    if ready.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError('Gateway readiness timeout')
            # Discard full-pipeline warmups; backend admission already warmed each expert.
            for row in selected[:3]:
                payload = {'model': 'auto', 'messages': prompt(row), 'max_tokens': row['max_new_tokens'],
                           'stream': True, 'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False}}
                request(client, base+'/v1/chat/completions', payload)
            rng = random.Random(20261006)
            for index, row in enumerate(selected):
                expert = ('awq', 'gptq')[predictions[row['id']]['selected_calibrated']]
                payload = {'messages': prompt(row), 'max_tokens': row['max_new_tokens'], 'stream': True,
                           'temperature': 0, 'chat_template': template, 'chat_template_kwargs': {'enable_thinking': False}}
                modes = ['direct', 'fixed_gateway', 'auto_gateway']
                rng.shuffle(modes)
                sample = {'id': row['id'], 'source': row['source'], 'prompt_tokens': len(row['input_ids']),
                          'max_tokens': row['max_new_tokens'], 'expected_expert': expert, 'order': modes}
                for mode in modes:
                    if mode == 'direct':
                        r = replicas[expert]
                        sample[mode] = request(client, r['base_url']+'/chat/completions', {**payload, 'model': r['model']})
                    else:
                        sample[mode] = request(client, base+'/v1/chat/completions',
                            {k:v for k,v in {**payload, 'model': 'auto' if mode == 'auto_gateway' else expert}.items()
                             if k != 'chat_template' or mode != 'auto_gateway'})
                assert sample['auto_gateway']['expert'] == expert, 'Runtime disagrees with trained checkpoint prediction'
                # Separate requests can vary even at temperature zero. Preserve
                # outputs and count differences without censoring latency samples.
                sample['outputs_equal'] = sample['direct']['content'] == sample['fixed_gateway']['content'] == sample['auto_gateway']['content']
                report['samples'].append(sample)
                save()
                print(json.dumps({'completed': index+1, 'total': len(selected), 'tokens': sample['prompt_tokens'],
                                  'auto_ttft_ms': sample['auto_gateway']['ttft_ms'],
                                  'router_ms': sample['auto_gateway']['router_ms']}), flush=True)
        def stats(values):
            import statistics
            values = sorted(values)
            def percentile(q):
                position = (len(values)-1)*q
                low = int(position)
                high = min(low+1, len(values)-1)
                return values[low] + (values[high]-values[low])*(position-low)
            return {'mean': statistics.mean(values), 'p50': percentile(.5), 'p95': percentile(.95), 'min': values[0], 'max': values[-1]}
        report['summary_ms'] = {mode: stats([s[mode]['ttft_ms'] for s in report['samples']])
                                for mode in ['direct', 'fixed_gateway', 'auto_gateway']}
        report['summary_ms']['router'] = stats([s['auto_gateway']['router_ms'] for s in report['samples']])
        report['summary_ms']['auto_minus_fixed'] = stats([s['auto_gateway']['ttft_ms']-s['fixed_gateway']['ttft_ms'] for s in report['samples']])
        report['output_difference_samples'] = sum(not s['outputs_equal'] for s in report['samples'])
        report['passed'] = True
    except Exception as exc:
        report['error'] = str(exc)
        raise
    finally:
        report['finished_at'] = time.time()
        save()
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()


if __name__ == '__main__':
    main()
