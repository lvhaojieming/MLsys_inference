"""Real backend integration: admission, hot add, traffic, active drain, re-enable."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = json.loads(Path(args.config).read_text())
    if len(source['replicas']) != 2 or any(r.get('launch') for r in source['replicas']):
        parser.error('This test requires exactly two existing, unmanaged replicas of the same expert')
    if len({r['expert'] for r in source['replicas']}) != 1 or source.get('router'):
        parser.error('This test isolates same-expert replica scheduling without Router')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / 'active-config.json'
    config = json.loads(json.dumps(source))
    first, second = [r['id'] for r in config['replicas']]
    node_ids = [r.get('node_id') for r in config['replicas']]
    node_mode = all(node_ids) and node_ids[0] != node_ids[1]
    def set_enabled(index, value):
        if node_mode:
            node = next(n for n in config['nodes'] if n['id'] == node_ids[index])
            node['enabled'] = value
            config['replicas'][index]['enabled'] = True
        else:
            config['replicas'][index]['enabled'] = value
    set_enabled(0, True)
    set_enabled(1, False)
    config['lifecycle_log_dir'] = str(output / 'lifecycle')
    config['admin_token_env'] = 'MOQE_SYSTEM_TEST_TOKEN'
    token = secrets.token_hex(24)
    auth = {'Authorization': 'Bearer ' + token}
    base = 'http://127.0.0.1:' + str(config['gateway']['port'])
    report = {'started_at': time.time(), 'source_config': str(Path(args.config).resolve()),
              'scope': 'real existing AWQ replicas; gateway process and config lifecycle',
              'expansion_unit': 'node' if node_mode else 'replica',
              'replica_nodes': dict(zip((first, second), node_ids)), 'checks': []}

    def save_config():
        temporary = config_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(config, indent=2))
        temporary.replace(config_path)

    def record(name, **details):
        report['checks'].append({'name': name, 'passed': True, **details})
        (output / 'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report['checks'][-1]), flush=True)

    def wait(predicate, timeout=180):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError('Test Gateway exited; inspect gateway.log')
            try:
                response = httpx.get(base + '/admin/config', headers=auth, timeout=5)
                response.raise_for_status()
                snapshot = response.json()
                if predicate(snapshot):
                    return snapshot
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        raise TimeoutError('Timed out waiting for pool state')

    def states(snapshot):
        return {r['id']: r['state'] for r in snapshot['instances']}

    def send(stream=False, long=False):
        payload = {'model': config['replicas'][0]['expert'], 'temperature': 0, 'stream': stream,
                   'max_tokens': 512 if long else config['admission']['max_tokens'],
                   'messages': [{'role': 'user', 'content':
                     'List the integers from 1 to 300, one per line. Do not omit any integers.'
                     if long else config['admission']['prompt']}],
                   'chat_template_kwargs': {'enable_thinking': False}}
        start = time.perf_counter()
        content, first_content, done = [], None, False
        with httpx.Client(timeout=180) as client:
            if not stream:
                response = client.post(base + '/v1/chat/completions', json=payload)
                response.raise_for_status()
                content = [response.json()['choices'][0]['message']['content']]
                headers = dict(response.headers)
            else:
                with client.stream('POST', base + '/v1/chat/completions', json=payload) as response:
                    response.raise_for_status()
                    headers = dict(response.headers)
                    for line in response.iter_lines():
                        if not line.startswith('data:'):
                            continue
                        data = line[5:].strip()
                        if data == '[DONE]':
                            done = True
                            break
                        for choice in json.loads(data).get('choices', []):
                            text = choice.get('delta', {}).get('content')
                            if text:
                                first_content = first_content or time.perf_counter() - start
                                content.append(text)
                assert done and first_content is not None, 'Stream did not finish correctly'
        text = ''.join(content).strip()
        assert text if long else text == config['admission']['expected'].strip(), text
        return {'replica': headers['x-moqe-replica'], 'stream': stream,
                'e2e_seconds': time.perf_counter() - start, 'ttft_seconds': first_content,
                'content': text}

    save_config()
    # Refuse to run against a pre-existing listener.
    import socket
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1', config['gateway']['port'])) != 0, 'Test port occupied'
    log = (output / 'gateway.log').open('wb')
    process = subprocess.Popen([sys.executable, '-m', 'moqe_serving', '--config', str(config_path)],
                               env={**os.environ, 'MOQE_SYSTEM_TEST_TOKEN': token}, stdout=log, stderr=log)
    report['gateway_pid'] = process.pid
    try:
        snapshot = wait(lambda s: states(s).get(first) == 'ready')
        record('initial_admission', snapshot=snapshot)
        record('initial_generation', response=send())
        record('initial_stream', response=send(stream=True))
        set_enabled(1, True)
        save_config()
        snapshot = wait(lambda s: states(s).get(second) == 'ready')
        record('config_hot_add_node' if node_mode else 'config_hot_add', snapshot=snapshot)
        with ThreadPoolExecutor(max_workers=4) as pool:
            responses = list(pool.map(lambda _: send(), range(8)))
        assert {r['replica'] for r in responses} == {first, second}, 'Traffic did not reach both replicas'
        record('both_replicas_serve', responses=responses)
        # Make first the sole target for a long request, then re-add second before draining first.
        set_enabled(1, False)
        save_config()
        wait(lambda s: states(s).get(second) == 'offline')
        with ThreadPoolExecutor(max_workers=1) as pool:
            running = pool.submit(send, False, True)
            wait(lambda s: any(r['id'] == first and r['inflight'] > 0 for r in s['instances']))
            set_enabled(1, True)
            save_config()
            wait(lambda s: states(s).get(second) == 'ready')
            set_enabled(0, False)
            save_config()
            wait(lambda s: states(s).get(first) == 'draining')
            other = send()
            assert other['replica'] == second
            long_result = running.result(timeout=180)
            assert long_result['replica'] == first
        snapshot = wait(lambda s: states(s).get(first) == 'offline')
        assert all(r['inflight'] == 0 for r in snapshot['instances'])
        record('active_request_node_drain' if node_mode else 'active_request_drain', snapshot=snapshot, active_response=long_result,
               new_response=other)
        record('remaining_replica_stream', response=send(stream=True))
        set_enabled(0, True)
        save_config()
        snapshot = wait(lambda s: states(s).get(first) == 'ready')
        record('re_enable_requires_admission', snapshot=snapshot)
        report['passed'] = True
    except Exception as exc:
        report['passed'] = False
        report['error'] = str(exc)
        raise
    finally:
        report['finished_at'] = time.time()
        (output / 'report.json').write_text(json.dumps(report, indent=2))
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
