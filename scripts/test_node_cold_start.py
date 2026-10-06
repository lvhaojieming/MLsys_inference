"""Verify a managed node at initial startup or hot addition, then drain and stop it."""
import argparse
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--initial-start', action='store_true',
                        help='Enable the managed node before starting Gateway')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    existing, new = config['replicas']
    assert existing['launch'] is None and new['launch'], 'Requires one existing and one managed replica'
    assert existing['expert'] == new['expert'] and existing['node_id'] != new['node_id']
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'active-config.json'
    config['lifecycle_log_dir'] = str(output / 'lifecycle')
    config['admin_token_env'] = 'MOQE_COLD_TEST_TOKEN'
    token = secrets.token_hex(24)
    auth = {'Authorization': 'Bearer ' + token}
    base = 'http://127.0.0.1:' + str(config['gateway']['port'])
    report = {'started_at': time.time(), 'scope': 'managed process cold start, not filesystem-cache cold start',
              'passed': False, 'checks': [], 'samples': []}
    new_node = next(n for n in config['nodes'] if n['id'] == new['node_id'])
    new_node['enabled'] = args.initial_start
    report['mode'] = 'initial-start' if args.initial_start else 'hot-add'

    def save():
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(config, indent=2))
        temporary.replace(path)

    def persist():
        (output / 'report.json').write_text(json.dumps(report, indent=2))

    def record(name, **data):
        item = {'name': name, 'passed': True, **data}
        report['checks'].append(item)
        persist()
        print(json.dumps(item), flush=True)

    def snapshot():
        response = httpx.get(base + '/admin/config', headers=auth, timeout=5)
        response.raise_for_status()
        return response.json()

    def wait(predicate, timeout=120):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError('Gateway exited; inspect gateway.log')
            try:
                s = snapshot()
                if predicate(s):
                    return s
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        raise TimeoutError('Pool state wait timed out')

    def state(s, replica):
        return next((r['state'] for r in s['instances'] if r['id'] == replica), None)

    def generate(stream=False):
        payload = {'model': existing['expert'], 'messages': [{'role': 'user', 'content': config['admission']['prompt']}],
                   'max_tokens': config['admission']['max_tokens'], 'temperature': 0, 'stream': stream,
                   'chat_template_kwargs': {'enable_thinking': False}}
        start = time.perf_counter()
        with httpx.Client(timeout=120) as client:
            if not stream:
                response = client.post(base + '/v1/chat/completions', json=payload)
                response.raise_for_status()
                content = response.json()['choices'][0]['message']['content']
                replica = response.headers['x-moqe-replica']
            else:
                parts, done = [], False
                with client.stream('POST', base + '/v1/chat/completions', json=payload) as response:
                    response.raise_for_status()
                    replica = response.headers['x-moqe-replica']
                    for line in response.iter_lines():
                        if not line.startswith('data:'):
                            continue
                        data = line[5:].strip()
                        if data == '[DONE]':
                            done = True
                            break
                        parts.extend(c.get('delta', {}).get('content') or '' for c in json.loads(data).get('choices', []))
                assert done, 'Incomplete stream'
                content = ''.join(parts)
        assert content.strip() == config['admission']['expected'].strip(), content
        return {'replica': replica, 'stream': stream, 'seconds': time.perf_counter() - start, 'content': content}

    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1', config['gateway']['port'])) != 0, 'Gateway port occupied'
    with socket.socket() as sock:
        assert sock.connect_ex((new_node['host'], new['backend_port'])) != 0, 'Cold-start backend port already occupied'
    save()
    log = (output / 'gateway.log').open('wb')
    spawned_at = time.time()
    process = subprocess.Popen([sys.executable, '-m', 'moqe_serving', '--config', str(path)],
                               env={**os.environ, 'MOQE_COLD_TEST_TOKEN': token}, stdout=log, stderr=log)
    enabled = args.initial_start
    try:
        wait(lambda s: state(s, existing['id']) == 'ready',
             timeout=config['admission_timeout_seconds'] + 300 if args.initial_start else 120)
        record('existing_node_available', response=generate())
        if args.initial_start:
            report['activation_at'] = spawned_at
        else:
            report['activation_at'] = time.time()
            new_node['enabled'] = True
            enabled = True
            save()
            wait(lambda s: state(s, new['id']) == 'starting')
            record('starting_is_not_schedulable', response=generate())
            assert report['checks'][-1]['response']['replica'] == existing['id']
        deadline = time.monotonic() + config['admission_timeout_seconds'] + 90
        while time.monotonic() < deadline:
            s = snapshot()
            current = state(s, new['id'])
            report['samples'].append({'at': time.time(), 'state': current})
            persist()
            if current == 'ready':
                break
            if current == 'unhealthy':
                raise RuntimeError('New node failed admission: ' + json.dumps(s))
            time.sleep(2)
        else:
            raise TimeoutError('Cold start exceeded deadline')
        report['ready_at'] = time.time()
        preparation = next(n for n in s['nodes'] if n['id'] == new['node_id']).get('preparation')
        if new_node.get('prepare'):
            assert preparation and preparation['passed'], 'Node preparation missing or failed'
            record('node_preparation', result=preparation)
        record('cold_start_admission', seconds=report['ready_at'] - report['activation_at'], snapshot=s)
        # Exercise each response mode across both replicas independently of
        # the round-robin cursor advanced by earlier probes.
        responses = [generate(stream) for stream in (False, True) for _ in range(4)]
        assert {r['replica'] for r in responses} == {existing['id'], new['id']}
        assert any(r['replica'] == new['id'] and r['stream'] for r in responses)
        record('new_node_real_traffic', responses=responses)
        new_node['enabled'] = False
        save()
        s = wait(lambda s: state(s, new['id']) == 'offline' and new['id'] not in s['managed_replicas'])
        enabled = False
        with socket.socket() as sock:
            assert sock.connect_ex((new_node['host'], new['backend_port'])) != 0, 'Managed process still listening'
        record('node_drain_and_process_stop', snapshot=s, remaining_response=generate())
        report['passed'] = True
    except Exception as exc:
        report['error'] = str(exc)
        raise
    finally:
        if enabled:
            new_node['enabled'] = False
            save()
            try:
                wait(lambda s: state(s, new['id']) == 'offline' and new['id'] not in s['managed_replicas'], timeout=120)
            except Exception as exc:
                report['cleanup_error'] = str(exc)
        report['finished_at'] = time.time()
        persist()
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()


if __name__ == '__main__':
    main()
