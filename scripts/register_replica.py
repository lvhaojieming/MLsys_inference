"""Register a started backend; gateway waits for readiness and gates admission."""
import argparse
import json
import os
from pathlib import Path
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gateway', required=True)
    parser.add_argument('--replica', required=True, help='JSON file with Replica fields')
    parser.add_argument('--token-env', default='MOQE_ADMIN_TOKEN')
    parser.add_argument('--output', required=True)
    parser.add_argument('--timeout', type=float, default=360)
    args = parser.parse_args()
    token = os.environ.get(args.token_env)
    if not token:
        parser.error('Management token environment variable is missing')
    replica = json.loads(Path(args.replica).read_text(encoding='utf-8'))
    request = urllib.request.Request(args.gateway.rstrip('/') + '/admin/instances',
                                    data=json.dumps(replica).encode(),
                                    headers={'Content-Type': 'application/json',
                                             'Authorization': 'Bearer ' + token})
    try:
        response = urllib.request.urlopen(request, timeout=args.timeout)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        result = json.load(response)
        status = response.status
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'http_status': status, 'result': result}, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)
    return 0 if status == 201 and result.get('state') == 'ready' else 1


if __name__ == '__main__':
    raise SystemExit(main())
