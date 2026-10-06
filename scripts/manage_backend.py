"""Linux helper: detach a configured backend and stop only its recorded process group."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def identity(pid):
    try:
        # Linux /proc start ticks distinguish a recycled PID from our backend.
        text = Path(f'/proc/{pid}/stat').read_text()
        fields = text[text.rfind(')') + 2:].split()
        if fields[0] == 'Z':
            return None
        return fields[19]
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('start', 'stop'))
    parser.add_argument('--pid-file', required=True)
    parser.add_argument('--log-file')
    parser.add_argument('--stop-timeout', type=float, default=30)
    arguments = sys.argv[1:]
    delimiter = arguments.index('--') if '--' in arguments else len(arguments)
    args = parser.parse_args(arguments[:delimiter])
    if os.name != 'posix' or not Path('/proc').exists():
        parser.error('This backend process helper requires Linux')
    path = Path(args.pid_file)
    command = arguments[delimiter + 1:]
    recorded = json.loads(path.read_text()) if path.exists() else None
    live = recorded and identity(recorded['pid']) == recorded['start_ticks']
    if args.action == 'start':
        if not command or not args.log_file:
            parser.error('start requires --log-file and a foreground backend command after --')
        if live:
            if recorded['command'] != command:
                raise RuntimeError('PID file belongs to a different live command')
            print(json.dumps({'status': 'already_started', 'pid': recorded['pid']}))
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        log = Path(args.log_file)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('ab') as output:
            process = subprocess.Popen(command, stdout=output, stderr=output,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        ticks = identity(process.pid)
        if ticks is None:
            raise RuntimeError('Backend exited immediately; inspect its log')
        record = {'pid': process.pid, 'start_ticks': ticks, 'command': command}
        temp = path.with_suffix(path.suffix + '.tmp')
        temp.write_text(json.dumps(record))
        temp.replace(path)
        print(json.dumps({'status': 'started', 'pid': process.pid}))
    else:
        if live:
            os.killpg(recorded['pid'], signal.SIGTERM)
            deadline = time.monotonic() + args.stop_timeout
            while identity(recorded['pid']) == recorded['start_ticks']:
                if time.monotonic() >= deadline:
                    os.killpg(recorded['pid'], signal.SIGKILL)
                    break
                time.sleep(0.1)
        elif recorded and identity(recorded['pid']) is not None:
            raise RuntimeError('PID was reused; refusing to stop another process')
        path.unlink(missing_ok=True)
        print(json.dumps({'status': 'stopped'}))


if __name__ == '__main__':
    main()
