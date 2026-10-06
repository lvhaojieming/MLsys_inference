"""Check an existing node environment and deploy the standalone process helper."""
import base64
import json
import logging
from pathlib import Path
import time

from .commands import run_node_command

logger = logging.getLogger(__name__)


RUNTIME_CHECK = """
import importlib, json, pathlib, sys
spec = json.loads(sys.argv[1])
modules = {}
for name in spec['modules']:
    module = importlib.import_module(name)
    modules[name] = str(getattr(module, '__version__', 'available'))
for name in spec['paths']:
    path = pathlib.Path(name)
    if not path.exists():
        raise RuntimeError('Required path missing: ' + name)
    if path.is_dir() and not any(path.iterdir()):
        raise RuntimeError('Required directory empty: ' + name)
print(json.dumps({'modules': modules, 'paths': spec['paths']}))
"""

DEPLOY_HELPER = """
import base64, os, pathlib, sys, tempfile
path = pathlib.Path(sys.argv[1])
source = base64.b64decode(sys.argv[2])
if path.exists() and path.read_bytes() == source:
    print('helper unchanged')
else:
    if path.exists() and not path.read_bytes().startswith(b'# Managed by MoQE\\n'):
        raise RuntimeError('Refusing to overwrite an unmanaged helper: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as output:
            output.write(source)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print('helper deployed')
"""


class NodePreparer:
    def __init__(self, log_dir, reports=None):
        self.log_dir = Path(log_dir)
        self.reports = reports if reports is not None else {}

    async def prepare_groups(self, groups):
        """Share report aggregation between startup, hot edits and the CLI."""
        prepared = {}
        if not groups:
            return prepared
        node_id = groups[0][1].id
        if any(name for name, _, _ in groups):
            report = {"passed": False, "in_progress": True, "profiles": {}}
            self.reports[node_id] = report
            preparer = NodePreparer(self.log_dir, report["profiles"])
            for name, node, members in groups:
                result = await preparer.prepare(node, members, report_key=name or "legacy")
                report["profiles"][name or "legacy"] = result
                prepared.update((r.id, result) for r in members)
            report["passed"] = all(r["passed"] for r in report["profiles"].values())
            report["in_progress"] = False
        else:
            for _, node, members in groups:
                result = await self.prepare(node, members)
                self.reports[node_id] = result
                prepared.update((r.id, result) for r in members)
        return prepared

    async def prepare(self, node, replicas, *, report_key=None):
        options = node.prepare
        report = {"passed": False, "in_progress": True, "started_at_unix": time.time(), "checks": []}
        self.reports[report_key or node.id] = report
        started = time.monotonic()
        phase = "communication"
        async def check(name, command, *, host=False):
            nonlocal phase
            phase = name
            report["current_check"] = name
            logger.info("node_preparation_check node=%s check=%s", node.id, name)
            suffix = "-profile-" + report_key.encode().hex() if report_key else ""
            log_path = self.log_dir / f"node-{node.id.encode().hex()}{suffix}-{name}.log"
            await run_node_command(node, command, log_path=log_path,
                timeout=options.command_timeout_seconds, env={} if host else options.env,
                in_container=not host, environment_scripts=() if host else options.environment_scripts)
            report["checks"].append({"name": name, "log_path": str(log_path)})
        try:
            await check("communication", ["true"], host=True)
            for index, command in enumerate(options.host_checks):
                await check(f"host-{index}", command, host=True)
            paths = list(dict.fromkeys((*options.required_paths,
                *(r.model_path for r in replicas if r.launch and r.model_path))))
            await check("runtime", [options.python, "-c", RUNTIME_CHECK,
                json.dumps({"modules": options.required_modules, "paths": paths})])
            for index, command in enumerate(options.runtime_checks):
                await check(f"runtime-{index}", command)
            source = b"# Managed by MoQE\n" + Path(__file__).with_name("backend_process.py").read_bytes()
            await check("helper", [options.python, "-c", DEPLOY_HELPER, options.helper_path,
                                   base64.b64encode(source).decode("ascii")])
            # Verify both compatibility and executable Python syntax before launch.
            await check("helper-check", [options.python, options.helper_path, "--help"])
            report["passed"] = True
        except Exception as exc:
            report.update(failed_check=phase, error=str(exc) or type(exc).__name__)
        report["in_progress"] = False
        report["current_check"] = None
        report["total_seconds"] = time.monotonic() - started
        return report
