"""One argv-based execution path for local, SSH and container commands."""
import asyncio
import os
from pathlib import Path
import shlex


async def run_node_command(node, command, *, log_path, timeout, env=None, in_container=True,
                           environment_scripts=()):
    env = env or {}
    argv = list(command)
    if in_container and environment_scripts:
        setup = "; ".join("source " + shlex.quote(path) for path in environment_scripts)
        argv = ["bash", "-c", "set -e; " + setup + "; exec " + shlex.join(argv)]
    if node and in_container and node.container:
        argv = ["docker", "exec", node.container, "env", *[f"{k}={v}" for k, v in env.items()], *argv]
    elif node and node.ssh_target and env:
        argv = ["env", *[f"{k}={v}" for k, v in env.items()], *argv]
    if node and node.ssh_target:
        argv = ["ssh", *node.ssh_options, node.ssh_target, shlex.join(argv)]
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as output:
        process = await asyncio.create_subprocess_exec(*argv, stdout=output, stderr=output,
                                                       env={**os.environ, **env})
        try:
            await asyncio.wait_for(process.wait(), timeout)
        except BaseException:
            if process.returncode is None:
                process.kill()
                await process.wait()
            raise
        if process.returncode:
            raise RuntimeError(f"Command exited {process.returncode}; see {path}")
