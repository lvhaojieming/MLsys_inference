import argparse
import asyncio
import json
import logging

import uvicorn

from .config import Settings
from .gateway.app import create_app
from .deployment.prepare import NodePreparer


async def prepare_only(settings):
    reports = {}
    preparer = NodePreparer(settings.lifecycle_log_dir, reports)
    for node in settings.nodes:
        replicas = [r for r in settings.active_replicas if r.node_id == node.id]
        groups = settings.preparation_groups(node, replicas) if node.enabled else []
        await preparer.prepare_groups(groups)
    print(json.dumps({"nodes": reports}, ensure_ascii=False, indent=2))
    return all(report["passed"] for report in reports.values())


def main():
    parser = argparse.ArgumentParser(description="Run the MoQE inference gateway")
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true",
                        help="Check active nodes and deploy helpers without starting models or Gateway")
    args = parser.parse_args()
    settings = Settings.load(args.config)
    logging.basicConfig(level=getattr(logging, settings.gateway.log_level.upper()))
    if args.prepare_only:
        raise SystemExit(0 if asyncio.run(prepare_only(settings)) else 1)
    uvicorn.run(create_app(settings, config_path=args.config), host=args.host or settings.gateway.host,
                port=args.port or settings.gateway.port, log_level=settings.gateway.log_level)


if __name__ == "__main__":
    main()
