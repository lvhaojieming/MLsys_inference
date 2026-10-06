import argparse
import logging
import uvicorn

from .config import Settings
from .gateway.app import create_app


def main():
    parser = argparse.ArgumentParser(description="Run the MoQE inference gateway")
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    settings = Settings.load(args.config)
    logging.basicConfig(level=getattr(logging, settings.gateway.log_level.upper()))
    uvicorn.run(create_app(settings, config_path=args.config), host=args.host or settings.gateway.host,
                port=args.port or settings.gateway.port, log_level=settings.gateway.log_level)


if __name__ == "__main__":
    main()
