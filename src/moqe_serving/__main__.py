import argparse
import logging
import uvicorn

from .config import Settings
from .gateway.app import create_app


def main():
    parser = argparse.ArgumentParser(description="Run the MoQE inference gateway")
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(Settings.load(args.config)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
