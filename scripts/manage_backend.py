"""Compatibility entry point; deploy the standalone implementation via node.prepare."""
import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] /
                      "src/moqe_serving/deployment/backend_process.py"), run_name="__main__")
