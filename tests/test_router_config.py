"""Deployment configuration rejects graph shapes unsafe to capture."""
import json
from pathlib import Path

import pytest

from moqe_serving.config import Settings


def load_config(tmp_path, **router_overrides):
    example = Path(__file__).resolve().parents[1] / "configs/v7_graph_ttft_cluster.json"
    config = json.loads(example.read_text(encoding="utf-8"))
    config["router"].update(router_overrides)
    path = tmp_path / "cluster.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return Settings.load(path)


def test_graph_deployment_loads_and_can_disable(tmp_path):
    settings = load_config(tmp_path, embedding_model="/new/models/embedding")
    assert settings.router.embedding_graph is True
    assert settings.router.graph_buckets == (64, 128, 256, 512, 1024)
    assert settings.router.embedding_model == "/new/models/embedding"
    assert load_config(tmp_path, embedding_graph=False).router.embedding_graph is False


@pytest.mark.parametrize("overrides", [
    {"embedding_graph": "false"},
    {"graph_buckets": []},
    {"graph_buckets": [128, 64]},
    {"graph_buckets": [64, 64]},
    {"graph_buckets": [True, 128]},
    {"graph_buckets": [64.5, 128]},
    {"graph_buckets": [1]},
    {"graph_threshold_margin": -0.01},
    {"graph_threshold_margin": 0.5},
    {"embedding_model": ""},
])
def test_invalid_graph_configuration_fails_before_startup(tmp_path, overrides):
    with pytest.raises(ValueError):
        load_config(tmp_path, **overrides)
