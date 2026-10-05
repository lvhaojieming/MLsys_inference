# MoQE Inference

异构量化专家 LLM 推理系统，后端通过 OpenAI-compatible 接口连接 vLLM / vllm-ascend。
训练代码独立维护，本工程只负责在线服务及系统实验。

## 当前实现

- FastAPI Gateway，`/health`、`/v1/models`、`/v1/chat/completions`。
- 静态专家池，按在途请求数选择副本。
- 标准 JSON 与 SSE 流式代理，后端模型名称映射。
- 后端失败返回 502，释放副本计数；请求携带 trace / expert / replica 响应头。
- 配置校验；后端凭据通过 `api_key_env` 引用环境变量。

当前请求的 `model` 显式指定 `awq` 或 `gptq`，尚未接入训练好的 router。
`/health` 只表示 Gateway 存活。副本计数属于单个 Gateway 进程。
尚未实现自动健康摘除、动态卡池、重试、跨专家容错、容量策略和完整指标。
流式响应开始后发生故障会中断，不会重新生成或拼接另一个专家的回答。

## 启动

```bash
python -m pip install -e '.[test]'
moqe-serve --config configs/dev.json --host 127.0.0.1 --port 8000
```

`dev.json` 不连接模型，仅供 Gateway 检查。
复制 `configs/example.json` 为 `configs/local.json`，填入真实专家 endpoint 和后端模型名，
再以该配置启动。示例端口不代表集群实际部署。

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"awq","messages":[{"role":"user","content":"你好"}],"stream":false}'
python -m pytest -q
```

## 目录与开发顺序

| 目录 | 用途 |
| --- | --- |
| `src/moqe_serving/gateway` | HTTP 入口、请求生命周期 |
| `src/moqe_serving/routing` | Router 推理契约，后续接入 checkpoint |
| `src/moqe_serving/pool` | 专家池、副本调度，后续动态成员与健康管理 |
| `src/moqe_serving/backends` | vLLM / Ascend 服务接口 |
| `src/moqe_serving/observability` | 后续指标与完整 trace |
| `src/moqe_serving/offline`、`placement` | 后续离线分析与部署规划 |
| `benchmarks` | 质量、性能、扩缩容和故障实验 |
| `configs/experiments` | 实验配置 |
| `tests`、`docs`、`results` | 测试、技术报告、生成结果 |

四周计划：第 1 周端到端接入及 router；第 2 周两级路由和动态卡池；
第 3 周容错与监控；第 4 周系统实验及结果整理。
技术报告是设计参考，实际功能以代码和测试为准。
