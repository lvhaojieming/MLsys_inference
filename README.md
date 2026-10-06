# MoQE Inference

异构量化专家 LLM 推理系统，后端通过 OpenAI-compatible 接口连接 vLLM / vllm-ascend。
训练代码独立维护，本工程只负责在线服务及系统实验。

## 当前实现

- FastAPI Gateway，`/health`、`/v1/models`、`/v1/chat/completions`。
- 专家池按在途请求数选择 READY 副本，可开启模型验收后入池，并在运行中为相同专家增加副本。
- 标准 JSON 与 SSE 流式代理，后端模型名称映射。
- 后端失败返回 502，释放副本计数；请求携带 trace / expert / replica 响应头。
- 配置校验；后端凭据通过 `api_key_env` 引用环境变量。

请求的 `model` 可显式指定 `awq` 或 `gptq`；配置 router 后，`model=auto` 自动选专家。
Ascend router 使用完整 prompt、冻结 embedding 和训练好的 checkpoint，每条请求只选一次专家。
`/health` 只表示 Gateway 存活。副本计数属于单个 Gateway 进程。
已支持阈值健康摘除、重新验收恢复和开发文档的五状态约束。
尚未实现全新专家池热加入、重试、跨专家容错、容量策略和完整指标。
同专家动态扩容、验收入池和 drain 的配置见 [pool_admission.md](docs/pool_admission.md)。
配置文件驱动的启动、节点与模型路径、同专家扩缩容见 [config_lifecycle.md](docs/config_lifecycle.md)。
vLLM 上下文、并发、内存比例、量化和缓存参数可直接填写 `replicas[].launch.vllm_args`；
启用配置监听后修改参数，会排空并重启该受管理实例，重新验收后入池。
已有统一环境但尚无模型进程时，配置 `nodes[].prepare` 自动检查通信、依赖、模型目录并部署进程 helper。
首次部署与热加入使用同一流程，步骤和配置见 [node_deployment.md](docs/node_deployment.md)。
昇腾卡池的公共环境和适配器使用 `runtime_profiles` 定义，副本通过 `runtime_profile`
引用；配置优先级和热修改行为见 [runtime_profiles.md](docs/runtime_profiles.md)。
流式响应开始后发生故障会中断，不会重新生成或拼接另一个专家的回答。

## 启动

```bash
python -m pip install -e '.[test]'
moqe-serve --config configs/dev.json --host 127.0.0.1 --port 8000
```

`dev.json` 不连接模型，仅供 Gateway 检查。
复制 `configs/example.json` 为 `configs/local.json`，填入真实专家 endpoint 和后端模型名，
再以该配置启动。示例端口不代表集群实际部署。

自动路由在配置中增加 `router` 字段：

```json
{
  "checkpoint": "/path/to/checkpoint_best.pt",
  "tokenizer": "/path/to/router-base-embedding",
  "training_code": "/path/to/MLsys",
  "embedding_model": "/path/to/Qwen3-Embedding-0.6B",
  "device": "npu:0",
  "embedding_graph": true,
  "graph_buckets": [64, 128, 256, 512, 1024],
  "graph_threshold_margin": 0.01,
  "expert_mapping": {
    "qwen3-14b/awq-w4a16/v1": "awq",
    "qwen3-14b/gptq-w4a16/v1": "gptq"
  }
}
```

`training_code` 提供与 checkpoint 一致的模型结构和 embedding 加载器。
V7 使用冻结的 Qwen3-Embedding-0.6B 和 CPU MLP，输出各专家概率，按 checkpoint 中的验证集阈值选择专家。
`embedding_model` 可覆盖 checkpoint 记录的 encoder 路径，便于迁移部署；必须使用训练时同一份模型和 tokenizer。
上述图执行配置适用于 V7。旧架构应移除 `embedding_model` 和三个图参数，
其 embedding 路径仍从 checkpoint 的 `training_config.base_model_path` 读取。
图执行默认关闭；开启后启动时完成所有桶的捕获、预热，再进入服务初始化后续步骤。
超过最大桶长度或接近决策阈值的请求使用原始前向。配置含义、精度验证和性能结果见 [v7_embedding_graph.md](docs/v7_embedding_graph.md)。
Router 配置变更需要重启 Gateway；节点和副本扩缩容仍按生命周期配置热更新。
环境需要兼容的 `torch_npu`、CANN、`transformers` 和 `safetensors`。
`.212` 物理 NPU 1 使用 `ASCEND_RT_VISIBLE_DEVICES=1`，进程内为 `npu:0`。
只启动一个 Gateway worker。模型启动时加载并预热，路由在工作线程中串行执行。

自动路由支持纯文本 system/user/assistant 消息，默认关闭 thinking，`max_tokens` 默认 128。
相同的 canonical chat template 同时传给专家后端；超长 prompt 会拒绝，不截断。
工具、多模态和自定义模板暂不支持。Router 概率不是校准正确率或 regret。
响应头 `X-MoQE-Expert`、`X-MoQE-Replica`、`X-MoQE-Input-Tokens`、`X-MoQE-Router-Ms`
标明决策；Router 耗时包含 tokenize 和前向，不包括等待时间。

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
| `src/moqe_serving/routing` | 在线 Router、checkpoint 加载和 Embedding 图执行 |
| `src/moqe_serving/pool` | 专家池、配置控制、验收、状态机与健康管理 |
| `src/moqe_serving/backends` | vLLM / Ascend 服务接口 |
| `src/moqe_serving/deployment` | 节点预检、helper 部署和统一命令执行 |
| `scripts` | 集群验收与性能实验入口，见 scripts/README.md |
| `configs` | 部署与实验配置模板 |
| `tests`、`docs`、`results` | 测试、技术报告、生成结果 |

技术报告是设计参考，实际功能以代码和测试为准。
