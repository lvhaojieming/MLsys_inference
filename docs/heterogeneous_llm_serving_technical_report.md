# 异构量化专家 LLM 推理系统
## 技术报告与开发手册

**版本**：v1.0  
**项目类型**：Heterogeneous LLM Serving / Precision-Diverse Expert Serving  
**系统前提**：Router 已完成训练并拥有可加载 checkpoint  
**目标后端**：vLLM、vllm-ascend 及 OpenAI-compatible 异构推理后端  
**核心模型主线**：Qwen3-14B  
**规模验证主线**：Qwen3-8B  
**核心实验集群**：H24，后续扩展至 D40 / S80 / F108

---

# 1. 文档目的

本手册用于指导异构量化专家 LLM 推理系统从代码仓库创建、模块开发、单元测试、端到端联调，到 H24 正式实验和论文结果产出的完整过程。

系统不试图把不同硬件强行组成同一个 Tensor Parallel 或 Pipeline Parallel 执行组，而是把同一基座模型的不同量化版本部署为多个相互独立的专家池：

```text
Expert A → Hardware Pool A
Expert B → Hardware Pool B
Expert C → Hardware Pool C
```

每个请求只执行一个专家。

在线系统根据：

```text
Router 预测的请求级质量
+
专家池当前容量 / 排队状态
+
可选的 session / prefix-cache 成本
```

决定请求最终进入哪个专家池。

系统采用三条设计原则：

1. 不修改 vLLM core；
2. 使用“池间 + 池内”两级调度；
3. 决策逻辑集中在 Gateway，底层池继续通过标准服务接口运行。

因此，本项目的核心研发对象不是 vLLM kernel，而是：

```text
Gateway
Router Runtime
Expert Selector
Capacity-aware Scheduler
Pool Registry
Replica Scheduler
Health Manager
Retry / Failover
Observability
Benchmark Harness
Placement Planner
```

---

# 2. 系统目标与非目标

## 2.1 系统目标

系统最终必须实现：

```text
Client
  ↓
Unified Gateway
  ↓
Router Runtime
  ↓
Expert Selection
  ↓
Capacity-aware Routing
  ↓
Homogeneous Expert Pool
  ↓
Replica Scheduler
  ↓
vLLM / vllm-ascend / vendor backend
  ↓
Streaming Response
```

并能够完成：

- 请求级量化专家选择；
- 同一专家的多副本负载均衡；
- 实例健康检查；
- 动态摘除和恢复；
- 首 token 前故障重试；
- 跨专家 fallback；
- 请求级完整 trace；
- 开环负载生成；
- 突发流量实验；
- 节点故障实验；
- quality-goodput Pareto 实验；
- H24 → D40 → S80 → F108 扩展。

## 2.2 第一阶段非目标

第一版不要立即实现：

```text
Kubernetes Operator
复杂服务发现
跨池 KV migration
跨厂商 Tensor Parallel
跨池 Prefill/Decode 分离
自动 Router 重训练
全自动 Placement Optimizer
```

这些会显著扩大工程面，却不是第一条论文 Hero Curve 成立的必要条件。

---

# 3. 系统总体架构

完整系统分为五个逻辑平面：

```text
                          ┌─────────────────────────┐
                          │      Offline Plane      │
                          │                         │
                          │ Hardware Profiler       │
                          │ Loss/Regret Dataset     │
                          │ Placement Planner       │
                          │ Shadow Labeler          │
                          └───────────┬─────────────┘
                                      │
                                      ▼

┌─────────┐       ┌───────────────────────────────────────┐
│ Client  │──────▶│              Gateway                  │
└─────────┘       │                                       │
                  │ API / Validation                      │
                  │ Request Context                       │
                  │ Tokenization                          │
                  │ Router Runtime                        │
                  │ Expert Selector                       │
                  │ Capacity Policy                       │
                  │ Replica Scheduler                     │
                  │ Retry / Failure Handling              │
                  └───────────────┬───────────────────────┘
                                  │
             ┌────────────────────┼────────────────────┐
             │                    │                    │
             ▼                    ▼                    ▼
       ┌──────────┐         ┌──────────┐         ┌──────────┐
       │ AWQ Pool │         │GPTQ Pool │         │INT8 Pool │
       │ 4090 ... │         │ A800 ... │         │910B ...  │
       └────┬─────┘         └────┬─────┘         └────┬─────┘
            │                    │                    │
       vLLM replicas        vLLM replicas      vllm-ascend
            │                    │                    │
            └────────────────────┼────────────────────┘
                                 ▼
                       Metrics / Event Logs
```

一条请求生命周期：

```text
收到请求
→ tokenizer
→ affinity 查询
→ Router 输出 regret
→ 选池
→ 池内选副本
→ 流式代理
→ 更新 workload / trace
```

first-token 前失败和流中失败具有不同处理语义，因为不同专家之间不能迁移 KV cache。

---

# 4. 核心设计原则

## 4.1 Router 和调度器严格解耦

Router 只负责：

\[
x\rightarrow\hat{\mathbf r}(x)
\]

其中：

\[
\hat{\mathbf r}(x)
=
[
\hat r_1(x),
\hat r_2(x),
\dots,
\hat r_K(x)
].
\]

它不应该直接决定最终硬件实例。

例如：

```json
{
  "awq": 0.018,
  "gptq": 0.004,
  "int8": 0.011
}
```

Expert Selector 再结合系统状态做选择。

---

## 4.2 Expert Selection 与 Replica Selection 分离

第一级：

```text
选择 AWQ / GPTQ / INT8
```

第二级：

```text
选 AWQ Pool 里的 4090-0 / 4090-1 / ...
```

因此：

```text
ExpertSelector != PoolScheduler
```

这是系统中非常重要的抽象边界。

---

## 4.3 硬件厂商差异封装在 Backend Layer

上层不能写：

```python
if gpu == "910B":
    ...
elif gpu == "A800":
    ...
```

而应该统一：

```python
await backend.stream_chat(instance, request)
```

因此：

```text
NVIDIA vLLM
Ascend vLLM
Kunlun backend
MTT backend
Iluvatar backend
```

只体现为不同 Backend Adapter。

---

## 4.4 所有专家常驻

正常请求路径不能出现：

```text
收到请求
→ 卸载 Expert A
→ 加载 Expert B
→ 推理
```

而应该：

```text
AWQ 常驻
GPTQ 常驻
INT8 常驻
```

从而避免 cold switching。

---

# 5. 推荐代码仓库结构

```text
hetero-llm-serving/
│
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
│
├── gateway/
│   ├── __init__.py
│   ├── app.py
│   ├── schemas.py
│   ├── request_context.py
│   ├── lifecycle.py
│   ├── retry.py
│   ├── error_handler.py
│   └── dependencies.py
│
├── routing/
│   ├── __init__.py
│   ├── router_runtime.py
│   ├── router_types.py
│   ├── expert_selector.py
│   ├── capacity_policy.py
│   ├── affinity.py
│   │
│   └── policies/
│       ├── __init__.py
│       ├── quality_only.py
│       ├── least_loaded.py
│       ├── capacity_aware.py
│       └── oracle.py
│
├── pool/
│   ├── __init__.py
│   ├── models.py
│   ├── capability.py
│   ├── registry.py
│   ├── scheduler.py
│   ├── load_tracker.py
│   ├── health.py
│   └── controller.py
│
├── backends/
│   ├── __init__.py
│   ├── base.py
│   ├── vllm_client.py
│   ├── ascend_client.py
│   ├── generic_openai_client.py
│   └── fake_backend.py
│
├── observability/
│   ├── __init__.py
│   ├── metrics.py
│   ├── request_record.py
│   ├── event_logger.py
│   └── tracing.py
│
├── offline/
│   ├── __init__.py
│   ├── profile_instances.py
│   ├── build_loss_matrix.py
│   ├── build_regret_matrix.py
│   ├── evaluate_router.py
│   └── shadow_labeler.py
│
├── placement/
│   ├── __init__.py
│   ├── candidate_builder.py
│   ├── cost_model.py
│   ├── solver.py
│   └── planner.py
│
├── benchmarks/
│   ├── __init__.py
│   ├── compatibility_test.py
│   ├── workload.py
│   ├── arrival.py
│   ├── load_test.py
│   ├── fault_injection.py
│   ├── replay_trace.py
│   ├── quality_eval.py
│   └── analyze_results.py
│
├── configs/
│   ├── base.yaml
│   ├── dev.yaml
│   ├── n16.yaml
│   ├── h24.yaml
│   ├── d40.yaml
│   ├── s80.yaml
│   ├── f108.yaml
│   ├── router.yaml
│   ├── workloads.yaml
│   │
│   └── experiments/
│       ├── e0_compatibility.yaml
│       ├── e1_complementarity.yaml
│       ├── e2_profile.yaml
│       ├── e3_pareto.yaml
│       ├── e4_cross_ablation.yaml
│       ├── e5_add_pool.yaml
│       ├── e6_failure.yaml
│       └── e9_overhead.yaml
│
├── scripts/
│   ├── start_gateway.sh
│   ├── stop_gateway.sh
│   ├── smoke_test.sh
│   ├── health_check.sh
│   ├── start_fake_backends.sh
│   ├── run_load_test.sh
│   └── run_fault_test.sh
│
├── deployments/
│   ├── docker/
│   ├── systemd/
│   └── examples/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── artifacts/
│   ├── router/
│   ├── profiles/
│   ├── regret/
│   └── placement/
│
├── logs/
│   ├── gateway/
│   ├── requests/
│   └── experiments/
│
└── results/
    ├── raw/
    ├── processed/
    ├── figures/
    └── tables/
```

---

# 6. 模块职责总表

| 模块 | 核心文件 | 职责 |
|---|---|---|
| Gateway | `gateway/app.py` | HTTP/OpenAI-compatible API |
| 生命周期 | `gateway/lifecycle.py` | 串联 Router、Selector、Pool、Backend |
| 请求上下文 | `gateway/request_context.py` | 保存单请求运行状态 |
| Router | `routing/router_runtime.py` | 加载已训练 checkpoint，输出 score |
| Expert Selector | `routing/expert_selector.py` | 根据 Router + 系统状态选择 expert |
| Capacity Policy | `routing/capacity_policy.py` | 计算拥塞价格 |
| Registry | `pool/registry.py` | 管理实例和状态 |
| Pool Scheduler | `pool/scheduler.py` | Expert 内选 replica |
| Load Tracker | `pool/load_tracker.py` | 维护实时 outstanding workload |
| Health | `pool/health.py` | 健康探测和状态转换 |
| Controller | `pool/controller.py` | add/drain/remove/recover |
| Backend | `backends/*.py` | 与实际推理服务交互 |
| Retry | `gateway/retry.py` | 故障重试与 fallback |
| Metrics | `observability/metrics.py` | 聚合指标 |
| Event Log | `event_logger.py` | 请求级 JSONL trace |
| Benchmark | `benchmarks/*` | 压测、故障、质量评测 |
| Offline Profiler | `offline/profile_instances.py` | 建立硬件性能画像 |
| Planner | `placement/*` | 离线专家/副本放置 |

---

# 7. 核心数据模型

## 7.1 InstanceState

```python
class InstanceState(str, Enum):
    STARTING = "starting"
    READY = "ready"
    DRAINING = "draining"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"
```

标准状态机：

```text
STARTING
    ↓
 READY
 ↙   ↘
DRAINING   UNHEALTHY
   ↓          ↓
OFFLINE     READY
```

---

## 7.2 InstanceInfo

```python
class InstanceInfo(BaseModel):
    instance_id: str

    expert_id: str

    model_name: str
    model_revision: str

    quantization: str
    backend_type: str

    node_id: str
    device_ids: list[int]

    endpoint: str
    accelerator_type: str

    state: InstanceState

    max_concurrency: int

    inflight_requests: int = 0
    inflight_tokens: int = 0

    throughput_tokens_per_sec: float | None = None

    metadata: dict = {}
```

---

# 8. Router Runtime 规范

Router 已经完成训练，因此在线 Runtime 不允许包含：

```text
optimizer
dataset
backward
gradient
training loop
```

只做：

```text
load
eval
inference
```

统一接口：

```python
@dataclass
class RouterOutput:
    expert_scores: dict[str, float]
    latency_ms: float
    router_version: str
```

调用：

```python
result = router.route(prompt)
```

输出：

```python
RouterOutput(
    expert_scores={
        "awq": 0.021,
        "gptq": 0.006,
        "int8": 0.015,
    },
    latency_ms=4.7,
    router_version="router-v1"
)
```

---

# 9. Router 启动校验

Router checkpoint 必须携带 metadata：

```json
{
  "router_version": "router-v1",
  "model_name": "Qwen3-14B",
  "model_revision": "...",
  "tokenizer_revision": "...",
  "expert_ids": [
    "awq",
    "gptq",
    "int8"
  ]
}
```

系统启动时必须检查：

```python
assert checkpoint.expert_ids == config.expert_ids
assert checkpoint.model_name == system.model_name
```

特别需要防止：

```text
训练：
0 = AWQ
1 = GPTQ
2 = INT8

线上：
0 = GPTQ
1 = AWQ
2 = INT8
```

这种错误不会 crash，却会让实验完全失效。

---

# 10. Gateway 设计

`gateway/app.py` 应当保持非常薄。

```python
@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    return await lifecycle.handle_chat(request)
```

建议 API：

```text
POST /v1/chat/completions
POST /v1/completions

GET /health
GET /ready
GET /metrics

GET /debug/pools
GET /debug/instances
```

Debug API 在正式实验环境应可关闭。

---

# 11. RequestContext

每次请求进入 Gateway 后创建：

```python
@dataclass
class RequestContext:
    request_id: str
    arrival_ts: float

    input_tokens: int = 0
    estimated_output_tokens: int = 0

    router_scores: dict | None = None

    selected_expert: str | None = None
    selected_instance: str | None = None

    retry_count: int = 0

    router_ms: float = 0
    queue_ms: float = 0
    ttft_ms: float = 0
    e2e_ms: float = 0
```

整个调用链只传：

```text
request
+
context
```

避免不同模块自行生成不一致统计。

---

# 12. Request Lifecycle

`gateway/lifecycle.py` 是在线系统的总编排器。

完整流程：

```text
1 Receive Request
       ↓
2 Validate
       ↓
3 Build Prompt
       ↓
4 Tokenize
       ↓
5 Router Runtime
       ↓
6 Obtain Expert Scores
       ↓
7 Expert Selector
       ↓
8 Registry 查询健康实例
       ↓
9 Pool Scheduler 选择 Replica
       ↓
10 LoadTracker.acquire()
       ↓
11 Backend.stream_chat()
       ↓
12 First Token
       ↓
13 Streaming
       ↓
14 Completion
       ↓
15 LoadTracker.release()
       ↓
16 Metrics
       ↓
17 RequestRecord
```

正常情况下 Gateway 只进行一次 Router inference。

如果某个 pool 后续失败，应复用已有 score vector，而不是重新运行 Router。

---

# 13. Expert Selection

## 13.1 Quality-only Baseline

对应原始 Router：

\[
e^*
=
\arg\min_e \hat r_e(x).
\]

实现：

```python
class QualityOnlyPolicy:
    def select(self, scores, state, ctx):
        return min(scores, key=scores.get)
```

它必须保留。

因为这是系统论文的重要 baseline：

```text
Original MoQE
```

---

# 14. Least-loaded Baseline

只使用当前 load：

\[
e^*
=
\arg\min_e P_e(t).
\]

这个 baseline 用于回答：

> 如果完全不考虑请求级质量，只做普通异构负载均衡，会怎样？

---

# 15. Capacity-aware Routing

完整方法基本形式：

\[
S_e(x,t)
=
V\hat r_e(x)
+
P_e(t).
\]

加入缓存后：

\[
S_e(x,t)
=
V\hat r_e(x)
+
P_e(t)
+
\kappa C_e(x).
\]

其中：

\[
\hat r_e(x)
\]

为 Router 输出的预计质量 regret；

\[
P_e(t)
\]

为实时拥塞成本；

\[
C_e(x)
\]

为 session/prefix cache 切换成本。

---

# 16. Load Tracking

第一版可以记录：

\[
N_e^{\text{inflight}}.
\]

但正式系统不要只用请求数量。

更合理的是：

\[
W_e(t)
=
\text{Outstanding Tokens}.
\]

例如：

```text
512 → 128
```

不能和：

```text
6144 → 2048
```

视为相同的一个 request。

因此推荐：

\[
P_e(t)
=
\frac{W_e(t)}{\mu_e},
\]

其中：

\[
\mu_e
=
\text{该 pool 的 measured tokens/s}.
\]

容量与请求长度分布有关，而不是一个绝对不变的常数。

---

# 17. Pool Registry

`registry.py` 只保存状态，不做调度。

标准接口：

```python
registry.add(instance)

registry.remove(instance_id)

registry.get(instance_id)

registry.ready_instances(expert_id)

registry.ready_experts()
```

不要：

```python
registry.select_best_instance()
```

选择属于 Scheduler。

---

# 18. Pool Scheduler

已确定 expert：

```text
INT8
```

假设存在：

```text
910B-0
910B-1
910B-2
910B-3
```

Pool Scheduler 再选择具体执行实例。

第一阶段实现：

```text
Round Robin
Least Inflight
```

正式主实验建议再增加：

```text
Least Outstanding Work
```

即：

\[
j^*
=
\arg\min_j W_j.
\]

---

# 19. LoadTracker

`load_tracker.py` 负责所有计数器。

标准生命周期：

```python
load_tracker.acquire(instance, ctx)

try:
    ...
finally:
    load_tracker.release(instance, ctx)
```

绝对不要在：

```text
Gateway
Scheduler
Backend
```

三个地方分别维护 inflight。

否则最终一定出现状态漂移。

---

# 20. Backend 抽象

统一基类：

```python
class BackendClient(ABC):

    async def health(self, instance):
        ...

    async def stream_chat(self, instance, payload):
        ...

    async def cancel(self, instance, request_id):
        ...

    async def metrics(self, instance):
        ...
```

上层只依赖 BackendClient。

---

# 21. vLLM Backend

`vllm_client.py`

负责：

```text
OpenAI-compatible request
Streaming response
Connection timeout
Read timeout
Cancellation
Backend errors
```

---

# 22. Ascend Backend

如果 `vllm-ascend` 暴露相同协议：

```python
class AscendClient(VLLMClient):
    pass
```

只有真正不兼容的接口才 override。

不要复制一份整个 client。

---

# 23. Capability Layer

不同 vendor backend 能力可能不同。

因此：

```python
class BackendCapabilities(BaseModel):
    streaming: bool
    cancellation: bool

    prefix_cache: bool
    priority: bool

    prompt_logprobs: bool

    metrics_level: str
```

例如：

```yaml
capabilities:
  streaming: true
  cancellation: true
  prefix_cache: true
  priority: true
  prompt_logprobs: true
  metrics_level: full
```

这样上层根据能力降级，而不是假定所有厂商完全等价。

---

# 24. Health Manager

建议每 2 秒探测一次。

不能一次失败马上：

```text
READY → DEAD
```

推荐：

```text
连续失败 3 次
→ UNHEALTHY
```

恢复：

```text
连续成功 2 次
→ READY
```

Scheduler 永远只能看到：

```text
READY
```

实例。

---

# 25. Controller

提供：

```python
add_instance()

enable_instance()

drain_instance()

remove_instance()

recover_instance()
```

其中 drain 语义：

```text
READY
 ↓
DRAINING
 ↓
不再接收新请求
 ↓
等待 inflight == 0
 ↓
OFFLINE
```

不要把：

```text
drain
```

等价成：

```text
kill -9
```

两者用于不同实验。

---

# 26. Retry 与故障语义

这是必须在编码前就确定的协议。

## Case A：first token 前失败

允许：

```text
Instance A
   ↓ fail
Same Expert / Instance B
```

如果整个 expert 不可用：

```text
重新从剩余 expert 中选择 next best
```

---

## Case B：已经返回 token 后失败

默认禁止透明续生成。

原因是：

```text
Expert A KV cache
≠
Expert B KV cache
```

因此：

```text
stream failure
→ 记录 partial failure
→ 终止请求
```

或者：

```text
restart whole request
```

但 restart 必须成为可观测事件。

---

# 27. Fake Backend

必须开发。

否则测试 Gateway 每次都需要 GPU/NPU。

需要支持：

```text
Normal
Fixed Latency
Random Latency
Slow TTFT
Slow Decode
HTTP 500
Timeout
Connection Failure
Stream Interruption
```

例如：

```text
fake://healthy
fake://slow
fake://fail-before-token
fake://fail-after-5-tokens
```

绝大部分 failure test 首先在 Fake Backend 上跑。

---

# 28. Observability

系统论文不能最后再补监控。

从 MVP 开始必须同时记录：

## 聚合指标

```text
requests_total
request_failures_total

router_latency

expert_selected_total

instance_inflight_requests
instance_inflight_tokens

pool_workload
pool_capacity

ttft
tpot
e2e_latency

retries
fallbacks
stream_failures
```

## Request-level Trace

每条请求一行 JSONL。

---

# 29. Request Record 格式

推荐：

```json
{
  "request_id": "req-001",
  "workload_id": "gsm8k-test",

  "arrival_ts": 12345.1,

  "input_tokens": 521,
  "output_tokens": 134,

  "router_scores": {
    "awq": 0.01,
    "gptq": 0.04,
    "int8": 0.02
  },

  "selected_expert": "awq",
  "selected_instance": "4090-2",

  "pool_work_snapshot": {
    "awq": 1823,
    "gptq": 493,
    "int8": 2381
  },

  "router_ms": 4.2,
  "queue_ms": 17.5,
  "ttft_ms": 201.3,
  "e2e_ms": 1841.5,

  "retry_count": 0,
  "status": "success"
}
```

---

# 30. 配置管理

实例和实验参数绝对不能硬编码进 Python。

例如：

```text
configs/h24.yaml
```

```yaml
system:
  model: Qwen3-14B

router:
  checkpoint: artifacts/router/router.pt
  device: cuda:0

  expert_ids:
    - awq
    - gptq
    - int8

routing:
  policy: capacity_aware
  V: 10.0
  cache_kappa: 0.0

scheduler:
  replica_policy: least_work

health:
  interval_sec: 2
  failure_threshold: 3
  recovery_threshold: 2

retry:
  same_expert_retries: 1
  cross_expert_fallback: true

instances:

  - instance_id: a800-gptq-0
    expert_id: gptq
    node_id: node-a800
    device_ids: [0]
    accelerator: A800
    backend: vllm
    endpoint: http://10.0.0.11:8000

  - instance_id: rtx4090-awq-0
    expert_id: awq
    node_id: node-4090
    device_ids: [0]
    accelerator: RTX4090
    backend: vllm
    endpoint: http://10.0.0.21:8000

  - instance_id: ascend-int8-0
    expert_id: int8
    node_id: ascend-01
    device_ids: [0]
    accelerator: Ascend910B
    backend: vllm-ascend
    endpoint: http://10.0.0.31:8000
```

---

# 31. 实验配置与系统配置分离

例如：

```text
configs/experiments/e3_pareto.yaml
```

```yaml
experiment:
  id: E3-H24-Pareto

cluster: h24

arrival_rates:
  - 1
  - 2
  - 4
  - 6
  - 8
  - 10

policies:
  - best_single
  - least_loaded
  - quality_only
  - capacity_aware

seeds:
  - 1
  - 2
  - 3
```

这样同一个系统代码可以运行所有 baseline。

---

# 32. Benchmark Harness

Benchmark 必须是系统的一等公民。

目录：

```text
benchmarks/
```

负责：

```text
workload construction
arrival generation
HTTP sending
failure events
result collection
quality evaluation
```

---

# 33. Open-loop Load Generator

正式 benchmark 必须采用：

```text
预先生成 arrival timestamps
```

而不是：

```text
上一请求完成
→ 下一请求才发
```

即：

```python
for request in arrivals:
    await sleep_until(request.timestamp)
    asyncio.create_task(send(request))
```

支持：

```text
Constant
Poisson
Burst
Trace Replay
```

---

# 34. Performance Profiling

正式系统实验前，需要建立：

\[
\mu_{e,h}
(
L_\text{input},
L_\text{output},
C
).
\]

建议至少测试四种 workload shape：

| 类型 | Input | Output | 目标 |
|---|---:|---:|---|
| Short | 512 | 128 | TTFT / Router overhead |
| Medium | 2048 | 512 | 常规服务 |
| Long input | 6144 | 256 | Prefill |
| Long output | 512 | 2048 | Decode |

并发：

\[
1,4,8,16,32.
\]

---

# 35. Compatibility Matrix

任何配置进入正式调度候选集合前都必须经过验收。

记录：

```text
Base model
Quant checkpoint
Hardware
Backend/version
TP/PP
Context length
KV dtype
Concurrency
Memory
Quality check
```

最终得到：

| Expert | HW | Backend | TP | Run | Stream | Context | Concurrency | HBM |
|---|---|---|---:|---|---|---:|---:|---:|
| AWQ | 4090 | vLLM | 1 | ✓ | ✓ | 8192 | 32 | ... |
| GPTQ | A800 | vLLM | 1 | ✓ | ✓ | 8192 | 32 | ... |
| INT8 | 910B | vllm-ascend | 1 | ✓ | ✓ | 8192 | ... | ... |

“模型能够启动”不能视为通过验收；必须完成生成、流式、高并发和目标上下文测试。

---

# 36. Unit Test 设计

至少包括：

```text
test_registry.py
test_scheduler.py
test_selector.py
test_load_tracker.py
test_health.py
test_retry.py
```

例如：

## Registry

验证：

```text
READY instance 可查询
UNHEALTHY 不返回
DRAINING 不接新请求
```

## Scheduler

验证：

```text
least-inflight 真的选择最空实例
```

## Retry

验证：

```text
first-token 前允许重试
first-token 后禁止透明 failover
```

---

# 37. Integration Test

使用 Fake Backend。

测试：

```text
Gateway
→ Router
→ Selector
→ Registry
→ Scheduler
→ Fake Backend
```

场景：

```text
Normal
Slow backend
Backend 500
Pool partially dead
Whole expert pool dead
Stream interruption
```

---

# 38. E2E Test

再接真实设备。

最小环境：

```text
1 × 4090
1 × A800
1 × 910B
```

实现：

```text
AWQ
GPTQ
INT8
```

此时不需要 24 卡。

先跑：

\[
1000
\]

条请求。

通过条件：

```text
Router 自动选择 Expert
正确选择 Backend
流式正常
无资源计数泄漏
trace 完整
无未解释系统错误
```

---

# 39. 开发里程碑

## M0 — Fake Backend

完成：

```text
normal
slow
failure
stream failure
```

验收：

```text
pytest tests/unit
```

全部通过。

---

## M1 — 单 Backend Gateway

链路：

```text
Client
→ Gateway
→ vLLM
```

验收：

```text
流式结果与直连基本一致
```

---

## M2 — Registry + Replica Scheduler

链路：

```text
Gateway
→ one Expert
→ multiple replicas
```

验收：

```text
RR / least-inflight 正确
```

---

## M3 — Multi-Expert

加入：

```text
AWQ
GPTQ
INT8
```

先支持 debug 强制 expert。

---

## M4 — Router Integration

链路：

```text
Prompt
→ Router
→ Expert
→ Backend
```

验收：

\[
1000
\]

条请求自动选择，无人工干预。

---

## M5 — Health + Retry

验收：

```text
kill one instance
→ 自动摘除
→ 请求继续

recover
→ 自动加入
```

---

## M6 — Observability

保证每次请求都有：

```text
router score
expert
instance
queue
TTFT
E2E
status
```

---

## M7 — Open-loop Benchmark

支持：

```text
Poisson
burst
constant
```

---

## M8 — Capacity-aware Routing

正式实现：

\[
V\hat r+P.
\]

到这里开始产生真正论文结果。

---

## M9 — Advanced System

再加入：

```text
affinity
placement
shadow labeling
dynamic controller
```

---

# 40. 第一轮正式 Baseline

H24 正式实验至少保留：

| 方法 | 含义 |
|---|---|
| Best Single | 最优单 Expert |
| Hardware-aware Least Loaded | 每硬件最佳模型 + 普通负载均衡 |
| Original MoQE | 只看 Router quality |
| Quality-threshold Fastest | 满足质量条件选最快 |
| Generic Joint Routing | 通用 quality-latency 方法 |
| Ours | Quality + Capacity + Placement |

---

# 41. 最重要的主实验

## Experiment A — Quality–Goodput Pareto

横轴：

\[
\text{Goodput@SLO}
\]

纵轴：

\[
\text{Quality}.
\]

逐渐提高：

\[
\lambda.
\]

比较：

```text
Best Single
Least Loaded
Original MoQE
Ours
Oracle
```

这是整篇系统论文最重要的结果。

---

# 42. Deployment × Routing 二乘二实验

固定 H24：

| | Simple Routing | Ours Routing |
|---|---|---|
| Hardware-aware Placement | A | B |
| Complementarity-aware Placement | C | D |

从而区分：

```text
Placement contribution
Routing contribution
Interaction
```

---

# 43. Add-Pool Experiment

依次：

```text
base pool
→ + A800
→ + 4090
→ + 910B
→ + other vendors
```

观察：

```text
Goodput
Quality
Pool traffic share
Queue
```

目标是验证：

> 添加异构/低成本资源可以增加服务容量，而 quality-aware routing 避免精度按容量比例直接下降。

---

# 44. Failure Experiment

时间线示例：

```text
t=0      normal
t=120    drain 910B-3
t=180    kill 910B-4
t=240    restore 910B-3
t=300    restore 910B-4
```

画：

```text
Pool capacity
Pool workload
Price
Traffic share
P99 TTFT
Accuracy
```

---

# 45. 实验数据集

正式质量任务建议：

```text
GSM8K
HumanEval+
MBPP+
IFEval
CMMLU
```

Long context 独立增加：

```text
LongBench
```

重要原则：

> 在线系统质量必须由实际经过 Gateway → Router → Pool → Backend 返回的答案计算。

离线 loss/regret matrix 只能用于：

```text
Router analysis
Oracle
Training
```

不能替代在线 accuracy。

---

# 46. 实验环境分层

## N16

```text
A800 ×8
4090 ×8
```

用途：

```text
同一生态基础验证
```

## H24

```text
A800 ×8
4090 ×8
910B ×8
```

用途：

```text
完整 baseline
Pareto
failure
ablation
```

## D40

用于：

```text
跨 vendor / 跨 software stack
```

## S80

用于：

```text
24 → 32 → 48 → 80
```

scale-out。

## F108

只跑：

```text
Ours
Strongest Baseline
Optional second baseline
```

不要重新跑全部笛卡尔积。

---

# 47. 实验结果目录

例如：

```text
results/
└── E3_pareto_h24/
    │
    ├── config_snapshot.yaml
    ├── git_commit.txt
    │
    ├── seed_1/
    │   ├── requests.jsonl
    │   ├── responses.jsonl
    │   └── metrics.json
    │
    ├── seed_2/
    ├── seed_3/
    │
    ├── summary.csv
    │
    └── figures/
        └── quality_goodput.pdf
```

每次实验必须记录：

```text
Git commit
Router version
Model revision
Tokenizer revision
Backend versions
Hardware config
Experiment config
Random seed
Start/end time
```

---

# 48. 公平性要求

不同方法必须共享：

```text
同一批请求
同一 arrival trace
同一 expert candidates
同一硬件库存
同一 backend 配置
同一 generation settings
```

不能：

```text
Method A 用自己最佳负载
Method B 用自己最佳负载
```

然后只比较 normalized throughput。

正式比较应使用相同绝对 arrival rate。

---

# 49. 推理设置必须统一

正式实验固定：

```text
tokenizer
chat template
thinking switch
generation parameters
stop criteria
context limit
output limit
```

否则不同 backend 的模型行为差异可能污染系统结论。

---

# 50. 在线与离线任务隔离

正式 H24 benchmark 期间：

```text
不要同时量化模型
不要训练 Router
不要生成 Shadow Labels
不要跑其他 GPU/NPU 作业
```

否则：

```text
background load
```

会污染 TTFT / P99 / throughput。

---

# 51. 性能指标

主指标：

\[
\text{Goodput@SLO}
\]

同时分别报告：

```text
Accuracy
TTFT P50
TTFT P95
TTFT P99
TPOT P50
TPOT P99
E2E P50/P99
Request failure rate
Reject rate
Pool utilization
Expert distribution
```

不要只有一个 composite score。

---

# 52. Router Overhead

Router 必须单独 profile：

```text
128 token
512
2048
4096
8192
```

测：

```text
P50
P95
P99
```

并同时记录完整链路：

\[
T_{\text{E2E}}
=
T_{\text{gateway}}
+
T_{\text{router}}
+
T_{\text{queue}}
+
T_{\text{backend}}.
\]

Router 如果独占额外 accelerator，也必须计入系统资源成本。

---

# 53. 系统风险

## 53.1 Queue signal lag

不要完全依赖：

```text
Prometheus polling
```

调度热路径应以 Gateway 本地 outstanding work 为主。

Monitoring 数据用于校正。

---

## 53.2 Counter Leak

任何：

```python
acquire()
```

必须对应：

```python
finally:
    release()
```

---

## 53.3 Retry Storm

同一个请求：

```text
max retry
```

必须有限。

整个 pool failure 时避免无限跨 Expert 尝试。

---

## 53.4 Router Mapping Error

这是最危险的 silent error。

启动必须验证：

```text
Router output order == System expert order
```

---

## 53.5 Cache Contamination

重复实验可能因为 prefix cache 命中导致吞吐异常提高。

需要把：

```text
no sharing
natural prefix
multi-turn
```

分开测试。

---

## 53.6 Gateway 成为瓶颈

正式测试前单独测试：

```text
Gateway max QPS
Gateway streaming bandwidth
CPU usage
network bandwidth
```

确保模型后端先达到瓶颈。

---

# 54. 开发顺序

正式开发按照以下顺序，不建议调整：

```text
Phase 1
pool/models.py
pool/registry.py
backends/base.py
backends/fake_backend.py
backends/vllm_client.py

Phase 2
pool/scheduler.py
pool/load_tracker.py
routing/router_types.py
routing/router_runtime.py
routing/expert_selector.py

Phase 3
gateway/schemas.py
gateway/request_context.py
gateway/lifecycle.py
gateway/app.py

Phase 4
pool/health.py
gateway/retry.py
observability/metrics.py
observability/event_logger.py

Phase 5
benchmarks/arrival.py
benchmarks/load_test.py
benchmarks/fault_injection.py

Phase 6
routing/capacity_policy.py
routing/policies/capacity_aware.py

Phase 7
routing/affinity.py
pool/controller.py

Phase 8
placement/*
offline/shadow_labeler.py
```

Router 已经训练好的情况下，不需要再把 Router training 放在主关键路径。

---

# 55. 第一阶段成功标准

不要以：

```text
代码全部写完
```

作为成功标准。

第一个真正 milestone 应该是：

> 三种真实 Expert、三个真实后端实例、1000 个真实请求能够经过 Router 自动选择 → Expert Pool → Backend → Streaming Response，并完整记录每条请求的 Router Score、Expert、Instance、TTFT、E2E 和状态，且没有资源计数泄漏。

完成以后打：

```text
v0.1.0
```

tag。

---

# 56. 第二阶段成功标准

增加：

```text
multiple replicas
health
retry
open-loop load
capacity-aware routing
```

然后能够产生第一张：

\[
\boxed{
\text{Quality vs Goodput@SLO}
}
\]

曲线。

一旦：

```text
Original MoQE
Least-loaded
Ours
```

三条曲线能够稳定复现，就意味着系统论文核心路径已经成立。

---

# 57. 第三阶段成功标准

扩展到 H24：

```text
A800 ×8
4090 ×8
910B ×8
```

完成：

```text
完整 baseline
Pareto
2×2 routing/placement ablation
add-pool
fault
router overhead
```

然后再考虑 D40 / S80。

第一批最关键产物不是“全部 108 卡上线”，而是：

```text
真实可执行配置矩阵
逐请求专家损失矩阵
H24 Quality–Goodput Curve
```

三项成立之后再扩集群。

---

# 58. 最终核心代码清单

整个项目文件很多，但真正的在线核心只有：

```text
gateway/
    app.py
    lifecycle.py
    retry.py

routing/
    router_runtime.py
    expert_selector.py
    capacity_policy.py

pool/
    registry.py
    scheduler.py
    load_tracker.py
    health.py

backends/
    vllm_client.py

observability/
    metrics.py
    event_logger.py
```

先把这些文件做到稳定、清晰和可测试，比同时完成整个目录更重要。

---

# 59. 项目 Definition of Done

项目进入正式论文实验阶段前，必须满足：

| 项目 | 要求 |
|---|---|
| Router | checkpoint 固定且映射检查通过 |
| Backend | 至少 3 个真实 expert 可服务 |
| Gateway | streaming 正常 |
| Registry | READY/UNHEALTHY/DRAINING 正常 |
| Replica Scheduler | least-work 正常 |
| Health | 自动摘除/恢复 |
| Retry | pre-token retry 正常 |
| Failover | whole-pool fallback 正常 |
| Metrics | P50/P95/P99 可统计 |
| Trace | 每请求完整 JSONL |
| Load Generator | open-loop |
| Fault Tool | drain/kill/recover |
| Profiling | tokens/s 数据完成 |
| Reproducibility | config + commit + seed 保存 |
| Baseline | Best Single / LL / MoQE / Ours |
| MVP Scale | 三专家真实环境跑通 |

达到这些条件以后，才能认为：

> **工程系统完成，可以正式进入 H24 MLSys 级别实验阶段。**

---

# 60. 最终研发主线

整个工程不要理解成“把很多异构 GPU 接起来”。

真正的开发主线只有：

```text
Step 1
让每个量化 Expert 稳定成为一个服务

        ↓

Step 2
让 Registry 知道有哪些服务

        ↓

Step 3
让 Router 给出请求级 Expert quality scores

        ↓

Step 4
让 Expert Selector 综合 quality + capacity

        ↓

Step 5
让 Pool Scheduler 选择真实 instance

        ↓

Step 6
让 Gateway 保证 streaming、retry 和 observability

        ↓

Step 7
用 open-loop workload 验证系统

        ↓

Step 8
跑出 Quality–Goodput Pareto

        ↓

Step 9
扩大到 H24

        ↓

Step 10
再扩展 D40 / S80 / F108
```

因此，**现在真正应该开始写的第一批代码**应锁定为：

```text
pool/models.py
pool/registry.py

backends/base.py
backends/fake_backend.py
backends/vllm_client.py

routing/router_types.py
routing/router_runtime.py

pool/scheduler.py
pool/load_tracker.py
```

完成后立即接：

```text
gateway/lifecycle.py
gateway/app.py
```

先形成一个真正工作的端到端闭环，再继续实现 Health、Retry 和 capacity-aware routing。

这会使整个项目始终保持“每增加一层能力，都有一个已经能够运行的系统”，而不是等几十个文件全部写完以后再第一次联调。
