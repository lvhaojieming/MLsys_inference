# 配置驱动的同专家扩缩容

启动时只传配置文件，其余部署参数由配置读取：

```bash
python -m moqe_serving --config configs/awq_pool_cluster.json
```

`awq_pool_cluster.json` 的 .213 节点、容器、模型路径和两个服务端口已根据运行进程核对。
它用一个 AWQ 专家池验证副本扩缩容，未配置 Router，也未部署到现有 Gateway。
两个后端目前均为其他实验常驻服务，因此 `launch: null`：只接入和退出调度，绝不停止这些后端。
修改第二个副本的 `enabled` 为 true，就将它加入目标副本列表。

跨节点实验配置是 `awq_nodes_cluster.json`：使用 .209 和 .213 两个真实节点的相同 AWQ 专家。
`nodes[].enabled` 是整节点的开关，`replicas[].enabled` 是节点内单个副本的开关；两者都为 true 才入池。
新增节点时添加 nodes 条目及归属该节点的 replicas 条目，逐个启动和验收；通过的实例才可调度。
停用节点时，其所有实例先同时停止接收新请求，再等待在途请求结束。
已排空的各副本按文档状态机转 OFFLINE；重新启用节点后重新验收，才能恢复 READY。
实例状态机仍只有五种状态，节点属于分组和目标配置，不额外引入一套节点状态枚举。
同节点多副本和跨节点副本都进入同一专家的调度候选；当前仍采用最少在途请求调度。
这些配置接入现有常驻服务（launch=null），整节点退出调度不会停止共享后端进程。

## 文件中的参数

| 配置字段 | 含义 |
|---|---|
| `gateway` | 监听地址、端口、日志级别 |
| `nodes` | 节点 ID、地址、SSH 目标、ssh_options（端口、身份文件等）、容器名称 |
| `runtime_profiles` / `replicas[].runtime_profile` | 昇腾后端环境和适配器的公共定义及副本引用 |
| `replicas` | 目标副本列表；一个条目对应一个服务实例，可占多张卡 |
| `enabled` | true 加入目标列表；false 退出；也可删除该条目 |
| `nodes[].enabled` | 一次启用或停用整个节点及其下全部副本 |
| `node_id` / `device_ids` | 副本所在节点与物理设备编号 |
| `model_path` / `model` | 后端文件系统内的模型路径、对外模型名称 |
| `base_url` / `backend_port` | Gateway 可达的服务地址与监听端口 |
| `launch.start_command` / `stop_command` | 配置管理的后端启动和停止命令，均为参数数组 |
| `launch.env` | 后端启动环境，例如可见设备、PYTHONPATH、CANN 库路径 |
| `launch.vllm_args` | 结构化 vLLM 参数，展开到启动命令的 `{vllm_args}` 位置 |
| `admission` | 验收题目、标准答案、生成参数与就绪轮询间隔 |
| `admission_timeout_seconds` | 后端就绪与完整验收的总时间预算 |
| `health` | 周期探测间隔、单次超时、连续失败与恢复阈值 |
| `drain_timeout_seconds` | 移除前等待已有请求结束的时间预算 |
| `config_poll_interval_seconds` | 检测配置变更的间隔 |
| `timeout_seconds` / `max_connections` | Gateway 后端 HTTP 客户端参数 |
| `default_max_tokens` / `default_enable_thinking` | 自动路由请求缺省生成设置 |
| `router` | checkpoint、tokenizer、模型代码路径、设备、专家映射 |
| `lifecycle_log_dir` | 启停命令的本地输出目录 |
| `admin_token_env` | 管理凭据的环境变量名；凭据本身通过环境提供 |

同一节点的 enabled 副本不得重复占用设备，服务 endpoint 不得重复，backend_port 必须与 URL 一致。
这些校验只覆盖当前配置，不能判断配置以外的训练或服务占用了哪些卡。

## 保存配置后的行为

```text
增加条目 / enabled=true
  → 可选：执行 start_command，启动独立后端
  → 等待模型就绪 → 实际生成预热 → 非流式与流式正确性测试
  → 通过后 READY，加入相同专家的副本调度

删除条目 / enabled=false
  → DRAINING，立刻停止分配新请求
  → 已有请求全部结束
  → 若是本控制器管理的后端，执行 stop_command；否则只退出调度
  → OFFLINE
```

新增验收失败时副本为 UNHEALTHY，原有 READY 副本继续服务。
修改现有副本的节点、设备、模型路径或启动参数，会先 drain 旧实例，再启动并验收替代实例。
仅对带 launch 的副本自动控制后端进程；launch=null 的节点、设备、模型路径属于部署信息，
实际后端的迁移、模型替换仍需外部执行，然后将新 endpoint 配入文件。
如果已有请求超过 drain 时间预算，保持停止分配新请求，不强杀后端；配置状态会记录错误。
问题修复后可调用 `POST /admin/config/reload` 重试（需管理令牌），或再次修改并保存配置。
仅验收失败的副本可先设 false，等待 OFFLINE，再设 true 重试。

建议编辑临时文件后原子替换配置。非法 JSON、重复设备、全新专家等配置错误在操作前被拒绝。
节点、副本和 runtime_profiles 支持热修改；Gateway 端口、Router checkpoint、验收设置等全局参数在重启时读取，
修改这些字段后必须重启，当前版本会明确拒绝将其作为热更新执行。
开启配置管理后，单副本注册、drain 和 validate API 禁用，配置文件是副本目标状态的唯一来源。

## 冷启动由配置管理

参考 `pool_lifecycle.example.json` 的第二个副本。该副本默认 disabled，路径和节点占位值需要修改。
配置中 `start_command` 通过 Linux helper 启动后台服务并返回；不要填持续阻塞的前台服务命令。
helper 后面的 vLLM 参数可以使用下面的结构化对象，包括量化类型、上下文、并发、内存比例等。
`stop_command` 仅停止 PID 文件中记录且进程身份匹配的进程组。

命令支持 `{id}`、`{expert}`、`{model}`、`{model_path}`、`{port}`、`{device_ids}`、`{host}`、`{container}`、`{backend_helper}` 替换。
Node 配有 ssh_target 时先通过 SSH 执行；配置 container 时在对应容器中执行。
Gateway 所在主机必须有 SSH/docker 客户端及目标访问权限；配置 `nodes[].prepare` 时自动部署独立 helper，
否则需要提前部署 `src/moqe_serving/deployment/backend_process.py` 到目标环境的 helper 路径。
Ascend 驱动/CANN 及量化适配器的必要环境通过 launch.env 或已有启动脚本提供。
示例的通用 vLLM 参数不能直接替代当前集群的 moqe_ascend_int4 适配启动参数。

Gateway 退出会停止监听配置，但不会停止常驻后端，避免重启 Gateway 影响推理进程。
重启时带 launch 的副本通过对应 PID 文件认领，helper 会检查已运行进程与命令一致。
一律使用单 Gateway worker；当前没有跨进程一致性控制、全新专家热添加。
状态机及阈值健康摘除/恢复见 [instance_state_machine.md](instance_state_machine.md)。
功能冒烟验收不替代任务质量评估、权重哈希验证、并发容量与长上下文测试。

## 直接配置 vLLM 参数

每个受管理副本的 `launch.vllm_args` 可单独设置后端参数：

```json
"vllm_args": {
  "quantization": "moqe_ascend_int4",
  "dtype": "float16",
  "tensor_parallel_size": 1,
  "max_model_len": 12288,
  "max_num_seqs": 8,
  "max_num_batched_tokens": 2048,
  "gpu_memory_utilization": 0.5,
  "enable_chunked_prefill": true,
  "enable_prefix_caching": false,
  "enforce_eager": true
}
```

| 参数 | 含义 |
|---|---|
| `max_model_len` | 后端允许的上下文长度，包含输入与输出 |
| `max_num_seqs` | vLLM 调度的最大序列数 |
| `max_num_batched_tokens` | 单次调度迭代的 token 预算 |
| `gpu_memory_utilization` | 每实例的设备内存利用比例；不是 Gateway 连接上限 |
| `tensor_parallel_size` / `pipeline_parallel_size` | 张量 / 流水线并行规模，必须匹配部署设备 |
| `dtype` / `quantization` / `kv_cache_dtype` | 计算、权重量化和 KV cache 类型，须与实际后端支持匹配 |
| `enable_chunked_prefill` | 是否开启分块 prefill |
| `enable_prefix_caching` | 是否开启 prefix cache |
| `enforce_eager` | 后端是否强制 eager；与 Router 的 `embedding_graph` 分属不同进程 |

键使用下划线或连字符，不带 `--`。整数/浮点数/字符串生成 `--key value`；
true 生成 `--key`，false 生成 `--no-key`，null 不传该参数。
只有目标版本支持 `--no-key` 时才使用 false；旧版 store_true 开关应使用 null
省略，或使用该版本提供的明确关闭参数。例如 `enforce_eager: null` 省略
`--enforce-eager`。字典序列化成单个 JSON 参数，非空列表传为多个参数值。
具体参数名和选项以目标节点安装版本的 `api_server --help` 为准：
系统不绑定固定版本的参数白名单，未知或不支持的参数会使后端启动失败，不能入池。
vLLM 官方参数说明见 https://docs.vllm.ai/en/v0.10.1/cli/serve.html 。

在 `launch.start_command` 的后端命令末尾放入一个独立的 `"{vllm_args}"`
数组元素；如果需要 Ascend 环境初始化，可放在 `bash -lc` 命令字符串中，
系统会先将结构化参数转换为 shell 引号保护的参数列表。不要给占位符再包引号。
`configs/pool_lifecycle.example.json` 展示数组形式，
`configs/awq_node_cold_start_cluster.json` 展示当前 Ascend 参数与共享环境脚本，避免在长 shell 命令里重复参数。
同一参数不能同时出现在对象和原启动命令里；配置校验会拒绝重复。
`model`、`served_model_name`、`host`、`port` 保留在副本字段和原启动命令中，
不能在对象里重复设置。旧配置不含 `vllm_args` 时行为保持不变。

开启 `watch_config` 后修改这些参数，会 drain 旧实例，等待请求完成，
停止受管理进程，再用新参数启动并完整验收。不会修改正在运行进程的内部配置。
`launch: null` 的外部常驻服务不受这些启动参数控制。
