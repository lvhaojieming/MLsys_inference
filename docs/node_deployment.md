# 在已有统一环境上首次部署模型服务

本流程假定每个节点已有驱动、Python、vLLM / vllm-ascend、量化适配器和模型文件，
但没有本系统的模型服务进程。系统不安装依赖、创建容器或修改节点网络；配置中的
容器必须已运行。一个命令完成节点预检、helper 部署、模型进程启动、预热和验收入池。

```bash
python3 -m moqe_serving --config configs/local.json
```

也可以先检查部署条件。此命令会部署 helper，但不加载 Router、不启动模型进程或 Gateway：

```bash
python3 -m moqe_serving --config configs/local.json --prepare-only
```

任何启用节点的准备失败都会返回非零退出码。没有配置 prepare 的节点会跳过；
报告中的 nodes 为空表示没有执行准备，不代表所有主机环境已通过。

## 配置

昇腾运行环境和适配器建议通过 runtime_profiles 共享，副本填写 runtime_profile。
见 [runtime_profiles.md](runtime_profiles.md)。不使用 profile 的节点仍可按下例直接填写 prepare；
引用 profile 的受管理副本即使未写 node.prepare，也会执行自动依赖检查和 helper 部署。

完整的 V7 + 两个受管理专家的首次部署模板为 configs/first_deployment.example.json。
复制为 configs/local.json，并替换节点、容器、Router 路径、权重路径和设备编号。
其中 awq/gptq 是通用量化参数；当前集群 Ascend INT4 必须按已验证的适配器配置
替换量化参数和环境，不能直接假定任意 vllm-ascend 版本支持这些通用值。
该模板尚未作为 AWQ/GPTQ 双专家全新进程同时启动的集群实验执行。

在需要首次部署的节点配置 prepare：

```json
{
  "id": "npu-212",
  "host": "10.107.206.212",
  "ssh_target": "root@10.107.206.212",
  "ssh_options": ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15"],
  "container": "moqe-cold-node-test",
  "enabled": true,
  "prepare": {
    "python": "python3",
    "required_modules": ["vllm", "torch_npu"],
    "required_paths": ["/workspace/zhangjinhao/moqe-runtime/adapter"],
    "host_checks": [["npu-smi", "info"]],
    "runtime_checks": [],
    "env": {"ASCEND_RT_VISIBLE_DEVICES": "3"},
    "environment_scripts": ["/usr/local/Ascend/ascend-toolkit/set_env.sh"],
    "helper_path": "/workspace/zhangjinhao/moqe-managed/manage_backend.py",
    "command_timeout_seconds": 120
  }
}
```

| 字段 | 含义 |
|---|---|
| `python` | 目标运行环境中的 Python 命令或可执行文件路径 |
| `required_modules` | 实际 import 的模块，不只是检查包名存在；默认 vllm |
| `required_paths` | 必须存在且目录不为空的依赖路径 |
| `host_checks` | 在节点主机执行的检查命令数组；例如设备/驱动检查 |
| `runtime_checks` | 在模型所在容器/主机执行的补充检查命令数组 |
| `env` | 节点运行环境的基础变量，供预检和模型启停使用；launch.env 可覆盖 |
| `environment_scripts` | 在运行环境检查和模型启停前通过 bash source 的绝对路径，例如 CANN set_env.sh；不应用于主机检查 |
| `helper_path` | 自动部署的独立进程 helper 的绝对路径 |
| `command_timeout_seconds` | 每一步命令的时间预算，不包含模型加载与生成验收 |

所有启用的受管理副本的 model_path 都会自动加入目录检查。这里只检查存在和非空，
不代表权重格式、完整性或任务质量合格；模型加载和生成验收仍然不可跳过。
host_checks/runtime_checks 是明确配置的检查命令，失败即阻止该节点的待启动副本入池；
不会自动修复环境。host_checks 不进入容器，其他检查进入节点的 container（如果配置）。
设备检查可验证驱动工具是否工作，但目前不会自动挑选空闲卡或扫描配置外的占用。

对应副本仍需配置 launch，包括 start_command、stop_command、env 和 vllm_args。
启动/停止命令使用 `{backend_helper}` 引用 prepare.helper_path。
CLI 数组中将 `"{vllm_args}"` 放在 vLLM 参数位置；shell 形式不要给该占位符包引号。
节点、副本和模型路径的示例见 configs/pool_lifecycle.example.json。
当前 Ascend INT4 环境的完整参数见 configs/awq_node_cold_start_cluster.json。
该配置只测试相同 AWQ 专家的节点扩容，router 为 null；完整自动路由部署需保留
AWQ/GPTQ 两个专家池，并加入 V7 router 字段。

## 执行顺序与失败处理

每次配置处理，先排空需要移除或替换的实例，然后为新实例建立 STARTING 状态。
每个存在待启动副本的节点只准备一次：

1. 验证节点命令通信，远程节点通过 SSH；命令默认采用非交互认证。
2. 执行 host_checks。
3. 在目标运行环境 import required_modules 并检查模型/依赖路径。
4. 执行 runtime_checks。
5. 部署独立 helper，再运行 --help 检查其可执行性。
6. 逐副本启动模型服务，等待 /v1/models，执行真实生成预热及非流式、流式验收。
7. 每个通过验收的副本转 READY，允许调度。

prepare 结果不会代替模型验收。通信检查包含控制链路，后续 HTTP 模型探测及
生成验收检查请求链路。新增节点采用相同流程，已有 READY 副本继续服务。
准备失败时，该节点待启动副本转 UNHEALTHY，不执行 start_command；原有实例不因
新增副本的准备失败而下线。修复原因后将节点或失败副本先停用、再启用即可重试。

重复处理未变化的已应用实例不重新执行准备。重新启用/更换副本时重新检查；
内容一致的 helper 不重写。部署采用临时文件和原子替换，仅允许更新带有 MoQE
管理标记的 helper，拒绝覆盖无标记的其他文件。Linux helper 自包含，不要求目标
安装 Gateway 包；PID 文件记录进程身份和命令，重复启动只认领一致的存活进程。
停止命令仅操作对应记录的进程组，Gateway 退出不会自动停止后端或删除 helper。

## 查看进度和日志

GET /admin/config 的 nodes[].preparation 包含 in_progress、current_check、passed、
checks、failed_check/error 和耗时，属于准备检查报告，不是新增的实例状态。
使用 profile 时各组的检查报告位于 nodes[].preparation.profiles，节点顶层显示汇总结果。
只有实例的 STARTING、READY、DRAINING、UNHEALTHY、OFFLINE 参与状态机。

准备日志保存在 lifecycle_log_dir/node-<节点ID的十六进制编码>-<检查名>.log。
其中 runtime 日志记录 import 错误或缺失路径，helper 日志记录部署是否更新。
模型进程加载日志仍由启动 helper 的 --log-file 指定；验收结果及状态转换在 Gateway
日志和管理接口中。准备过程中模型副本不可调度；/health 只代表 Gateway 存活，
完整自动路由就绪应检查 /ready。

## 验证

同一个集成脚本覆盖首次启动和热加入，避免维护两份重复的冷启动代码。
要求配置包含一个已有服务和一个受管理的同专家副本，且测试端口空闲。
脚本会启动临时 Gateway、发送非流式及流式请求、验证两个副本实际接到流量，
最后停用新增节点、等待其进程停止，并关闭临时 Gateway。

```bash
python3 scripts/test_node_cold_start.py --config configs/awq_node_cold_start_cluster.json \
  --output /path/to/results/initial --initial-start
python3 scripts/test_node_cold_start.py --config configs/awq_node_cold_start_cluster.json \
  --output /path/to/results/hot-add
```

这是模型进程冷启动，模型文件和运行环境已存在，不等于磁盘缓存冷启动。

2026-10-06 在 .209 控制端与 .212 NPU 3 上验证当前 AWQ 环境：热加入五项检查通过，
从启用到 READY 约 157.55 秒；首次启动五项检查通过，从 Gateway 进程启动到确认
READY 约 145.15 秒。测试包含真实 14B 模型加载、生成与流式验收、业务请求和
停用后的进程/端口清理。两次都保持原有 .209 后端运行，且未切换线上 V7 Gateway。
这些结果验证单个受管理 AWQ 节点，不代表双专家同时冷启动或所有部署环境已验证。
