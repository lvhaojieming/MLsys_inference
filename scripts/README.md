# 运维与实验入口

正常服务从 `python3 -m moqe_serving --config ...` 启动。首次部署预检使用同一入口
的 `--prepare-only`，不需要额外的部署脚本。进程 helper 的唯一实现位于
`src/moqe_serving/deployment/backend_process.py`，节点准备会将其独立部署到目标环境。
`manage_backend.py` 仅保留源代码 checkout 的兼容入口，不应单独复制到远端。

| 脚本 | 用途 |
|---|---|
| `test_node_cold_start.py` | 相同专家的真实节点冷启动、热加入、验收、请求与停止；首次启动加 --initial-start |
| `test_pool_lifecycle.py` | 已有后端的副本/节点扩缩容和在途请求排空 |
| `smoke_real_backends.py` | 后端接口冒烟检查 |
| `smoke_router_ascend.py` | Router 在 Ascend 环境加载和推理检查 |
| `check_auto_gateway.py` | 自动路由 Gateway 检查 |
| `register_replica.py` | 非配置管理模式的单副本注册入口 |
| `benchmark_cold_start.py` | Gateway 启动计时；不代替受管理后端的冷启动和清理流程 |
| `benchmark_v7_ttft.py` | V7 直接后端、固定 Gateway、自动路由的 TTFT 对照 |
| `validate_embedding_graph.py` | 图执行与保存预测的专家选择及概率差异验证 |
| `profile_v7_router.py` | 同步分解 Router 的分词、Embedding、head 耗时 |
| `probe_embedding_graph.py` | 更换运行时后验证 causal 路径与图捕获的数值一致性 |

这些入口用于复现实验或检查真实集群，不在默认 pytest 中执行。自动回归测试位于
tests，公共模拟后端集中在 admission_support，测试文件之间不互相导入。
真实集群实验产物应写入独立输出目录，不提交日志、权重或预测缓存到仓库。
