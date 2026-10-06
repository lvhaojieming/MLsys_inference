# 模型验收与动态入池

这条流程是服务上线与动态扩容，不是仅测启动耗时：

当前支持完整的配置驱动方式，见 [config_lifecycle.md](config_lifecycle.md)。
下文单副本管理 API 用于未开启 watch_config 的模式；开启配置监听后通过文件修改目标副本列表。

```text
启动专家进程（外部启动脚本）
  → 注册 endpoint：STARTING
  → 保持 STARTING：等待 /v1/models，核对模型名并验收
  → 实际生成与预热：17 + 25 必须回答 42
  → 流式生成必须回答 42 且收到 [DONE]
  → 全部通过：READY，开始接收业务请求
  → 失败或超时：UNHEALTHY，不参与调度
```

开启 `admission_enabled: true` 后，初始配置的所有副本也经过上述验收。
默认启用验收；有副本的配置不得关闭验收，不能跳过 STARTING 直接就绪。
`/health` 表示 Gateway 存活；`/ready` 要求所需专家池各至少有一个 READY 副本。

## 启动配置

参考 `configs/admission.example.json`，填写实际 endpoint 和模型名。
Router 配置沿用已验证的 checkpoint、tokenizer、设备和专家映射。
启动前设置 `MOQE_ADMIN_TOKEN`（通过环境变量传入，勿写入配置或日志）。

```bash
python -m moqe_serving --config configs/admission.local.json --port 18081
```

只启动一个 Gateway worker。注册表和在途请求计数属于该进程内存。
重启后新增副本需要写入配置或重新注册；当前没有持久化控制平面。

## 运行中为相同专家新增副本

先用对应机器的启动脚本启动新后端，再将以下字段保存为 JSON：

```json
{"id":"awq-new-card","expert":"awq","base_url":"http://NODE:PORT/v1","model":"moqe-qwen3-awq"}
```

```bash
python scripts/register_replica.py --gateway http://127.0.0.1:18081 \
  --replica configs/new-replica.json --output results/new-replica-admission.json
```

Gateway 在验收期间继续服务已有 READY 副本，无需重启。
默认等待后端就绪与验收的总预算为 300 秒，可配置。
验收结果记录在返回值、管理快照和 admission 日志中。

管理接口均要求 `Authorization: Bearer <token>`：

| 接口 | 用途 |
|---|---|
| `GET /admin/instances` | 查看状态、在途数量和验收结果 |
| `POST /admin/instances` | 新增并验收副本，通过才入池 |
| `POST /admin/instances/{id}/validate` | 修复失败副本或重启后重新验收 |
| `POST /admin/instances/{id}/drain` | 停止分配新请求，已有请求结束后 OFFLINE |

当前只允许给已配置的专家池新增副本，例如 AWQ 池从一张卡扩到两张卡。
Router checkpoint 和专家映射保持不变，验收后的新副本参与池内最少在途请求调度。
新增 INT8 等全新专家会被拒绝；这一版不扩展专家集合。

## 验收边界

当前入池门槛是模型名、简单确定答案、非流式及完整流式的功能冒烟测试。
它不验证权重哈希、量化类型、长上下文正确性、并发容量或完整任务准确率。
单副本管理 API 不远程启动或停止模型进程；配置驱动模式可调用配置里的启动与停止命令。
周期健康探测达到失败阈值后自动摘除，连续成功达到恢复阈值后重新验收才恢复 READY。
正式实验前仍需独立完成上下文、并发、质量与容量测试。
