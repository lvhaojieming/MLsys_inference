# 实例状态机：对应技术报告第 7.1、24、25、36 节

```text
STARTING
    ↓ 加载、预热、验收全部通过
 READY
 ↙   ↘
DRAINING   UNHEALTHY
   ↓          ↓ 重新检查并验收通过
OFFLINE     READY
```

状态枚举与开发文档完全一致，定义于 `pool/models.py`。
测试和预热是状态内的工作，不新增 VALIDATING 状态。
Registry 的 `transition()` 集中校验转换规则并记录原因和时间，Controller、验收流程不得自行修改状态。
只有 READY 实例能获取业务请求 lease；STARTING、DRAINING、UNHEALTHY、OFFLINE 全部排除。

- 新增副本：从 STARTING 开始，先启动、预热、验收，再转 READY。
- 减少副本：先转 DRAINING，禁止新请求，等待已有请求完成后转 OFFLINE。
- 无在途请求时也必须记录 READY → DRAINING → OFFLINE，不能直接跳转。
- 健康失败：READY → UNHEALTHY，停止新请求；通过恢复验收后 UNHEALTHY → READY。
- 并发保护：drain 优先于晚到的验收结果，已 OFFLINE 的实例不能被验收回调重新放入池中。
- 禁止重复并发验收，禁止 OFFLINE → READY 和 DRAINING → READY 等跳过检查的转换。

开发文档图只画正常服务路径。实现补充异常与重启边，不增加状态：
STARTING → UNHEALTHY（启动或验收失败）；STARTING/UNHEALTHY → DRAINING（取消或撤下）；
OFFLINE → STARTING（明确重新启用，重新启动和验收）。

转换记录可通过日志 `instance_transition` 和 `GET /admin/config` 的 transitions 获取。
当前已实现周期健康探测、连续失败与恢复阈值。失败达到配置阈值后 READY → UNHEALTHY；
连续探测成功达到恢复阈值后仍须重新通过生成与流式验收，才允许 UNHEALTHY → READY。
一个实例缓慢恢复不会阻塞其他实例的周期探测。DRAINING/OFFLINE 不参与自动恢复。
参数位于 config.health；真实集群故障注入与恢复验收尚未执行。
