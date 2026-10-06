# 昇腾运行环境与适配器配置

`expert` 决定 Router 的逻辑专家池，`runtime_profile` 决定实例使用的运行环境和
后端适配。一个 AWQ 专家池可以包含多个昇腾运行 profile 的副本，不改变 Router
的输出维度。当前只支持 Ascend + vLLM/vllm-ascend，不提供其他硬件后端。

```json
{
  "runtime_profiles": {
    "ascend_moqe_int4": {
      "accelerator": "ascend",
      "backend": "vllm",
      "adapter": "moqe_ascend_int4",
      "environment_scripts": [
        "/usr/local/Ascend/ascend-toolkit/set_env.sh",
        "/usr/local/Ascend/nnal/atb/set_env.sh"
      ],
      "env": {"MOQE_ASCEND_INT4_ADAPTER": "1", "OMP_NUM_THREADS": "4"},
      "vllm_args": {"dtype": "float16", "max_num_seqs": 8}
    }
  }
}
```

副本引用名称：`"runtime_profile": "ascend_moqe_int4"`。
start_command 中使用 `{backend_module}` 选择后端模块，使用 `{vllm_args}` 注入参数，
`{backend_helper}` 引用节点的 helper。当前 backend_module 为
`vllm.entrypoints.openai.api_server`，通过安装环境中的 vllm-ascend 插件运行。
完整集群示例见 configs/awq_node_cold_start_cluster.json。

## 字段与生效规则

| 字段 | 作用 |
|---|---|
| `accelerator` | 默认 ascend，当前只接受 ascend |
| `backend` | 默认 vllm，当前只接受 vllm |
| `adapter` | native 保留后端默认行为；其他名称作为默认 quantization 参数 |
| `device_env` | 可省略；如填写，必须为 ASCEND_RT_VISIBLE_DEVICES |
| `env` | 公共运行环境变量；适配器依赖的 PYTHONPATH 和开关需明确配置 |
| `environment_scripts` | 预检和模型启停共同加载的环境脚本，均为绝对路径 |
| `required_modules` | 额外导入检查；系统始终检查 vllm、torch_npu、vllm_ascend |
| `host_checks` | 未填写时检查 npu-smi info；填写 [] 可关闭自动主机设备检查 |
| `runtime_checks` | 在目标运行环境执行的额外检查命令数组 |
| `vllm_args` | 公共后端参数，副本的 launch.vllm_args 可覆盖 |

环境变量的优先级为 profile.env → node.prepare.env → launch.env。
ASCEND_RT_VISIBLE_DEVICES 由副本的 device_ids 自动生成，不接受 env 中另行指定
其他卡号；预检时使用同一节点该 profile 待启动副本的设备编号集合。
环境脚本按 profile → node.prepare 顺序加载，重复路径只加载一次。
准备使用 profile 和节点的公共环境；实例的 launch.env 覆盖在模型启动时生效。
如某依赖只存在于特定 PYTHONPATH 中，应放在公共 profile.env 或 node.prepare.env，
确保预检和服务均能导入。

`adapter` 不负责安装量化代码或转换权重。自定义适配器必须已经存在于运行环境，
配置其需要的路径、模块和开关。默认 quantization 来自 adapter，profile.vllm_args
以及副本参数可显式覆盖，最终参数仍必须由目标 vLLM 版本支持。
native profile 通常由副本设置具体 quantization；不能据此假定所有量化格式均受支持。

## 首次启动、扩容和修改

受管理副本引用 profile 时必须归属节点；即使节点未写 prepare，系统也会生成
默认准备设置并部署 helper。可在 node.prepare 指定 Python、helper 路径和时间预算。
只有外部已有服务（launch=null）引用 profile 时，profile 仅作为部署信息；
除非节点明确配置 prepare，否则不执行准备、改变设备或启动/停止外部服务。

同节点按 profile 分组准备，各组独立检查依赖和模型路径。失败只阻止该组待启动
实例入池；其他组通过后仍可 READY。所有实例沿用原有五状态和真实推理验收。
GET /admin/config 的 nodes[].preparation.profiles 按 profile 显示阶段与结果，
instances[].runtime_profile 显示实例归属。准备日志按节点和 profile 编码分开保存。

开启 watch_config 后可以修改 runtime_profiles。只排空并重启引用发生变化的
profile 的实例，使用旧环境停止旧服务，再按新环境准备、启动并验收。
增加未被引用的 profile 不影响现有实例。切换副本 runtime_profile 同样走替换流程。
删除仍被副本引用的 profile、未知引用、其他硬件类型和重复 CLI 参数会在启停前拒绝。

未写 runtime_profiles/runtime_profile 的旧配置保持原有行为。
首次部署模板为 configs/first_deployment.example.json；其中 native 量化参数为通用
示意值，当前集群的 MoQE INT4 应使用已验证的 ascend_moqe_int4 配置和对应权重。
