RFPO-MLP 消融实验
================

RFPO-MLP 保留条件 Transformer decoder，将其 CLS 输出与时间 embedding 相加，再拼接当前去噪步展平后的 base velocity，通过 MLP 预测整个 action chunk 的残差速度。使用以下配置启动单任务和多任务实验；它们分别与对应的 DiT 配置保持一致，仅修改 actor 架构和实验名称。

在 RLinf 根目录运行::

    bash examples/embodiment/run_async.sh libero_object_async_rfpo_td3_openpi_pi05_mlp
    bash examples/embodiment/run_async.sh libero_object_async_rfpo_td3_openpi_pi05_multi_mlp

单任务配置保持 task_id_filter=[0] 和 global_batch_size=256；multi 配置保持 task_id_filter=null 和 global_batch_size=512。其余设置，包括 backbone checkpoint、critic、优化器、去噪步、引导步、探索噪声、target smoothing、残差 L2、评测和保存周期，均保持原配置的值。

网络配置
--------

在 actor.model.actor 下设置 architecture: mlp 或 architecture: dit；默认值为 dit，原启动配置继续使用 DiT。mlp_hidden_dim 控制 MLP 隐藏层宽度，mlp_num_hidden_layers 控制隐藏层数量，默认分别为 256 和 3（另有一层线性输出层）。hidden_size 仍控制条件 decoder 和时间 embedding 的宽度；num_dit_blocks 只用于 DiT，num_heads 和 mlp_ratio 仍用于条件 decoder。

默认的 CLS 和时间特征为 [B, 256]，当前步 base velocity 为 [B, 10, 7]。拼接后为 [B, 326]，经 Linear(326, 256)、ReLU、Linear(256, 256)、ReLU、Linear(256, 256)、ReLU、Linear(256, 70)，再恢复为 [B, 10, 7]。MLP 的输入和输出长度由 sampler.rfpo_action_chunk 确定；该网络在所有引导步骤间共享。

输出继续表示残差速度，最后一层为无界线性输出，权重使用标准差 1e-4 的非零初始化。sampler 负责叠加 base velocity、添加 TD3 噪声并积分；actor 的返回字段保持不变。冻结 π 参数的同时，训练保留经过 base velocity、suffix 和去噪轨迹的梯度。

DiT 与 MLP 使用不同的残差网络参数。切换架构时启动新实验；恢复 checkpoint 时使用原架构和网络尺寸。DiT 的参数名称和初始化行为保持兼容。比较性能时同时报告残差 actor 参数量，因为默认 MLP 的容量小于 DiT。

验证
----

在项目测试环境运行::

    pytest tests/unit_tests/test_models.py -k rfpo

这些测试覆盖 actor 输出、跨去噪输入的梯度、条件 padding mask、Gaussian 接口、chunk 校验，以及两个启动配置相对基线的训练设置一致性。LIBERO 端到端训练还需要原配置指定的 π 权重、资源和 GPU 环境。
