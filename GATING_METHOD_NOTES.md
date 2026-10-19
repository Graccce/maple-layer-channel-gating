# MaPLe 层级—通道门控说明

本实现是课程实验候选方法，没有准确率提升或首创性结论。

## 原机制与新增方法

MaPLe 将文本浅层上下文投影为视觉浅层提示，并将每层深层文本提示通过对应线性层映射为视觉深层提示。代码中深层数量为 `PROMPT_DEPTH - 1`；标准配置深度 9，因此有 8 组深层提示。文本和视觉编码器的冻结策略沿用原实现。

仅对深层视觉投影增加：

`z_l = W_l P_t_l + b_l`

`g_l = 1 + 0.5 tanh(alpha_l)`

`P_v_l = g_l * z_l`

`none` 不创建门控参数或门控计算，保留原始投影路径。`scalar` 每层一个零维参数，整体缩放该层全部 token/通道。`channel` 每层一个输出通道向量，广播到所有提示 token。数量取实际投影列表长度，通道维度取对应 `Linear.out_features`。

这是按层共享的静态可学习门控，不依赖每张图片，不是输入条件动态门控。浅层提示、文本深层提示、原始优化器、学习率和训练参数均不变。

## 初始化、参数和计算

alpha 零初始化，不消耗随机数；初始门控为 1。实数公式的门控范围为 (0.5, 1.5)，浮点饱和可能达到边界。门控的 tanh 用 fp32 计算，再转为投影输出 dtype，保留 autocast 的输出类型；参数由模型统一迁移设备。

设深层数量 L，每层视觉宽度 D_l，提示 token 数 N：scalar 增加 L 个参数，channel 增加 sum(D_l) 个参数。两者都增加 sum(N*D_l) 次乘法；tanh 次数分别是 L 和 sum(D_l)。none 没有新增参数或张量运算。

标准 ViT-B/16、N=2、L=8：scalar +8，channel +6144。按当前模型结构计算，原提示参数预期 3,555,072，scalar 3,555,080，channel 3,561,216；以真实训练初始化日志为最终核对依据。门控虽计算很少，是否影响实际时间仍应测量。

## 参数冗余与研究假设

对可训练仿射投影，静态门控可吸收为 `W'_l = diag(g_l) W_l`、`b'_l = diag(g_l) b_l`。scalar 同理。因此不能说它扩大了原模型的函数表达空间。可能差异来自重参数化、梯度路径、初始化附近的优化轨迹或优化器正则化，需要实验验证。

假设 H1：恒等初始化的逐层缩放可能改变优化轨迹；H2：通道级缩放相对标量可能影响 Base/Novel 权衡；H3：差异可能小于随机种子波动。三者都可能不成立。负结果同样是有效课程结果。

建议消融：none/scalar/channel 同配置；三个配对种子；固定为 1 的 gate；仅诊断性的冻结投影消融（单独实验，明确改变训练条件）；不同 gate 学习率或正则化（与主比较分开）；训练后折叠 gate 到 W,b 并验证等价；记录每层门控均值、标准差、极值和饱和比例。不要提前添加大量变量导致主实验不可解释。

## Checkpoint 与数值边界

新门控模型保存 `_gate_schema=[1, mode]` 和逐层 gate 参数。none 不保存新 schema，保持原始参数键集合。评估只重建 class-dependent 的 token_prefix/token_suffix，其他学习参数严格匹配；显式 INIT_WEIGHTS、普通 load_model、Dassl resume 都检查 gate 家族。原始 none、scalar、channel 不交叉加载；缺失/多余 gate 或形状、版本不匹配报错，不默默用零初始化补齐。没有实现 baseline→gate 热启动，主比较均从相同随机初始化训练。

旧目录当前源码的 `gate_alphas` 是二维参数，使用 sigmoid，初始为 0.5；但旧 MaPLe_Ours 磁盘 checkpoint 的实际形状是 [8]，与当前源码不一致。历史公式不能仅凭参数键推断，更不能把旧结果当成新方案。新代码明确拒绝旧 gate 家族；旧实验必须先恢复其匹配源码和配置才能评估，直接使用当前原目录也可能形状不匹配。checkpoint 的家族标识不证明已训练，需同时核对 epoch、训练完成日志和来源。

tanh 饱和会减小梯度；半精度会有量化误差，小门控变化可能暂时不改变半精度输出。原项目浅层 `proj.half()` 是已有行为，纯 fp32 路径可能存在 dtype 问题；本改动不混入该项修复。正式比较先统一使用已有 fp16 配置，再独立验证 AMP。CPU bfloat16 测试不能替代 CUDA fp16/AMP 验证。

## 相关工作核查清单

重点检索 CLIP prompt learning 中的 layer-wise prompt weighting、channel-wise prompt scaling、跨模态 prompt projection/gating、static vs instance-conditioned prompts，以及线性层重参数化、逐通道缩放和优化效应。查原论文、附录、代码和发表年份，区分作用对象、输入条件性、初始化和训练参数；本次没有完成系统的新颖性检索，不宣称首创。

可靠起点：

- [MaPLe 官方代码](https://github.com/muzairkhattak/multimodal-prompt-learning)
- [MaPLe 原论文](https://arxiv.org/abs/2210.03117)

