# 原论文基线复现核对

目标是先确认原始 MaPLe 的复现质量，再解释门控相对同条件基线的变化。
不能以“代码运行成功”代替“复现论文数值”，也不能将官方权重评估当作从头训练。

## 已完成六数据集

论文参考来自 MaPLe CVPR 2023 / arXiv:2210.03117v3 的 Table 3。
本项目使用 16-shot、ViT-B/16、N_CTX=2、PROMPT_DEPTH=9、SGD lr=0.0035、5 epochs，seeds 1/2/3。

| 数据集 | 论文 Base | 本项目 none Base | 论文 Novel | 本项目 none Novel | 论文 HM | 本项目逐种子 HM 均值 |
|---|---:|---:|---:|---:|---:|---:|
| OxfordPets | 95.43 | 95.4280 | 97.76 | 97.7815 | 96.58 | 96.5902 |
| DTD | 80.36 | 80.7099 | 59.18 | 55.8776 | 68.16 | 66.0045 |
| Caltech101 | 97.74 | 97.8481 | 94.36 | 95.1965 | 96.02 | 96.5040 |
| Flowers102 | 95.92 | 95.8848 | 72.46 | 73.4279 | 82.56 | 83.1611 |
| EuroSAT | 94.07 | 91.1349 | 73.23 | 73.1709 | 82.35 | 81.1041 |
| FGVCAircraft | 37.44 | 36.5746 | 35.61 | 35.8728 | 36.50 | 36.2045 |

Pets 与论文报告接近。DTD Novel 均值比论文低约 3.30 个百分点，尚不能写成“全面达到论文精度”。
按三种子的 Base/Novel 均值再算 HM，DTD 为 66.0364，与逐种子 HM 均值的差异不足以解释论文差距。
Caltech101、Flowers102 和 FGVCAircraft 的 none 基线与论文报告较接近；EuroSAT Base 低约
2.94 个百分点，逐种子波动也较大。硬件、软件版本、采样/随机状态和训练日志差异仍可能影响
数值，当前没有证据支持单一原因。
已有原始单批次/前向一致性检查不等于完成全训练逐批一致性证明。

门控主结论只能使用本项目同数据、同划分、同种子和同训练配方的 `none` 作为配对对照。
不得将 DTD 的本地基线偏低包装成超过原论文的证据。

## DTD 作者公开日志与固定上游源码重放

作者公开 DTD 日志中的三个种子为：Base `81.13/81.37/78.59`，Novel
`60.87/56.76/59.90`，均值对应论文的 `80.36/59.18`。日志中的学习率序列、16 shots、
训练/验证/测试样本数、N_CTX=2、八个深层投影、ViT-B/16、fp16、batch size 和 epoch 数与
本项目一致。公开日志使用 PyTorch 1.9.0+cu111、torchvision 0.10.0、Python 3.7、A100；
本项目使用 PyTorch 2.1.2+cu118、torchvision 0.16.2、Python 3.10、RTX 3090。日志中的
历史 trainer/config 名称带有 `CoCoOp`，但模型键包含 MaPLe 的深层 prompt 与投影，且均值与
论文 MaPLe 表一致；因此这里只将它作为作者公开 MaPLe 运行证据，不把它误称为当前 CoCoOp 基线。

为了判断本项目门控改动是否意外改变 `none`，使用 SHA-256 固定的原始官方
`trainers/maple.py` fixture，在同一环境、同一 DTD 数据与三个种子上重新完整训练和评估。
重放结果与本项目已有 `none` checkpoint 的全部张量逐元素完全相同：三个种子的
`unequal_tensor_count` 均为 0，Base/Novel correct/total 也逐项相同。由此可以确认本项目
`none` 路径没有引入相对于固定上游源码的训练行为变化。它仍不能解释作者环境、few-shot cache
或依赖版本造成的论文数值差距。

最终独立验收共检查 57 个新增训练结果，确认 checkpoint 浮点张量有限、冻结 CLIP 权重一致、
新增数据集 fixed split/few-shot cache 未变化、冻结投影对照的投影参数未更新。机器可读证据位于
本地恢复包 `expansion_20261022/remote_recovery_20261022/gpu_20261022_02/`。

## 核对顺序

1. 保留当前全部结果、split、few-shot cache、日志和 checkpoint；不得为了匹配论文选择性换种子。
2. 最终报告并列给出论文值、作者公开日志、本项目 none 和门控结果，明确来源与聚合方式。
3. 将 DTD 和 EuroSAT 的基线差距列为复现限制；不利用 Novel 测试结果继续调参。
4. 若要进一步定位环境差异，应作为独立复现实验固定旧版依赖和作者 cache，使用新的 RUN_ID。

参考：[论文](https://arxiv.org/abs/2210.03117)、[官方运行说明](https://github.com/muzairkhattak/multimodal-prompt-learning/blob/main/docs/RUN.md)。

