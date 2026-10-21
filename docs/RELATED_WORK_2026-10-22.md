# 相关工作初步核查（2026-10-22）

本次核查原论文/正式会议页面，目的是约束课程报告的表述，不是证明方法首创。检索覆盖跨模态提示耦合、输入条件提示、静态逐通道调制和提示正则化；并非穷尽式查新。

| 工作 | 原文核实的机制 | 与本项目的关系 |
|---|---|---|
| [MaPLe, CVPR 2023](https://arxiv.org/abs/2210.03117) | 在多层耦合语言和视觉提示 | 本项目直接复现的基线；新增操作仅在深层文本提示的视觉投影之后 |
| [CoCoOp, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_Conditional_Prompt_Learning_for_Vision-Language_Models_CVPR_2022_paper.pdf) | 由轻量网络为每张图像生成输入条件 token | 用来区分动态提示与本项目的样本共享静态参数；本项目没有图像条件生成器 |
| [SSF, NeurIPS 2022](https://papers.neurips.cc/paper_files/paper/2022/file/00bb4e415ef117f2dee2fc3b778d806d-Paper-Conference.pdf) | 第 3.2 节采用输入无关的通道缩放与平移，并将其吸收进前一线性层 | 静态缩放和可折叠性已有直接先例。SSF 冻结被调制的预训练操作，本项目的提示投影本来可训练，不能等同于 SSF，也不能把重参数化称为新原理 |
| [DCP, Findings of ACL 2023](https://arxiv.org/abs/2305.17903) | CMPA 通过多头注意力耦合视觉与语言提示 | 说明深层跨模态交互已有研究；注意力交互不等同于本项目的逐层对角缩放 |
| [PromptSRC, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Khattak_Self-regulating_Prompts_Foundational_Model_Adaptation_without_Forgetting_ICCV_2023_paper.html) | 使用预训练表示约束、训练轨迹集成和文本多样性进行提示正则化 | 是防止提示过拟合的相关方向；本项目目前只有原交叉熵训练，没有复现其正则化或借用其实验数字 |

## 本项目应采用的学术定位

建议表述为“在 MaPLe 的深层跨模态提示投影后，研究恒等初始化、有界静态缩放参数化的优化影响”。标量与通道版本分别增加每层一个参数和每层一个输出通道向量，不宣称增加模型函数空间。

对可训练仿射投影，`g ⊙ (Wx+b) = (diag(g)W)x + g⊙b`。因此实验应重点检查优化轨迹、参数组学习率、权重衰减及数值精度，而不是将门控权重直接解释为层/通道的因果重要性。

本项目已完成的 fixed-one 与完整 CLIP 折叠诊断见[消融报告](ABLATIONS_2026-10-21.md)。后者在 fp16 下出现少量预测改变，说明实数代数等价和有限精度的无损转换需要分开讨论。

## 尚未完成的核查

- 继续检查 layer-wise prompt weighting、跨模态投影门控、逐通道提示调制是否已有与本实现几乎相同的设计。
- 检索到题为 *Bilateral Adaptive Cross-Modal Fusion Prompt Learning for CLIP* 的候选；[OpenReview 原链接](https://openreview.net/pdf?id=5yGVBCG6nY)本次返回浏览器验证页，未完成全文和发表状态核查，不据搜索摘要下方法等同结论。
- 若扩展消融，先单独冻结提示投影或单独改变 gate 学习率，并保持其他配置一致；这些实验尚不能作为现有结果引用。

论文内不同训练配方、模型、数据划分和汇总方式的准确率不直接拼接进本项目的公平比较表。本次核查支持“已有静态调制与重参数化相关工作”的判断，不支持“首次提出层级—通道门控”的判断。
