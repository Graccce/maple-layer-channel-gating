# 论文框架与写作要点（草稿）

> 用途：正式论文（CVPR LaTeX 或 Word）的提纲。每节写明"要写什么、用哪些证据、要避免哪些说法"。
> 方括号 `【】` 中的数字等第二阶段实验完成后填入；来源文件均在本仓库 `docs/` 下。

## 暂定题目

- 中文：**从静态缩放到输入条件门控：MaPLe 深层跨模态提示投影的再思考**
- 英文：*Beyond Static Scaling: Input-Conditioned Cross-Modal Gating for Multi-modal Prompt Learning*

## 摘要（约 200 字，最后写）

四句话结构：
1. 背景：MaPLe 通过逐层将文本 prompt 线性投影为视觉 prompt 实现视觉—语言协同提示学习。
2. 发现：我们首先复现 MaPLe，并研究在深层投影后加入可学习静态门控；理论上它可被吸收进投影矩阵、
   不增加表达能力，实验上在六个数据集上结果正负参半，并在 FGVCAircraft 上暴露出训练不稳定。
3. 方法：据此提出输入条件跨模态门控（由当前层 CLS token 生成逐图像、逐通道门控，零初始化、
   无需额外前向），并结合零样本知识一致性约束缓解新类遗忘。
4. 结果：在六个数据集 base-to-novel 设定下，平均 HM 【+x.xx】，Novel 【+x.xx】，参数仅增加【约 0.4M】。

## 1 引言（约 1 页）

要点顺序：
1. CLIP 零样本强，但下游适配时全量微调代价大、易遗忘 → prompt learning（CoOp、CoCoOp、VPT）。
2. MaPLe 的核心：视觉、语言两侧同时加 prompt，用 `F_l` 把文本 prompt 投影成视觉 prompt，实现深层耦合。
3. **问题提出**：投影输出对每张图片完全相同，"该给视觉 prompt 多大强度、哪些通道"是固定的。
   一个自然的想法是加门控 —— 我们先做了静态门控，并发现它在理论和实验上都站不住（这是本文第一个贡献）。
4. **证据引出第二个问题**：用本项目自己的数据说明 MaPLe 在新类上的遗忘：DTD 上 zero-shot Novel 60.27
   > MaPLe 55.88；Flowers 77.52 > 73.43（`RESULTS_EXPANDED_2026-10-22.md` Zero-shot 表）。
5. 贡献列表（3 条）：
   - 对 MaPLe 深层投影上的静态门控给出可吸收性分析与六数据集 × 三种子的系统负结果，并用三组预登记诊断
     （fixed-one 逐位等价、门控学习率、逐 epoch 轨迹）把 FGVCAircraft 的崩溃定位为优化轨迹敏感；
   - 提出输入条件跨模态门控（ICG），在不增加图像编码器前向次数的前提下使门控不可被静态吸收；
   - 结合零样本一致性约束，在六个数据集上取得【……】，并通过消融说明各部分作用。

**避免**：不说"首次提出门控"（SSF 已有静态缩放，CoCoOp 已有输入条件 prompt）；新颖点表述为
"在 MaPLe 的深层跨模态投影位置、以层内 CLS 为条件、零初始化恒等起点"的具体设计与分析。

## 2 相关工作（约 0.75 页）

三段，每段最后一句点明与本文的差异：
- **视觉—语言提示学习**：CoOp、CoCoOp（输入条件 token，需额外一次图像前向）、VPT、MaPLe、DCP。
- **防遗忘 / 正则化**：KgCoOp（文本特征约束）、PromptSRC（特征 + logits 一致性、轨迹集成）。
  明确写出本文 KC 损失的权重直接沿用 PromptSRC，**不是本文原创**，本文贡献在于与门控的组合与分析。
- **重参数化与特征调制**：SSF（静态 scale & shift，可折叠进前一线性层）、FiLM / SE（输入条件通道调制）。
  用来支撑"静态门控可吸收、条件门控不可吸收"的论证。
- 已核查文献列表见 `RELATED_WORK_2026-10-22.md`；待补全 *Bilateral Adaptive Cross-Modal Fusion Prompt
  Learning for CLIP* 的全文核查。

## 3 预备知识：MaPLe 复现（约 0.5 页）

- 公式：浅层 `P_v^0 = F_0(P_t^0)`；深层 `P_v^l = F_l(P_t^l), l=1..J-1`，J=9，N_CTX=2。
- 复现表（`BASELINE_REPRODUCTION_AUDIT.md`）：论文 Table 3 vs 本项目 none 六数据集 Base/Novel/HM。
- 必须写清的复现限制：DTD Novel 低 3.30、EuroSAT Base 低 2.94；与作者日志在 PyTorch 版本、GPU
  上不同；用固定上游源码重放验证了 none 路径逐位一致（说明差距不是我们改代码造成的）。

## 4 静态门控：一个失败但有信息量的尝试（约 1 页，本文特色章节）

4.1 方法：`g_l = 1 + 0.5 tanh(α_l)`，scalar / channel 两种，零初始化。
4.2 **可吸收性命题**：`g ⊙ (W x + b) = (diag(g)W) x + g ⊙ b` → 函数空间不变，只能影响优化轨迹。
    可写成一个 Proposition + 一行证明。
4.3 六数据集结果表（`PROJECT_NOTES.md` 第四节）：scalar/channel 的 ΔHM 正负参半；Aircraft channel −11.89。
4.4 诊断（`DIAGNOSTIC_RESULTS_2026-10-22.md`）：
   - D1 fixed-one 与 none 330 个张量逐位相同 → 不是实现 bug；
   - D2 门控 LR ×0.1 / ×10 都不崩，×1 崩 → 非单调敏感；
   - D3 第 2 epoch 崩溃时门控偏离仅 5.2e-4；崩溃恰好发生在 warmup（1e-5 常数）结束、学习率跳到
     0.0035 的第一个 epoch，此时其余参数梯度范数高达 379 → 指向训练配方本身的脆弱性。
   - 第二阶段补充：Aircraft none/channel 五种子与梯度裁剪结果【填表】。
4.5 小结（过渡到第 5 节）：有效的门控必须 (i) 依赖输入，不能被吸收；(ii) 恒等初始化，保证起点即原模型；
   (iii) 不显著增加计算。

## 5 方法：输入条件跨模态门控 + 知识一致性（约 1.25 页，配一张结构图）

5.1 **ICG（Input-Conditioned Gate）**
   - `c_{l-1}(x)`：进入第 l 个视觉 block 时的 CLS token（上一 block 输出，已含图像信息）。
   - `g_l(x) = 1 + 0.5 tanh(U_l GELU(D_l LN(c_{l-1}(x))))`，`D_l: 768→r`，`U_l: r→768`，r=32。
   - `P̃_v^l(x) = g_l(x) ⊙ F_l(P_t^l)`，按通道作用于该层所有视觉 prompt token。
   - 三个性质，各配一句论证或单元测试：不可吸收（g 依赖 x）；U 零初始化 → 训练起点与 MaPLe 逐位一致；
     复用流内 CLS，无额外前向（对比 CoCoOp 需额外一次图像编码 + 每图重编码文本）。
   - 参数量：每层 2·768·r + r + 768 ≈ 50k，8 层约 0.4M（相对 MaPLe 3.55M）。【用日志实际值替换】
5.2 **KC（Knowledge Consistency）**
   - `L = L_CE + λ_t‖f_t − f_t^zs‖_1 + λ_v‖f_v − f_v^zs‖_1 + λ_kl KL(p‖p^zs)`，λ = 25/10/1（取自 PromptSRC）。
   - 冻结 CLIP 教师只在训练时使用，推理时不需要，不进 checkpoint。
5.3 结构图建议：在 MaPLe 原图基础上，视觉第 l 层左侧画出 CLS → 小 MLP → g 的支路，与 F_l 输出相乘。

## 6 实验（约 2 页）

6.1 设置：ViT-B/16、16-shot、5 epochs、SGD 0.0035、fp16、seeds 1/2/3、RTX 3090；
    六个数据集；同一 fixed split 与 few-shot cache；HM 逐种子计算后再取均值 ± std；所有超参事先固定
    （`EXPERIMENT_PLAN_V2.md`），不用 Novel 测试调参。
6.2 **主表**（核心）：行 = 数据集 + 平均；列 = CLIP zero-shot / MaPLe（论文）/ MaPLe（复现）/ +static channel /
    +ICG / +KC / +ICG+KC，每格 Base/Novel/HM。【填数】
6.3 消融：cond_scalar vs cond_channel；rank 8/32/64；KC 各项单独作用。【填数】
6.4 稳定性：Aircraft 五种子 HM 散点或箱线图；梯度裁剪前后标准差。【填数】
6.5 门控可视化：不同类别图片的 g_l(x) 均值/方差随层变化（热力图），说明门控确实依赖输入。
    【需要再写一个导出门控统计的小脚本】
6.6 开销：可训练参数、训练时间、显存峰值（`results.json` 中已有 training_seconds / peak_allocated_bytes）。

## 7 讨论与局限（约 0.5 页）

- 只评估了 11 个数据集中的 6 个，没做跨数据集迁移与域泛化。
- 3 个种子统计功效有限；部分数据集（DTD、EuroSAT）种子间波动大于方法间差异，要如实写。
- KC 权重沿用他人工作，未针对本设置调优；复现基线在 DTD/EuroSAT 上与原论文仍有差距。
- 负结果（静态门控）本身的价值：说明"加参数"不等于"加能力"，重参数化只改变优化。

## 8 结论（约 5 行）

## 附录候选

- 静态门控折叠验证（`ABLATIONS_2026-10-21.md`：CPU float64 精确、fp16 下少量预测改变）。
- 冻结投影消融表（`RESULTS_EXPANDED_2026-10-22.md`）。
- 工程可复现性：逐位等价测试、checkpoint 家族校验、预登记流程、代码地址。

## 图表清单

| 编号 | 内容 | 状态 |
|---|---|---|
| 图 1 | MaPLe + ICG 结构图 | 待画 |
| 图 2 | Aircraft channel 逐 epoch 精度与门控偏离（D3 数据） | 数据已有 |
| 图 3 | 门控热力图（层 × 通道或层 × 类别） | 待导出 |
| 表 1 | 复现 vs 原论文 | 已有 |
| 表 2 | 静态门控六数据集 | 已有 |
| 表 3 | 主结果 | 第二阶段 |
| 表 4 | 消融 | 第二阶段 |
| 表 5 | 稳定性 / 梯度裁剪 | 第二阶段 |
