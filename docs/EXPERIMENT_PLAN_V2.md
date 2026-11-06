# 第二阶段实验预登记（2026-11-06）

第一阶段结论：静态 scalar/channel 门控可被吸收进可训练投影 W，不扩大函数空间；六数据集结果
有正有负，FGVCAircraft channel 的崩溃定位为优化轨迹敏感（见 `DIAGNOSTIC_RESULTS_2026-10-22.md`）。
第二阶段只改变下列三个因素，其余配方（ViT-B/16、16-shot、N_CTX=2、PROMPT_DEPTH=9、5 epochs、
SGD lr 0.0035、fp16、seeds 1/2/3、同一 fixed split 与 few-shot cache）与第一阶段完全一致。

## 方法变量

| 标签 | 改动 | 配置 |
|---|---|---|
| `none` | 原版 MaPLe（直接复用第一阶段配对结果） | — |
| `cond_channel` | 输入条件门控：`g_l(x) = 1 + 0.5 tanh(U_l GELU(D_l LN(cls_{l-1}(x))))`，按通道作用于第 l 层视觉深层 prompt | `GATE_TYPE cond_channel`, `GATE_RANK 32` |
| `kc` | 零样本知识一致性：`L = CE + 25·L1(f_t, f_t^zs) + 10·L1(f_v, f_v^zs) + 1·KL(p ‖ p^zs)` | `KC_TEXT/IMAGE/LOGIT_WEIGHT 25/10/1` |
| `cond_channel_kc` | 两者组合 | 以上两者 |
| `*_clip` | 全局梯度范数裁剪（稳定性实验） | `GRAD_CLIP c` |

- `cls_{l-1}(x)` 是进入第 l 个带 prompt 视觉 block 时的 CLS token（上一 block 输出），因此无需
  CoCoOp 那样额外跑一次图像编码器。U 零初始化 → 第一步 g ≡ 1，与 `none` 前向逐位一致（单元测试
  `test_identity_at_init_matches_none_bitwise`）；D 由私有 generator 初始化，不推进全局 RNG，
  数据顺序与其他参数初始化与 `none` 配对。
- KC 的权重直接沿用 PromptSRC（ICCV 2023）公开配置的 25/10/1，零样本文本特征使用与本项目
  zero-shot 参考相同的单模板 `CUSTOM_TEMPLATES`。**不根据 Novel 测试结果调这些超参。**
- `GATE_RANK=32` 事先固定；rank / cond_scalar 只作为事后消融，单独报告。

## 实验列表

1. 环境一致性：EuroSAT `none` seed 1 在新代码目录重跑，checkpoint 与第一阶段逐张量比较。
2. 稳定性（FGVCAircraft）：`none`、`channel` 补 seeds 4/5；`none_clip`、`channel_clip` seeds 1–5。
   裁剪阈值 c 依据 `none` 训练日志中的梯度范数（训练集统计，与测试无关）事先确定后再启动。
   **已确定 c = 10**（2026-11-06 11:20，依据 Aircraft none seed 4：warmup 期均值约 18、峰值 38；
   lr 升至 3.5e-3 后稳定在 4–5。c≈2× 稳定期典型值，只截断尖峰）。
3. 主比较：六数据集 × {`cond_channel`, `kc`, `cond_channel_kc`} × seeds 1/2/3 = 54 次训练。
4. 消融（主比较完成后）：`cond_scalar`、rank 8/64、仅 KC 文本项等，选三个代表数据集。

## 报告规则

- 每个种子先算 HM，再报均值 ± 样本标准差；ΔHM 为与同数据集同种子 `none` 的配对差。
- 保留所有失败与负结果；不替换种子，不挑数据集。
- 新输出写入 `output/v2/`，checkpoint 评估后只保留 SHA-256（磁盘限制），`KEEP_CKPT=1` 例外。

## 追加预登记：教师可信度加权 KC（2026-11-06 13:20，`kc_rw`）

**动机与诚实说明**：主比较显示固定权重 KC 在 DTD/Flowers 提升 Novel，但在 EuroSAT HM −7.33。
该方法是**看到上述测试结果后**提出的（数据集层面的事后设计），论文中必须如实写明；但其公式与超参
不再根据任何测试结果调整，且只使用训练集信息。

- 样本权重 `w_i = p_zs(y_i | x_i)`：冻结 CLIP 用零样本模板对**训练图像**给出的真实类别概率（用训练标签，不碰测试集）。
- 图像特征 L1 项、logits KL 项按样本乘 `w_i`；文本 L1 项（类级别）乘训练中已见样本 `w` 的累计均值 `r̄`。
- λ 仍为 25/10/1，不调；`KC_REWEIGHT=none` 时与原 KC 数学上等价。
- 实验：六数据集 × seeds 1/2/3，标签 `kc_rw`，配置 `KC_* 25/10/1 + KC_REWEIGHT teacher_conf`。
- 判定：报告与 none 和固定 KC 的配对 ΔHM；无论正负都写入论文。
