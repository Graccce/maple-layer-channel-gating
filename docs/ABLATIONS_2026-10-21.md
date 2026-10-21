# OxfordPets 门控消融（2026-10-21）

在原有三种子正式实验之后，新增 seed 1 的固定门控训练对照，以及训练后门控折叠的完整 CLIP 评估。使用相同 RTX 3090、数据划分、16 shots、ViT-B/16、2 context tokens、PROMPT_DEPTH=9、SGD 配方和 5 epoch；配置沿用[正式实验](RESULTS_2026-10-20.md)。这是单种子诊断，不是新增三种子均值。

## 固定门控为 1

保留 scalar/channel 的运算路径，将 alpha 置零并冻结，其他参数照常训练。已有优化器中的冻结 alpha 没有梯度，不会更新。训练后 checkpoint 的 8 个 alpha 均为零；与原 seed-1 none checkpoint 比较，328 个共享状态张量逐元素完全相同（不比较按类别重建的固定 token 缓冲区）。

| 对照 | Base correct/total | Novel correct/total | Base (%) | Novel (%) | HM (%) |
|---|---:|---:|---:|---:|---:|
| none（原正式 seed 1） | 1794/1881 | 1741/1788 | 95.3748 | 97.3714 | 96.3627 |
| scalar fixed-one | 1794/1881 | 1741/1788 | 95.3748 | 97.3714 | 96.3627 |
| channel fixed-one | 1794/1881 | 1741/1788 | 95.3748 | 97.3714 | 96.3627 |

该对照表明，本次恒等门控运算和参数注册没有单独改变训练结果。它不证明学习门控具有统计显著收益，也不替代更多种子或数据集。固定门控 checkpoint 的家族仍是 scalar/channel，不能冒充已学习的门控结果，也不能直接作为 none checkpoint 加载。

## 训练后投影折叠：完整 CLIP 的 fp16 评估

从已有 scalar/channel seed-1 epoch-5 checkpoint 加载全部模型。每个测试 batch 使用同一输入分别通过原模型和折叠模型：W'=diag(g)W，b'=g⊙b，折叠路径跳过门控乘法；其余 CLIP 参数、提示和精度相同。完整遍历 Base 与 Novel，核对 logits、预测和正确数。折叠仅存在内存中，不改写磁盘 checkpoint，不绕过正常家族检查。

| 模式 | 子集 | 样本数 | 预测变化数 | 原正确数 → 折叠正确数 | 最大绝对 logit 差 |
|---|---|---:|---:|---:|---:|
| scalar | base | 1881 | 0 | 1799 → 1799 | 0.0546875 |
| scalar | new | 1788 | 1 | 1745 → 1744 | 0.0546875 |
| channel | base | 1881 | 1 | 1796 → 1795 | 0.0781250 |
| channel | new | 1788 | 3 | 1742 → 1741 | 0.0468750 |

Scalar 的 Base 正确数不变，Novel 少 1 张（约 -0.0559 个百分点）；channel 的 Base 少 1 张（约 -0.0532 个百分点），Novel 少 1 张。Channel Novel 有 3 张预测变化，但正确数净减少 1，不能把预测变化数直接当作准确率损失。

所有 logits 均有限。float64 投影级代数验证与完整 CLIP fp16 数值结果应分开：实数中的可吸收性仍成立，但有限精度的运算顺序与舍入会传播差异，不能声称折叠后逐元素一致，也不能将该内存诊断当作可无损部署的 checkpoint 转换。

## 验证与来源

- AutoDL 完整 CPU 测试增至 14 项，1.423 秒通过；新增测试覆盖 scalar/channel 折叠以及原始投影权重不被修改。原模型训练代码和严格加载逻辑未改。
- 两组 fixed-one 新训练及各自 Base/Novel 评估完成。完整模型折叠先完成一轮诊断，再在相同 checkpoint 上补充正确数统计；没有重复原有 9 组正式训练。
- 完整数值、运行脚本摘要及源代码归档摘要见 [JSON](results/2026-10-21-ablations.json)。训练使用的初版诊断脚本与加入正确数统计的最终脚本分别保存在 AutoDL 审计目录；后者用于最终折叠表和 14 项测试。训练算法未在两版之间改变。
- 本轮不追加可训练参数、耗时或显存的正式比较；先前报告中的相关指标仍对应原 9 组正式实验。
- 实验输出相对路径：`output/gating_v1/ablation/pets_20261021_ablation01/oxford_pets/`；审计目录：`gating_audit/gpu_20261021_01/`。原始 checkpoint 未覆盖。
- 计算完成后核对无训练/下载进程、GPU 无计算进程，已关机并在控制台回读确认。

## DTD 数据准备受阻

按照 [Oxford 官方发布](https://www.robots.ox.ac.uk/~vgg/data/dtd/)下载 625239812 字节的 DTD-R1.0.1 图像包，校验目标采用 [torchvision 0.16 源码](https://docs.pytorch.org/vision/0.16/_modules/torchvision/datasets/dtd.html)记录的 MD5 `fff73e5086ae6bdbea199a49dfb8a4c1`。按 [CoOp 数据说明](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md#dtd)获取固定 `split_zhou_DescribableTextures.json`，不改用随机新划分。

实际图像下载约 18 KB/s；固定划分链接直连和实例加速请求均超时，1 MiB 图像范围探测也在 30 秒内未完成。已终止本次慢速下载，保留未完成文件；没有解压或采用不完整数据，因此没有 DTD 训练或准确率结果。准备脚本仅在图像 MD5、5640 张/47 类、划分覆盖及互斥检查全部通过后才采用新目录。

后续先将完整官方图像包和固定划分传至实际数据根目录，验证上述摘要及结构，再运行 DTD 三模式 seed-1 smoke；通过后才进入与 OxfordPets 配方一致的三种子正式比较。无需为下载占用 GPU。

## 复现命令

以下在现有研究环境运行；为每次重跑使用新的报告路径和输出目录：

```bash
python scripts/maple/run_gate_ablation.py fixed_one NEW_FIXED_REPORT.json \
  --root "$DATA_ROOT" --seed 1 --trainer MaPLe \
  --dataset-config-file configs/datasets/oxford_pets.yaml \
  --config-file configs/trainers/MaPLe/vit_b16_gating_channel.yaml \
  --output-dir output/NEW_FIXED_RUN \
  DATASET.NUM_SHOTS 16 DATASET.SUBSAMPLE_CLASSES base

python scripts/maple/run_gate_ablation.py fold_check NEW_FOLD_REPORT.json \
  --root "$DATA_ROOT" --seed 1 --trainer MaPLe \
  --dataset-config-file configs/datasets/oxford_pets.yaml \
  --config-file configs/trainers/MaPLe/vit_b16_gating_channel.yaml \
  --output-dir output/NEW_FOLD_RUN --eval-only \
  --model-dir "$TRAINED_CHANNEL_MODEL_DIR" --load-epoch 5 \
  DATASET.NUM_SHOTS 16 DATASET.SUBSAMPLE_CLASSES new
```

报告文件已存在时脚本拒绝执行。`fold_check` 要求 eval-only；`fixed_one` 拒绝恢复或评估已有 checkpoint。完整 fp32、AMP、GPU 中断恢复、门控学习率/正则、冻结投影消融和相关工作系统检索仍待完成。
