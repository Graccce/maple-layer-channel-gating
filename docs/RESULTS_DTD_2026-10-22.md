# DTD 三模式配对实验（2026-10-22）

完成三模式 × seeds 1/2/3，共 9 组正式运行；此前三组 seed-1 单轮 smoke 全部通过。结果不支持“通道粒度更细就普遍更好”的结论。

## 配方与数据

复用原训练入口、三份门控配置与 `run_gating.sh`。ViT-B/16，16 shots，2 context tokens，PROMPT_DEPTH=9，fp16；SGD，学习率 0.0035，momentum 0.9，weight decay 0.0005，5 epoch，warmup 1 epoch；训练 batch=4、测试 batch=100、workers=8，OMP_NUM_THREADS=4。RTX 3090（24576 MiB），torch 2.1.2+cu118。模型、优化器和加载逻辑未改。

图像包与官方 MD5 一致，固定 CoOp split 的 SHA-256、47 类/5640 张、完整互斥覆盖及全部图片 PIL verify 均通过。train/val/test=2820/1128/1692；Base 测试 864 张、Novel 828 张。三个种子的 16-shot 缓存均在任何训练之前生成，三模式复用同一缓存，训练后摘要未变。[数据准备方法](DTD_DATA_PREPARATION.md)给出可复用命令和校验值。

同一 epoch-5 checkpoint 分别做 Base 与 Novel test。准确率由 correct/total 精确计算；先逐种子计算 HM，再报告三种子的均值与样本标准差。旧 `results.json` 的显示舍入值不用于此表。

## 正式结果

| 模式 | Base (%) | Novel (%) | HM (%) | 可训练参数 | 任务时间 (s) | 峰值 allocated (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| none | 80.7099 ± 0.6375 | 55.8776 ± 3.3931 | 66.0045 ± 2.5281 | 3,555,072 | 28.5788 ± 0.6107 | 928.2783 |
| scalar | 80.0926 ± 1.5046 | 58.2126 ± 3.1238 | 67.3659 ± 1.6439 | 3,555,080 | 30.0111 ± 1.8974 | 928.3452 |
| channel | 79.8225 ± 1.2696 | 55.4750 ± 3.3258 | 65.4030 ± 1.9737 | 3,561,216 | 29.9758 ± 1.0326 | 928.4429 |

时间包括 Trainer.train 的末次评估，不含单独测试与初始化；显存为 PyTorch allocated 峰值，reserved 数值见 JSON。短任务、运行顺序和额外数据核查都会影响时间，不能据此得出因果性的速度或开销结论。

| 模式 | seed | Base correct/864 | Novel correct/828 | HM (%) |
|---|---:|---:|---:|---:|
| none | 1 | 700 | 492 | 68.5586 |
| none | 2 | 701 | 460 | 65.9516 |
| none | 3 | 691 | 436 | 63.5033 |
| scalar | 1 | 685 | 505 | 68.9437 |
| scalar | 2 | 707 | 454 | 65.6630 |
| scalar | 3 | 684 | 487 | 67.4909 |
| channel | 1 | 681 | 491 | 67.6801 |
| channel | 2 | 702 | 441 | 64.3434 |
| channel | 3 | 686 | 446 | 64.1853 |

### 相对 none 的配对 HM 差值（百分点）

| 模式 | seed 1 | seed 2 | seed 3 | 均值 ± 样本标准差 |
|---|---:|---:|---:|---:|
| scalar | +0.3851 | -0.2886 | +3.9876 | +1.3614 ± 2.2992 |
| channel | -0.8785 | -1.6082 | +0.6821 | -0.6015 ± 1.1700 |

Scalar 的 HM 平均比 none 高 1.3614 个百分点，Novel 平均高 2.3349 个百分点、Base 平均低 0.6173 个百分点；但配对差值标准差达 2.2992，seed 2 为负、seed 3 的 +3.9876 贡献较大。Channel 平均 HM 低 0.6015 个百分点，前两个种子为负。三种子结果不构成稳定泛化收益或统计显著性的证明。

### 与 OxfordPets 放在一起看

| 数据集 | Scalar 配对 HM 平均差 | Channel 配对 HM 平均差 |
|---|---:|---:|
| OxfordPets | +0.0995 | +0.1536 |
| DTD | +1.3614 | -0.6015 |

两数据集共 18 组正式实验。标量版本的平均方向较一致，但不是所有种子都改善；通道版本在 OxfordPets 的微小正向变化没有在 DTD 重现。不能仅展示获益数据集，也不能将更多门控参数等同于更强表达能力。保留 none 为默认模式。本次未根据 Novel 结果反复调参；后续若调整门控学习率、正则或投影冻结，应使用训练/验证数据制定方案，明确另起消融。

## 核验、饱和与来源

- 9 组训练与两类评估均返回 success，checkpoint 均为第 5 轮，浮点张量全部有限。
- 冻结权重在全部运行间一致，并与此前 OxfordPets 的冻结 CLIP 状态完全一致。各层门控均有非零学习变化。
- 每组源码归档与训练前 manifest 匹配；固定 split 与 seeds 1/2/3 的 few-shot 缓存摘要在训练后保持一致。
- 定义近饱和为 `abs(tanh(alpha)) >= 0.95`，对应 gate 不在 `(0.525,1.475)` 内。本次三种子 scalar/channel 共 18456 个 alpha，近饱和数为 0；这只排除了该阈值下的饱和现象，不解释泛化效果。
- 实验前 17 项 CPU 测试通过（1.478 秒）；数据准备器补充“下载 JSON 被重命名时仍安装固定文件名”的保护与回归测试后，最终 18 项通过（1.503 秒）。学习模型未改，已采用的数据始终使用正确文件名。运行时准备器版本及最终版本的摘要分开保存在 JSON；原版本保留在审计目录。
- 本机与 AutoDL 上的镜像归档都匹配官方 MD5；镜像固定版本、数据摘要、逐种子统计、门控分布、float64 投影折叠诊断和核验标记见 [机器可读结果](results/2026-10-22-dtd.json)。
- 补查 OxfordPets：历史 seed 2/3 的 none 首次创建缓存，另外两模式读取已有缓存。在隔离目录重放后，两种路径的数据加载后 Torch RNG 和前两个增强批次摘要相同（workers=8、原 transform）。这是有限范围的重放，不宣称完成全训练的逐批重现；未改写旧结果。

静态门控可吸收到可训练投影中；相关先例及学术定位见[相关工作核查](RELATED_WORK_2026-10-22.md)。已有完整 CLIP fp16 折叠诊断显示少量预测变化，详见[消融报告](ABLATIONS_2026-10-21.md)，不能把 float64 投影代数等价描述成 fp16 checkpoint 无损转换。

## 输出、复现与保留项

正式输出：`output/gating_v1/formal/dtd_20261022_formal01/dtd/{none,scalar,channel}/seed{1,2,3}/`。调试输出位于 `output/gating_v1/smoke/dtd_20261022_debug01/`。审计目录为 `gating_audit/gpu_20261022_01/`。原始日志、源代码快照及 checkpoint 留在 AutoDL；本机已保存结果与验证 JSON。GitHub 不包含数据、缓存、原始日志或模型权重。

```bash
# 已完成数据准备与三个种子的缓存预生成后，用新 RUN_ID 重跑。
set -e
for seed in 1 2 3; do
  for mode in none scalar channel; do
    bash scripts/maple/run_gating.sh "$DATA_ROOT" dtd "$mode" "$seed" dtd_replicate01 formal
  done
done
python scripts/maple/analyze_gating_runs.py \
  output/gating_v1/formal/dtd_replicate01/dtd --epoch 5 --output dtd_replicate01.json
```

计算结束后确认无剩余训练、下载或 GPU 计算进程，再关闭实例。存储仍有释放期限，GitHub 代码和结果摘要不是 checkpoint 备份。AMP/完整 fp32 路径、GPU 中断恢复、冻结投影/门控学习率/正则化消融、更多数据集及更充分种子仍未验证。
