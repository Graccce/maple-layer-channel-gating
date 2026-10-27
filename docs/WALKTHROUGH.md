# MaPLe 门控改进：从零走一遍

这份说明把整个项目从头到尾过一遍：先讲我们做了什么、结果怎么样，再讲原理，然后带你按代码、按命令把项目跑起来，最后讲结果怎么看、我们查了哪些东西。看完之后，你应该能自己复现，也能对着代码把每一处讲清楚。

先给结论：项目复现了 MaPLe（CVPR 2023），在它"深层文本提示投影到视觉空间"这一步后面加了一个可学习的门控，跑了六个数据集。结论是门控没有带来普遍提升，FGVCAircraft 上的 channel 模式还崩了；崩溃原因我们通过三组诊断定位到了优化轨迹层面，不是代码 bug。

## 1. 先看结果

六个数据集上的 Harmonic Mean（3 个种子的均值 ± 样本标准差，单位 %）：

| 数据集 | none | scalar | channel |
|---|---|---|---|
| OxfordPets | 96.59 ± 0.22 | 96.69 ± 0.14 | 96.74 ± 0.27 |
| DTD | 66.00 ± 2.53 | 67.37 ± 1.64 | 65.40 ± 1.97 |
| Caltech101 | 96.50 ± 0.09 | 96.47 ± 0.36 | 96.33 ± 0.14 |
| Flowers102 | 83.16 ± 1.08 | 82.49 ± 1.48 | 82.62 ± 1.82 |
| EuroSAT | 81.10 ± 4.66 | 81.58 ± 3.95 | 82.38 ± 3.29 |
| FGVCAircraft | 36.20 ± 0.37 | 35.57 ± 0.36 | 24.32 ± 10.20 |

读这张表的方式是横着比：scalar 在 DTD 上平均 +1.36 个点，channel 在 EuroSAT 上平均 +1.28 个点，看起来不错，但同一个改动换到别的数据集就是负的；FGVCAircraft 的 channel 直接从 36.20 掉到 24.32，标准差 10.20，崩得厉害。整体上，不支持"门控普遍有用"这个说法。逐种子的 Base / Novel 精确数字在 [RESULTS_EXPANDED_2026-10-22.md](RESULTS_EXPANDED_2026-10-22.md) 和 `docs/results/` 的 JSON 里。

## 2. 背景：MaPLe 在干嘛

先花一点篇幅把基线讲清楚，不然门控加在哪一步都说不明白。

CLIP 这类视觉语言大模型，全量微调很贵，还容易丢掉预训练学到的通用能力。所以有一类方法叫 prompt learning：模型完全冻结，只训练几个额外拼在输入上的可学习 token，让模型看到更适合下游任务的提示，不碰原来的权重。CoOp 只在文本侧加 prompt，VPT 只在视觉侧加。

MaPLe 在这基础上更进一步：文本侧和视觉侧都加 prompt，而且让两边的浅层 prompt 互相"翻译"。文本浅层 prompt 经过一个线性层投影成视觉浅层 prompt，视觉浅层 prompt 也投影回文本侧。到了深层，每个 Transformer block 都有自己的 prompt，文本深层 prompt 会被投影到视觉空间，当作视觉深层的 prompt 用。

我们改的就是最后这一步。原论文里深层视觉 prompt 的生成是：

```text
z_l = F_l(P_t_l)
```

`F_l` 是第 l 层的线性投影，`P_t_l` 是第 l 层的深层文本 prompt，`z_l` 是喂给视觉 Transformer 第 l 层的深层视觉 prompt。ViT-B/16 下 `PROMPT_DEPTH=9`，第 0 层是浅层耦合，剩下 8 层各有一个这样的投影。

## 3. 我们加的改动

改动很小：在 `z_l` 算出来之后、送进视觉 Transformer 之前，乘一个门控：

```text
g_l = 1 + 0.5 * tanh(alpha_l)
P_v_l = g_l * z_l
```

`alpha_l` 是新加的可学习参数，初始化为 0，所以一开始 `g_l` 恒等于 1，训练第一步和原版完全一样。tanh 把它限制在 (0.5, 1.5) 区间里。

三种模式，用配置项 `TRAINER.MAPLE.GATE_TYPE` 切换：

- `none`：不加门控，就是原版，也是默认值。
- `scalar`：每层只学一个标量，整层所有通道、所有 token 一起缩放，多 8 个参数。
- `channel`：每层学一个 768 维向量（ViT-B/16 的视觉宽度是 768），按通道缩放，同一层所有 prompt token 共享，多 8 × 768 = 6144 个参数。

这里的门控是按层共享的静态参数，不依赖输入图片，所有图片用同一套门控。它不是"看着每张图动态生成权重"的条件门控，这两个概念要分清楚。

还有一个从一开始就知道的理论问题：`z = g ⊙ (W x + b)` 这种形式，静态的 `g` 可以吸收进新的 `W' = diag(g) W` 和 `b' = g ⊙ b`，所以它并没有扩大模型的表达空间。它能产生效果，只能来自改变优化轨迹。这一点是理解后面所有结果的关键。

## 4. 代码结构：每个文件干什么

仓库根目录下，和这次改动直接相关的文件是这些：

```text
train.py                          # 训练入口，解析参数、搭好 trainer 开跑
trainers/maple.py                 # MaPLe 模型本体，门控实现在这里
trainers/gating_checkpoint.py     # 新旧 checkpoint 的校验和加载
configs/trainers/MaPLe/           # 三种门控模式的配置文件
configs/datasets/                 # 每个数据集的名称配置
scripts/maple/run_gating.sh       # 主实验的一键运行脚本
scripts/maple/profile_run.py      # 训练/评估并记录时间、显存
scripts/maple/gpu_smoke.py        # GPU 上的小规模自检
scripts/maple/analyze_gating_runs.py  # 精确统计 Base/Novel/HM 和门控分布
scripts/maple/summarize_gating.py     # 单个 run 的快速汇总
datasets/                         # 各数据集的数据加载器
clip/                             # CLIP 模型代码（官方原版，未改）
tests/                            # CPU 单元测试
docs/                             # 所有结果文档和 JSON 证据
```

### 4.1 门控具体在哪几行

打开 `trainers/maple.py`，看 `MultiModalPromptLearner` 类。三个关键位置：

第一处是初始化，在 `__init__` 里建完 `compound_prompt_projections` 之后：

```python
self.gate_type = getattr(cfg.TRAINER.MAPLE, "GATE_TYPE", "none")
if self.gate_type not in ("none", "scalar", "channel"):
    raise ValueError("TRAINER.MAPLE.GATE_TYPE must be none, scalar, or channel")
if self.gate_type != "none":
    self.gate_alphas = nn.ParameterList([
        nn.Parameter(torch.zeros(() if self.gate_type == "scalar" else (layer.out_features,)))
        for layer in self.compound_prompt_projections
    ])
    self.register_buffer("_gate_schema", torch.tensor(
        [1, 1 if self.gate_type == "scalar" else 2], dtype=torch.int64))
```

这段做了四件事：从配置读模式；对非法值直接报错；`scalar` 每层建一个零维参数、`channel` 每层建一个 `out_features` 维（768）参数，全部零初始化；再存一个 `_gate_schema` 标记，用来区分 checkpoint 家族。门控数量是跟着 `compound_prompt_projections` 的长度走的，不是硬编码 8。

第二处是前向，在 `forward()` 的循环里：

```python
for index, layer in enumerate(self.compound_prompt_projections):
    z = layer(self.compound_prompts_text[index])
    if self.gate_type != "none":
        alpha = self.gate_alphas[index]
        gate = (1.0 + 0.5 * torch.tanh(alpha.float())).to(dtype=z.dtype)
        z = z * gate
    visual_deep_prompts.append(z)
```

注意 `tanh` 是在 fp32 下算的，再转回投影输出的 dtype，这样配合 fp16/AMP 不会出数值问题。`none` 模式这段 `if` 直接跳过，前向路径和原版完全一致。

第三处是 checkpoint 加载保护，`_load_from_state_dict` 里先调了 `validate_gate_state`。这意味着原始 MaPLe 的 checkpoint、scalar 的、channel 的不能交叉加载，家族、形状对不上会直接报错，不会默默拿零初始化补上。

### 4.2 运行脚本拆开看

`run_gating.sh` 的用法是：

```bash
bash scripts/maple/run_gating.sh DATA_ROOT DATASET MODE SEED RUN_ID {smoke|formal} [standard|frozen_projection]
```

它内部依次做了这些事：检查参数和数据目录、检查固定划分文件存在、确认有 GPU、创建输出目录（已经存在就拒绝，防止覆盖）、把当前 commit 和源码打包存进去、调 `profile_run.py` 跑 base 训练、再对 base / new 两个子集各做一次评估、最后调 `summarize_gating.py` 出 `results.json`。

`smoke` 阶段是 1 个 epoch、warmup 关掉、workers 0、测试 batch 4，用来快速验证流程；`formal` 才是 5 个 epoch 的正式配置。最后一个可选参数默认 `standard`，`frozen_projection` 是冻结深层投影的消融，结果会单独放到 `ablation` 目录，不混进主比较。

## 5. 环境怎么搭

正式实验的环境是：RTX 3090（24G）、Python 3.10.8、PyTorch 2.1.2 + cu118、torchvision 0.16.2。训练用 fp16。

依赖分两层：

- PyTorch、torchvision、CLIP、Dassl 由环境提供。Dassl 是官方 MaPLe 的框架，`train.py` 直接 `from dassl.engine import build_trainer`。能跑原版 MaPLe 的 AutoDL 镜像可以直接用，不用再装。
- 仓库 `requirements.txt` 只列了 `ftfy`、`regex`、`tqdm` 三个小包，环境里没有就补一下。

数据不进 Git。数据目录的组织方式是：`DATA_ROOT` 下每个数据集一个文件夹，里面放图片，再放对应的固定划分 JSON。`run_gating.sh` 里写死了每个数据集要用的划分文件：

| 数据集 | 目录名 | 固定划分文件 |
|---|---|---|
| OxfordPets | oxford_pets | split_zhou_OxfordPets.json |
| DTD | dtd | split_zhou_DescribableTextures.json |
| Flowers102 | oxford_flowers | split_zhou_OxfordFlowers.json |
| Caltech101 | caltech-101 | split_zhou_Caltech101.json |
| EuroSAT | eurosat | split_zhou_EuroSAT.json |
| FGVCAircraft | fgvc_aircraft | images_variant_train.txt |

这些划分来自 CoOp 的公开配置，保证和原论文的 base/new 切分一致。DTD 的下载和校验有专门脚本 `scripts/maple/prepare_dtd.py`，其他数据集按官方 MaPLe/CoOp 的方式准备即可。

## 6. 一步步跑起来

### 6.1 先把代码弄下来

```bash
git clone https://github.com/Graccce/maple-layer-channel-gating.git
cd maple-layer-channel-gating
```

### 6.2 先做静态和 CPU 检查

```bash
python scripts/check_repository.py
python -m unittest discover -s tests -v
```

这两步不碰 GPU，跑得快，先确认代码和环境没坏。测试覆盖了门控的三种模式初始化、checkpoint 家族校验、配置解析、结果统计这些。

### 6.3 有卡了先做 GPU 自检

```bash
for mode in none scalar channel; do
  python scripts/maple/gpu_smoke.py "$mode"
done
```

这一步会用真实 CLIP 跑一个 batch，确认三种模式的前向、反向、优化器更新都正常，恒等初始化时 logits 和原版逐位一致。

### 6.4 跑 OxfordPets 单种子 smoke

```bash
export DATA_ROOT=/absolute/path/to/datasets
for mode in none scalar channel; do
  bash scripts/maple/run_gating.sh "$DATA_ROOT" oxford_pets "$mode" 1 pets_debug01 smoke
done
```

这一步是 1 epoch 的完整流程演练，确认数据路径、checkpoint 保存恢复、base/new 评估、计时和显存字段都对。

### 6.5 正式实验

过了 smoke 之后，把 stage 换成 `formal`，换一个新的 RUN_ID，按需要跑种子 1/2/3：

```bash
for seed in 1 2 3; do
  for mode in none scalar channel; do
    bash scripts/maple/run_gating.sh "$DATA_ROOT" oxford_pets "$mode" "$seed" pets_formal01 formal
  done
done
```

换数据集就换第二个参数，比如 `dtd`、`eurosat`。每个 RUN_ID 只跑一遍，脚本拒绝复用已有输出目录，想重跑就换新 RUN_ID，不会覆盖旧结果。

### 6.6 汇总结果

```bash
python scripts/maple/analyze_gating_runs.py \
  output/gating_v1/formal/pets_formal01/oxford_pets --epoch 5 --output pets_formal01.json
```

`analyze_gating_runs.py` 用日志里的 correct/total 精确计算，不用显示到一位小数的准确率；`summarize_gating.py` 适合看单个 run 的快速结果。

## 7. 输出目录和结果文件怎么看

一次 formal run 结束后，目录长这样：

```text
output/gating_v1/formal/pets_formal01/oxford_pets/channel/seed1/
├── base_commit.txt          # 这次运行对应的 git commit
├── source.patch             # 相对 HEAD 的改动
├── source.tar.gz            # 关键源码打包，留作审计
├── train_base/              # base 训练的输出、checkpoint、日志
├── test_base/               # base 子集的评估输出
├── test_new/                # new 子集的评估输出
├── train_metrics.json       # 训练时间、显存峰值
├── test_base_metrics.json
├── test_new_metrics.json
└── results.json             # summarize_gating.py 生成的结果
```

`results.json` 里有 Base / Novel 准确率和 Harmonic Mean，`train_metrics.json` 里有任务耗时和 PyTorch allocated 显存峰值。注意显存是 allocated 峰值，不是整卡 nvidia-smi 占用。

## 8. 指标怎么看

这个任务是 base-to-novel 泛化：训练时只见过 base 类，novel 类是没见过的类，用来衡量泛化。所以每个 checkpoint 要在 base 和 novel 两个测试集上分别评一次。

三个指标：Base Accuracy、Novel Accuracy、Harmonic Mean。HM 是两者的调和平均：

```text
HM = 2 * Base * Novel / (Base + Novel)
```

调和平局会惩罚两边差距大的情况，比简单平均更能反映"base 和 novel 是不是都还行"。我们的比较方式是先对每个种子算 HM，再对三个种子求均值和样本标准差，这样种子之间的波动也能看出来。

## 9. 我们查了什么（诊断实验）

FGVCAircraft 的 channel 崩得太厉害，所以补了三组诊断，都是先定好要查什么再跑，没有边跑边调超参。结论是崩溃来自优化轨迹，不是实现 bug。

- D1（固定恒等门控）：把 channel 的 alpha 冻结成 0，也就是门控恒等于 1，和同种子 none 比 checkpoint。三个种子的 330 个共享张量逐位相同，说明门控等于 1 时前向路径和原版没区别。
- D2（门控学习率消融）：只把门控学习率乘 0.1 或 10。FGVCAircraft 的崩溃在两个乘子下都消失，唯独默认 1.0 崩；EuroSAT 三个乘子都稳定。说明它对学习率非单调敏感。
- D3（轨迹记录）：FGVCAircraft channel seed 2 逐 epoch 记门控偏离和梯度。精度在第 2 个 epoch 崩，但那时门控偏离 1 只有 5.2e-4，几乎还是恒等，门控梯度也比其他参数小两三个数量级。

具体数字和脚本见 [DIAGNOSTIC_RESULTS_2026-10-22.md](DIAGNOSTIC_RESULTS_2026-10-22.md)。

## 10. 结论和后续

这次改动的最终结论是：验证了一个合理但失败的假设——静态门控因为能被吸收进线性投影，没有扩大表达空间，实测也没有带来稳定收益，反而在个别数据集上把优化搞崩了。负结果本身查清楚了原因，就是有价值的。

如果继续做，更有希望的方向是把门控改成输入条件的：让门控由当前图片的特征生成，这样它依赖输入，没法被静态吸收，才真正增加表达能力。动手之前还需要补几件事：gate 学习率消融做成正式消融、补纯 fp32 / AMP 验证、种子从 3 个加到 5 个、相关工作再查一轮（SSF 已经有静态通道缩放的做法，所以定位成"研究重参数化的优化影响"，不碰首创）。

## 11. 常见问题

- 数据在哪下载？数据不进 Git。OxfordPets / Flowers102 / Caltech101 / EuroSAT / FGVCAircraft 按官方 MaPLe 或 CoOp 的说明下载，固定划分用 CoOp 公开的 `split_zhou_*.json`；DTD 用 `scripts/maple/prepare_dtd.py`，它会做 MD5 和划分校验。
- checkpoint 加载报家族不匹配怎么办？这是故意的保护，说明你在拿别的模式或旧版的权重加载。none / scalar / channel 的 checkpoint 不互通，先确认配置里的 `GATE_TYPE` 和权重来源一致。
- 回退到原版怎么做？配置里 `TRAINER.MAPLE.GATE_TYPE none` 就行，`none` 是默认值，前向路径和原版完全一致，原始基线一直没动过。
- 数据、原始日志和模型权重在哪？都在 AutoDL 的实验机上，GitHub 只放代码、配置和聚合结果，不放大文件和权重。
