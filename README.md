# MaPLe 层级—通道门控

深度学习课程项目。在 [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning)（CVPR 2023）的基础上做了一点小改动：给深层文本提示投影到视觉空间的那一步加一个可学习的门控，看看对 base-to-novel 泛化有没有影响。

详细的原理、改动、实验数据和后续计划见 [docs/PROJECT_NOTES.md](docs/PROJECT_NOTES.md)。

改动本身很小。原来深层提示的投影是 `P_v = F(P_t)`，我在后面乘一个门控：

```
g = 1 + 0.5 * tanh(alpha)
P_v = g * F(P_t)
```

`alpha` 初始是 0，所以一开始 `g = 1`，和原版完全一样。三种模式：`none`（原版，默认）、`scalar`（每层一个标量，整层缩放）、`channel`（每层一个 768 维向量，按通道缩放，所有 token 共享）。门控是随模型一起学习的静态参数，不依赖输入图片，不是动态门控。

## 实验结果

在 OxfordPets、DTD、Caltech101、Flowers102、EuroSAT、FGVCAircraft 六个数据集上跑了 none / scalar / channel 各 3 个种子的对比（16 shots、5 epochs、冻结 CLIP 编码器，和原论文一致）。结果比较散：

- channel 在 EuroSAT 上平均 HM 高 1.28 个点，但在 FGVCAircraft 上掉 11.9 个点，种子之间波动很大；
- scalar 在 DTD 上平均高 1.36 个点，其他数据集大多略降。

六个数据集整体看，不支持"门控普遍有用"这个说法。FGVCAircraft 上 channel 为什么崩也查了：把门控固定成 1 时输出和 none 逐位一致，说明不是实现 bug；把门控的学习率调大或调小，崩溃就消失；逐 epoch 看，精度崩的时候门控还几乎等于 1。所以更像是优化过程的问题，不是前向实现的问题。具体数字见 [docs](docs/) 下的结果文档。

## 安装与运行

能跑原版 MaPLe 的 AutoDL 环境可以直接用，不用升级包。全新环境按 [环境说明](docs/ENVIRONMENT.md) 装。数据和 few-shot 缓存不进 Git。

```bash
git clone https://github.com/Graccce/maple-layer-channel-gating.git
cd maple-layer-channel-gating
python scripts/check_repository.py
python -m unittest discover -s tests -v
```

有卡的情况下，先验一个真实 CLIP 批次，再跑 OxfordPets 单种子一轮：

```bash
export DATA_ROOT=/absolute/path/to/datasets   # 里面要有 oxford_pets 子目录

set -e
for mode in none scalar channel; do
  python scripts/maple/gpu_smoke.py "$mode"
done
for mode in none scalar channel; do
  bash scripts/maple/run_gating.sh "$DATA_ROOT" oxford_pets "$mode" 1 pets_debug01 smoke
done
```

过了之后把最后一个参数改成 `formal`，用新 RUN_ID 跑 5 epoch，再扩展到种子 1/2/3。输出在 `output/gating_v1/STAGE/RUN_ID/DATASET/MODE/seedN/`，每个目录存 checkpoint、base/new 评估、计时、显存峰值和 `results.json`。配置系统也可以直接传 `TRAINER.MAPLE.GATE_TYPE channel`。

## 来源与许可证

基于官方 MaPLe commit `69bce21ae8eda80ad6187534b2dce09cf6c59e17` 的源码快照，保留 [MIT LICENSE](LICENSE) 和 [原始 README](docs/UPSTREAM_README.md)。完整来源和 fixture 校验见 [UPSTREAM.md](UPSTREAM.md)。
