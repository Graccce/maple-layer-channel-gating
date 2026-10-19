# MaPLe 层级—通道门控

深度学习课程研究项目：复现 [MaPLe (CVPR 2023)](https://github.com/muzairkhattak/multimodal-prompt-learning)，在深层文本提示的视觉投影后增加恒等初始化的静态门控。

**当前状态：实现与 CPU 轻量测试完成，完整 CLIP GPU 验证和正式准确率实验待运行。没有新方法性能提升或首创性结论。**

## 方法

`z_l = F_l(P_t_l)`，`g_l = 1 + 0.5*tanh(alpha_l)`，`P_v_l = g_l * z_l`。

| 模式 | 参数 | 用途 |
|---|---|---|
| `none`（默认） | 无新增参数 | 原始 MaPLe 基线 |
| `scalar` | 每个深层投影一个标量 | 逐层整体缩放 |
| `channel` | 每个深层投影一个输出通道向量 | 逐通道缩放，所有 token 共享 |

门控零初始化时等于 1。深层数量由投影列表决定，通道数取各层 `out_features`。保持原优化器和冻结策略。它是样本共享的静态参数，**不是根据图像生成的动态门控**。静态缩放可以吸收进可训练的线性投影；潜在收益来自优化过程的变化，必须实测。

## 安装与运行

已有能运行原始 MaPLe 的 AutoDL 环境可以继续使用，不必升级包。全新环境按 [环境说明](docs/ENVIRONMENT.md) 安装。保留原始数据划分和 few-shot 缓存；不要将数据提交到 Git。

```bash
git clone https://github.com/Graccce/maple-layer-channel-gating.git
cd maple-layer-channel-gating
python scripts/check_repository.py
python -m unittest discover -s tests -v
```

有卡模式下，先验证一个真实 CLIP 批次，再做 OxfordPets 的单种子一轮验证：

```bash
# 设置为你的真实数据根目录，其中应有 oxford_pets 子目录。
export DATA_ROOT=/absolute/path/to/datasets

set -e
for mode in none scalar channel; do
  python scripts/maple/gpu_smoke.py "$mode"
done
for mode in none scalar channel; do
  bash scripts/maple/run_gating.sh "$DATA_ROOT" oxford_pets "$mode" 1 pets_debug01 smoke
done
```

通过后将最后一个参数改为 `formal`，使用新 RUN_ID 跑 5 epoch，再扩展种子 1/2/3。DTD 先准备数据再改 dataset 参数。脚本没有 GPU 或数据时会退出；遇到已有输出目录拒绝覆盖。

输出：`output/gating_v1/STAGE/RUN_ID/DATASET/MODE/seedN/`。每组保存训练 checkpoint、base/new 评估、计时、GPU allocated/reserved 峰值、源代码快照及 `results.json`。

现有配置系统也可直接传递 `TRAINER.MAPLE.GATE_TYPE channel`。三份配置位于 `configs/trainers/MaPLe/vit_b16_gating_*.yaml`，其他默认参数与原配置一致。

## 实验与证据

- 指标：Base、Novel、逐种子 HM、可训练参数量、训练任务时间和峰值显存。
- 使用同一数据划分、配对种子、shots、优化器、轮数和精度比较 none/scalar/channel。
- checkpoint 家族严格匹配；只有类别相关固定 token 缓冲区在 base→novel 评估时重建。
- CPU 测试采用真实 prompt learner 和 Dassl 优化器、小型 CLIP fixture，不代表完整 CLIP GPU 通过。
- 自动 CI 仅做低成本静态检查；完整 CPU 测试需安装研究环境后执行。没有设置定时训练或自动占用 GPU。

详见 [方法说明](GATING_METHOD_NOTES.md)、[验证记录](docs/VALIDATION.md)、[实验计划](docs/EXPERIMENTS.md) 和 [待办](docs/ROADMAP.md)。

## 来源与许可证

本项目基于官方 MaPLe commit `69bce21ae8eda80ad6187534b2dce09cf6c59e17` 的完整源码快照。保留 [MIT LICENSE](LICENSE) 和 [原始 README](docs/UPSTREAM_README.md)，不把上游成果归为本项目原创。完整来源和 fixture 校验见 [UPSTREAM.md](UPSTREAM.md)。

MaPLe 为 CVPR 2023；课程“近三年”的口径需另行确认。
