# DTD 数据准备

训练前先准备完整图像和固定划分，下载无需占用 GPU。不要让缺失的固定划分触发代码自动生成新的随机划分。

本项目使用 [Oxford DTD-R1.0.1](https://www.robots.ox.ac.uk/~vgg/data/dtd/) 的 5640 张图像与 [CoOp 数据说明](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md#dtd)引用的 `split_zhou_DescribableTextures.json`。CoOp 划分为每类 60/24/36 张，共 train=2820、val=1128、test=1692；这是本实验的划分，不能与 DTD 官方原论文的其他分区协议混称。

| 文件 | 校验值 | 来源 |
|---|---|---|
| `dtd-r1.0.1.tar.gz` | MD5 `fff73e5086ae6bdbea199a49dfb8a4c1` | [torchvision 0.16 源码](https://docs.pytorch.org/vision/0.16/_modules/torchvision/datasets/dtd.html) |
| 相同归档 | SHA-256 `e42855a52a4950a3b59612834602aa253914755c95b0cff9ead6d07395f8e205` | 本项目对下载到本机与 AutoDL 的归档分别实测；均通过上述官方 MD5 |
| 固定划分 JSON | SHA-256 `f26ebeccd2b58e68d70f07a0dc39fe49bf7b69024bac34396b954dfd87969c38` | 从 CoOp 指向的公开文件下载后实测 |

本次 Oxford 直连过慢，改用保存原始归档的公开镜像，并固定其版本为 `2afd1ec578464e8d2423a3c0cff74353df43d924`。镜像不作为新的数据定义；以官方 MD5 验证内容相同后才采用。镜像可用性会变化，链接失效时不要跳过校验。

```bash
mkdir -p dtd_download
curl -L --fail -C - \
  'https://hf-mirror.com/datasets/xingslong/dtd/resolve/2afd1ec578464e8d2423a3c0cff74353df43d924/dtd-r1.0.1.tar.gz' \
  -o dtd_download/dtd-r1.0.1.tar.gz
curl -L --fail \
  'https://drive.google.com/uc?export=download&id=1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x' \
  -o dtd_download/split_zhou_DescribableTextures.json

# DATA_ROOT 必须设为实际数据根目录；其 dtd 子目录必须尚不存在。
python scripts/maple/prepare_dtd.py \
  --archive dtd_download/dtd-r1.0.1.tar.gz \
  --split dtd_download/split_zhou_DescribableTextures.json \
  --data-root "$DATA_ROOT"
```

准备器先验证归档和划分摘要，拒绝危险 tar 成员，检查 47 类、每类 120 张、划分互斥且完整覆盖，并对全部图片执行 PIL verify。全部通过后才将临时目录移入 `dtd/`，写入 `provenance.json`。已有数据集不会被覆盖；失败的临时目录保留供检查。即使输入 JSON 被重命名，安装时仍写入 MaPLe 要求的固定文件名。

Google Drive 在 AutoDL 不可访问时，可以在可访问网络的本机下载 JSON，再通过 Jupyter 文件上传到实例；上传前后应核对上述 SHA-256。不要把下载得到的 HTML 验证页当作 JSON，也不要更换数据划分来规避下载问题。

正式比较前预生成 seeds 1/2/3 的 16-shot 缓存，再让所有模式读取同一份缓存。记录并在训练后复核每份缓存摘要，可以避免首次生成缓存与后续读取缓存造成不同的随机数消耗。缓存包含实例内图像路径，不上传 GitHub；只公开摘要。

```bash
python - "$DATA_ROOT" <<'PY'
import hashlib
from pathlib import Path
import sys
from dassl.config import get_cfg_default
from dassl.utils import set_random_seed
from datasets.dtd import DescribableTextures

root = Path(sys.argv[1])
for seed in (1, 2, 3):
    set_random_seed(seed)
    cfg = get_cfg_default()
    cfg.DATASET.ROOT = str(root)
    cfg.DATASET.NUM_SHOTS = 16
    cfg.DATASET.SUBSAMPLE_CLASSES = 'all'
    cfg.SEED = seed
    DescribableTextures(cfg)
    path = root / f'dtd/split_fewshot/shot_16-seed_{seed}.pkl'
    print(path.name, hashlib.sha256(path.read_bytes()).hexdigest())
PY
```
