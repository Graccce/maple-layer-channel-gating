# Environment

The implementation was initially tested in an existing Python/PyTorch AutoDL environment with torch 2.1.2+cu118 and torchvision 0.16.2+cu118. Existing working environments should not be upgraded just for this change.

For a new Linux environment, the following is a reproducible starting point, not a claim that every GPU/runtime combination has been tested:

```bash
conda create -n maple-gating python=3.10 -y
conda activate maple-gating
pip install numpy==1.26.4
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch
git checkout c61a1b570ac6333bd50fb5ae06aea59002fb20bb
pip install -r requirements.txt
pip install -e .
cd ..
python scripts/check_repository.py
python -m unittest discover -s tests -v
```

For CPU-only tests choose PyTorch's CPU wheels instead. Tests use CPU bfloat16 autocast; they do not validate CUDA fp16 or CUDA AMP. Full CLIP inference may download pretrained weights through the upstream loader.

The upstream shallow projection is unconditionally converted to half precision. This project preserves that baseline behavior; a pure fp32 full-model path may encounter dtype mismatch. The fixture normalizes that shallow projection for lightweight CPU testing only. Use the original fp16 experiment configuration for the initial GPU comparison. Test AMP separately before reporting it.

Data preparation follows [DATASETS.md](DATASETS.md). Pass your actual data root to scripts; no cloud instance address, password, project path, or dataset location is embedded in this repository.
