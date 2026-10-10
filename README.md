[![CI](https://github.com/sherifabdelrady/diffctrl/actions/workflows/ci.yml/badge.svg)](https://github.com/sherifabdelrady/diffctrl/actions)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Metric](https://img.shields.io/badge/FID-8.2%20%C2%B7%20CLIP--Score%200.31-FCE7F3)

# DiffCtrl — Controlled Diffusion Generation

**ControlNet on Stable Diffusion XL for spatially-conditioned image synthesis.
Depth, Canny, pose, and segmentation conditioning. LoRA fine-tuning per domain
with only 1,000 images. 94% user preference over unconditional SDXL.**

## Results

| Metric | Score | Baseline (SDXL) |
|--------|-------|-----------------|
| FID ↓ | **8.2** | 12.4 |
| CLIP-Score ↑ | **0.31** | 0.28 |
| Spatial Control Acc ↑ | **91.4%** | N/A |
| User Preference ↑ | **94%** | 6% |

## Architecture

```
Text Prompt + Condition Image (depth/canny/pose/seg)
       │                │
 SDXL U-Net        ControlNet (zero-conv copy)
    │  ◄──────────────────────────│
    ▼
Generated Image (1024×1024)
    │
  + LoRA adapters (style / domain fine-tuning, 1K images, 3h A100)
```

Key decisions:
- **Zero-convolution layers**: initialized to zero — ControlNet adds
  signal progressively without corrupting base SDXL weights
- **xFormers memory-efficient attention**: 40% VRAM reduction, enables
  SDXL at batch size 2 on 24GB consumer GPU
- **LoRA (r=8, α=16)**: per-domain style in 3h; DreamBooth baseline
  needs 6h and shows more overfitting
- **DDIM sampler (20 steps)**: same quality as DDPM at 1000 steps;
  4s/image on A100

## Quick Start

```bash
git clone https://github.com/sherifabdelrady/diffctrl
cd diffctrl
pip install -r requirements.txt

# Generate with depth conditioning
python demo.py --condition depth --prompt "a cyberpunk street at night, rain"

# Fine-tune LoRA on your images
python train_lora.py --images ./data/my_style/ --steps 1000 --output ./lora_weights/

# Generate with your LoRA
python demo.py --lora ./lora_weights/ --prompt "a portrait in my style"
```

## Conditioning Modes

| Mode | Input | Use Case |
|------|-------|----------|
| `canny` | edge map | product shots, architecture |
| `depth` | depth map | 3D-consistent generation |
| `pose` | skeleton | character control |
| `seg` | segmentation mask | scene composition |

## Portfolio

→ [sherifabdelrady.github.io/cv-portfolio](https://sherifabdelrady.github.io/cv-portfolio/)

## License

MIT © Sherif Abd El-Rady
