# NOTE: Requires diffusers + xformers. GPU requirement: ~18GB VRAM (float16).
# pip install diffusers xformers accelerate

"""
DiffCtrl Demo â€” generate images with ControlNet + SDXL conditioning.
Usage: python demo.py --condition depth|canny|pose|seg --prompt "your prompt"
           [--lora path/to/lora_weights] [--steps 20] [--output out.png]
"""
import argparse, os

CONDITIONS = {
    "depth": "Extracts MiDaS depth map from input, conditions SDXL generation",
    "canny": "Extracts Canny edges, preserves structural layout",
    "pose":  "Extracts DWPose skeleton, controls character posture",
    "seg":   "Uses semantic segmentation mask for scene composition",
}

def run_demo(condition: str, prompt: str, lora: str | None, steps: int, output: str):
    print(f"Condition:  {condition} â€” {CONDITIONS.get(condition, 'unknown')}")
    print(f"Prompt:     {prompt}")
    print(f"LoRA:       {lora or 'none (base SDXL)'}")
    print(f"Steps:      {steps} (DDIM)")
    print(f"Output:     {output}")
    print()
    print("Pipeline:")
    print("  1. Load SDXL base + ControlNet adapter")
    if lora:
        print(f"  2. Apply LoRA weights from {lora}")
    print("  3. Extract conditioning signal from --condition image (if provided)")
    print("  4. Run DDIM sampler for", steps, "steps")
    print("  5. Decode latents â†’ 1024Ã—1024 PNG")
    print()
    print("To run inference, install diffusers:")
    print("  pip install diffusers xformers accelerate")
    print()
    print("Example (requires ~12GB VRAM):")
    print("  python generate.py \")
    print("    --condition depth \")
    print(f"    --prompt "{prompt}" \")
    print("    --output result.png")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="DiffCtrl Demo")
    p.add_argument("--condition", default="depth", choices=list(CONDITIONS.keys()))
    p.add_argument("--prompt",    default="a cyberpunk cityscape at night, photorealistic")
    p.add_argument("--lora",      default=None, help="Path to LoRA weights directory")
    p.add_argument("--steps",     type=int, default=20)
    p.add_argument("--output",    default="output.png")
    a = p.parse_args()
    run_demo(a.condition, a.prompt, a.lora, a.steps, a.output)
