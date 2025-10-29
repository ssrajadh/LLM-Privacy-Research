# LLM Privacy Research: Adversarial Image Generation

Research project investigating privacy vulnerabilities in face recognition systems using diffusion-based adversarial image generation.

## Overview

This project generates forged facial images using Stable Diffusion img2img to evaluate the robustness of face recognition and multimodal LLM systems against adversarial attacks. The goal is to create realistic variations of celebrity faces while preserving or manipulating specific attributes to test privacy boundaries.

## Project Structure

```
LLM-Privacy-Research/
├── data/
│   ├── list_attr_celeba.txt       # CelebA attribute annotations (40 attributes)
│   ├── list_identity_celeba.txt   # Identity mappings for CelebA images
│   ├── raw/                        # Original CelebA images (202,599 images)
│   ├── organized/                  # Filtered dataset organized by identity
│   │   ├── manifest.csv           # Identities with multiple images
│   │   └── {identity_id}/         # Subdirectories per identity
│   └── forgeries/                  # Generated forged images
│       ├── mapping.json           # Detailed mapping of real→forged images
│       └── mapping.csv            # CSV version for easier viewing
├── scripts/
│   ├── generate_forged_img2img.py  # Main forgery generation script
│   ├── organize_identities.py      # Organize dataset by identity
│   ├── prepare_shuffled_sets.py    # Prepare evaluation sets
│   ├── embedding_attack.py         # Embedding-based attack evaluation
│   ├── evaluate_attack.py          # Attack success rate evaluation
│   └── validate_attributes.py      # Attribute validation utilities
└── requirements.txt                # Python dependencies
```

## Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for faster generation)
- ~30GB disk space for CelebA dataset
- ~10GB for model weights

### Installation

```bash
# Clone the repository
git clone https://github.com/ssrajadh/LLM-Privacy-Research.git
cd LLM-Privacy-Research

# Install dependencies
pip install -r requirements.txt

# Download CelebA dataset
# Place images in data/raw/
# Place list_attr_celeba.txt and list_identity_celeba.txt in data/
```

### Dependencies

Key packages:
- `torch>=2.0.0` - PyTorch for model inference
- `diffusers>=0.21.0` - Hugging Face Diffusers for Stable Diffusion
- `transformers>=4.30.0` - Model loading and tokenization
- `pillow>=9.0.0` - Image processing
- `numpy>=1.21.0` - Numerical operations
- `pandas>=1.3.0` - Data manipulation

## Usage

### 1. Organize Dataset by Identity

First, organize the raw CelebA images by identity (groups images of the same person):

```bash
python scripts/organize_identities.py \
  --in-dir data/raw \
  --meta data/list_identity_celeba.txt \
  --out-dir data/organized \
  --min-images 20
```

This creates subdirectories in `data/organized/` with identities that have at least 20 images.

### 2. Generate Forged Images

Generate adversarial forgeries using Stable Diffusion img2img:

```bash
python scripts/generate_forged_img2img.py \
  --model runwayml/stable-diffusion-v1-5 \
  --in-dir data/organized \
  --meta data/list_attr_celeba.txt \
  --identity-meta data/list_identity_celeba.txt \
  --out-dir data/forgeries \
  --d 3 \
  --denoise 0.45 \
  --template "a high-quality portrait photo of a person, {Male}, {Young}, {Black_Hair}, {Smiling}" \
  --steps 50 \
  --cfg-scale 7.5 \
  --limit 100
```

**Key Parameters:**

- `--model`: Stable Diffusion model (default: runwayml/stable-diffusion-v1-5)
- `--in-dir`: Input directory with real images
- `--meta`: Path to attribute metadata file
- `--identity-meta`: Path to identity mapping file (for organized directories)
- `--out-dir`: Output directory for generated forgeries
- `--d`: Number of forgeries per real image (default: 3)
- `--denoise`: Denoising strength 0-1 (lower=preserve identity, higher=more variation)
- `--template`: Prompt template with attribute placeholders
- `--steps`: Number of inference steps (higher=better quality, slower)
- `--cfg-scale`: Classifier-free guidance scale (default: 7.5)
- `--limit`: Limit number of images to process (for testing)

**Denoising Strength Guide:**
- `0.3-0.4`: Strong identity preservation, subtle changes
- `0.45-0.55`: Balanced (recommended for most cases)
- `0.6-0.8`: Significant variation, weaker identity preservation

**Inference Steps Guide:**
- `20-30`: Fast generation, lower quality
- `40-60`: Balanced quality and speed (recommended)
- `70-100`: High quality, slower generation

### 3. Attribute Template Format

The `--template` parameter supports CelebA attribute placeholders:

Available attributes (from list_attr_celeba.txt):
- Physical: `{Young}`, `{Male}`, `{Bald}`, `{Black_Hair}`, `{Blond_Hair}`, `{Brown_Hair}`, `{Gray_Hair}`
- Facial: `{Smiling}`, `{Eyeglasses}`, `{Attractive}`, `{High_Cheekbones}`, `{Pointy_Nose}`
- Accessories: `{Wearing_Earrings}`, `{Wearing_Hat}`, `{Wearing_Lipstick}`, `{Wearing_Necklace}`
- Expression: `{Mouth_Slightly_Open}`, `{Smiling}`
- And 30+ more attributes...

**Example Templates:**

```bash
# Basic portrait
--template "a high-quality portrait photo of a person"

# With specific attributes
--template "a portrait photo of a person, {Male}, {Young}, {Smiling}, professional lighting"

# Style-specific
--template "a {Young} person with {Black_Hair}, {Smiling}, studio photograph, high resolution"

# Multiple attribute variations
--template "headshot of a person, {Attractive}, {High_Cheekbones}, {Wearing_Lipstick}"
```

### 4. Output Format

The script generates:

**Images:**
- Forged images saved as: `{original_name}_forg_{seed}.png`
- Example: `000001_forg_772618682.png`

**Mapping Files:**

`mapping.json` - Complete metadata:
```json
[
  {
    "real": "000001.jpg",
    "forged": ["000001_forg_123.png", "000001_forg_456.png", "000001_forg_789.png"],
    "prompt": "a high-quality portrait photo of a person, Young, Smiling",
    "seeds": [123, 456, 789],
    "model": "runwayml/stable-diffusion-v1-5",
    "cfg_scale": 7.5,
    "steps": 50,
    "denoising_strength": 0.45
  }
]
```

`mapping.csv` - Flattened for analysis:
```csv
real,prompt,model,cfg_scale,steps,denoising_strength,forged1,seed1,forged2,seed2,forged3,seed3
000001.jpg,"a portrait...",runwayml/stable-diffusion-v1-5,7.5,50,0.45,000001_forg_123.png,123,...
```

## Troubleshooting

### NSFW Filter Blocking Images

The safety checker is disabled by default for research purposes. If you still encounter issues:
- The code already sets `safety_checker=None`
- This is appropriate for academic research on facial recognition

### Melted/Mosaic Images on GPU

If generated images look distorted:
1. Increase `--steps` to 50-100
2. Adjust `--denoise` (try 0.4-0.5)
3. Ensure you're using CUDA with float16 (automatic in the script)
4. Check GPU memory: `nvidia-smi`

### Out of Memory

If you run out of GPU memory:
```bash
# Reduce batch processing or enable CPU offload
# The script processes images one at a time, so this shouldn't happen
# If it does, try lowering --steps or use a smaller model
```

### No Images Generated

Check:
1. Input directory exists and contains images: `ls data/organized/*/`
2. Metadata files are present: `ls data/*.txt`
3. Use `--limit 2` for testing
4. Check that identity mapping loaded: look for "Loaded identity mapping" in output

## Research Applications

This tool supports research in:

1. **Privacy Attack Evaluation**: Test face recognition robustness against adversarial inputs
2. **Multimodal LLM Security**: Evaluate vision-language models on manipulated faces
3. **Synthetic Data Generation**: Create privacy-preserving datasets
4. **Attribute Manipulation**: Study controllable face generation
5. **Identity Preservation**: Analyze trade-offs between realism and identity retention
