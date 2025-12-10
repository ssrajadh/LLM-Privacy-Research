# Diffusion-Face-Attack

Adversarial attack framework for face recognition systems using diffusion-based image forgeries.

## Overview

This project investigates vulnerabilities in face recognition systems by generating adversarial forgeries using Stable Diffusion img2img. We systematically evaluate attack effectiveness, parameter sensitivity, and defense mechanisms to understand the security implications of diffusion-based facial image manipulation.

## Task Division

Soham: Dataset Preparation and Preprocessing, Initial Setup (added face dataset, filtered face data, organized identities) + Forged Image Generation (initial + with parameters) + Infrastructure Setup (Google Colab)

Youngju: Attack Evaluation Pipeline + Attack Test Results + Mapping-based forgeries + Defense Mechanisms + Open-set Evaluation + Defense Test Results

Harshit: Midterm Data Visualization

## Project Structure

```
Diffusion-Face-Attack/
├── data/
│   ├── list_attr_celeba.txt       # CelebA attribute annotations (40 attributes)
│   ├── list_identity_celeba.txt   # Identity mappings for CelebA images
│   ├── raw/                        # Original CelebA images (202,599 images)
│   ├── organized/                  # Filtered dataset organized by identity
│   │   ├── manifest.csv           # Identities with multiple images
│   │   └── {identity_id}/         # Subdirectories per identity
│   ├── forgeries/                  # V1 generated forged images (150 images)
│   │   ├── mapping.json           # Detailed mapping of real→forged images
│   │   └── mapping.csv            # CSV version for easier viewing
│   └── forgeries_v2/               # V2 parameter grid forgeries (2,160+ images)
│       ├── metadata_v2.json       # Comprehensive metadata with parameter grid
│       └── visualizations/        # Side-by-side comparison images
├── scripts/
│   ├── analyze_forgery_quality_v2.py    # Quality analysis with CLIP scores
│   ├── defense_mechanisms.py            # Defense strategy implementations
│   ├── embedding_attack.py              # Embedding-based attack evaluation
│   ├── evaluate_attack.py               # V1 attack success rate evaluation
│   ├── evaluate_attack_openset.py       # V2 open-set attack evaluation
│   ├── generate_forged_img2img.py       # V1 forgery generation
│   ├── generate_forged_img2img_v2.py    # V2 parameter grid generation
│   ├── generate_mapping_v2.py           # V2 mapping file generator
│   ├── generate_paper_plots.py          # Publication figure generation
│   ├── organize_identities.py           # Organize dataset by identity
│   ├── prepare_shuffled_sets.py         # Prepare evaluation sets (V1/V2 compatible)
│   ├── run_experiments.py               # Automated defense experiments
│   └── validate_attributes.py           # Attribute validation utilities
├── results/
│   ├── eval_arcface.json           # V1 ArcFace evaluation results
│   ├── eval_facenet.json           # V1 FaceNet evaluation results
│   └── experiments_v2/             # V2 defense experiment results
│       ├── experiment_summary.json # Aggregated defense results
│       ├── ANALYSIS_REPORT.md     # Defense analysis report
│       └── results_table.csv      # Defense comparison table
├── figures/                        # Generated plots and visualizations
├── requirements.txt                # Python dependencies
└── requirements-colab.txt          # Google Colab compatible dependencies
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
git clone https://github.com/ssrajadh/Diffusion-Face-Attack.git
cd Diffusion-Face-Attack

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

#### Option A: Basic Generation (Single Parameter Set)

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

#### Option B: Parameter Grid Exploration (V2 - For Final Report)

Generate forgeries with systematic parameter variations for comprehensive evaluation:

```bash
python scripts/generate_forged_img2img_v2.py \
  --model stabilityai/stable-diffusion-2-1 \
  --in-dir data/organized \
  --meta data/list_attr_celeba.txt \
  --identity-meta data/list_identity_celeba.txt \
  --out-dir data/forgeries_v2 \
  --template "a high-quality portrait photo of a person, {Male}, {Young}, {Black_Hair}, {Smiling}" \
  --denoise-grid 0.2 0.4 0.6 0.8 \
  --steps-grid 20 40 60 \
  --forgery-counts 1 3 5 9 \
  --cfg-scale 9.0 \
  --limit 50 \
  --create-visualizations \
  --viz-samples 10
```

**V2 Features:**
- **Parameter Grid**: Systematically explores combinations of denoising strength, inference steps, and forgery counts
- **Quality Metrics**: Automatically calculates sharpness, brightness, and contrast for each forgery
- **Metadata Tracking**: Comprehensive JSON metadata with all parameters and quality scores
- **Visualizations**: Generates side-by-side real vs. forged comparison grids
- **Scalability**: Processes N identities with d forgeries per parameter combination

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

**V2-Specific Parameters:**
- `--denoise-grid`: List of denoising strengths to explore (e.g., 0.2 0.4 0.6 0.8)
- `--steps-grid`: List of inference step counts (e.g., 20 40 60)
- `--forgery-counts`: List of forgery quantities per parameter set (e.g., 1 3 5 9)
- `--create-visualizations`: Generate side-by-side comparison visualizations
- `--viz-samples`: Number of identities to create visualizations for (default: 5)

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

**V2 Output Format:**

`metadata_v2.json` - Comprehensive metadata with parameter grid:
```json
{
  "experiment_config": {
    "model": "stabilityai/stable-diffusion-2-1",
    "denoising_grid": [0.2, 0.4, 0.6, 0.8],
    "steps_grid": [20, 40, 60],
    "forgery_counts": [1, 3, 5, 9]
  },
  "summary": {
    "total_identities_processed": 50,
    "total_parameter_combinations": 12,
    "total_forgeries_generated": 1050
  },
  "results": [...]
}
```

`visualizations/` - Side-by-side comparison images

### 5. Analyze Forgery Quality (V2)

Analyze quality metrics and generate visualizations:

```bash
python scripts/analyze_forgery_quality_v2.py \
  --metadata data/forgeries_v2/metadata_v2.json \
  --output-dir results/quality_analysis \
  --use-clip
```

**Output:**
- `forgery_quality_analysis.csv` - Detailed metrics for each forgery
- `quality_summary.csv` - Aggregated statistics by parameter set
- `quality_visualizations/` - Plots showing:
  - Quality score vs. denoising strength
  - CLIP similarity vs. parameters
  - Quality metric heatmaps
  - Distribution plots for sharpness, brightness, contrast

**Quality Metrics:**
- **CLIP Similarity**: Image-image similarity between real and forged (0-1, higher = more similar)
- **Quality Score**: Combined metric based on sharpness and contrast (0-100, higher = better)
- **Sharpness**: Laplacian variance (edge detection)
- **Brightness**: Mean pixel intensity
- **Contrast**: Standard deviation of pixel intensities

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

### 5. Prepare Test Sets for Attack Evaluation

Generate shuffled test sets with mapping-based forgery selection:

**For V1 Forgeries:**
```bash
python scripts/prepare_shuffled_sets.py \
  --data-dir data/organized \
  --forgery-dir data/forgeries \
  --forgery-mapping data/forgeries/mapping.json \
  --output data/shuffled_sets \
  --num-sets 50 \
  --seed 42
```

**For V2 Forgeries:**
```bash
python scripts/prepare_shuffled_sets.py \
  --data-dir data/organized \
  --forgery-dir data/forgeries_v2 \
  --forgery-mapping data/forgeries_v2/metadata_v2.json \
  --output data/shuffled_sets_v2 \
  --num-sets 50 \
  --seed 42
```

**Parameters:**
- `--data-dir`: Directory with organized identity folders
- `--forgery-dir`: Directory with generated forged images
- `--forgery-mapping`: Path to mapping.json file
- `--output`: Output directory for test sets
- `--num-sets`: Number of test sets to generate
- `--seed`: Random seed for reproducibility

**Test Set Structure:**
Each test set contains:
- 1 query image (target identity)
- 1 target real image + 3 forged (to find)
- 9 other identities (each with 1 real + 3 forged)
- Total: 40 candidate images (10 identities × 4 images)

### 6. Evaluate Re-identification Attack

#### Basic Attack Evaluation (V1)

Test face recognition robustness using embedding-based attacks:

```bash
python scripts/evaluate_attack.py \
  --test-sets data/shuffled_sets/test_sets_metadata.json \
  --model facenet \
  --output results/eval_facenet.json

python scripts/evaluate_attack.py \
  --test-sets data/shuffled_sets/test_sets_metadata.json \
  --model arcface \
  --output results/eval_arcface.json
```

**Parameters:**
- `--test-sets`: Path to test set metadata
- `--model`: Model to use (`facenet` or `arcface`)
- `--output`: Output file for results
- `--max-sets`: (Optional) Limit number of test sets

**Models:**
- `facenet`: InceptionResnetV1 trained on VGGFace2 (128-dim embeddings)
- `arcface`: InsightFace ArcFace model (512-dim embeddings, requires GPU)

**Example Results:**
```
ArcFace Top-1 Accuracy: 98% (49/50)
FaceNet Top-1 Accuracy: 96% (48/50)
- Query: Target's profile photo
- Found: Target's real image among candidates
- Success: 96-98% re-identification rate despite forgery protection
```

#### Open-Set Evaluation with Defense Mechanisms (V2)

Evaluate attacks with defensive preprocessing:

```bash
python scripts/evaluate_attack_openset.py \
  --forgery-dir data/forgeries_v2 \
  --metadata data/forgeries_v2/metadata_v2.json \
  --data-dir data/organized \
  --model arcface \
  --defense none \
  --num-distractors 20 \
  --num-tests 50 \
  --output results/experiments_v2/openset_arcface_none_n20_t50_s42.json
```

**Defense Options:**
- `none`: No defense (baseline)
- `noise_X`: Gaussian noise with std=X (e.g., `noise_0.01`, `noise_0.05`, `noise_0.1`, `noise_0.2`)
- `clip_X`: CLIP-based preprocessing with threshold X (e.g., `clip_0.8`, `clip_1.0`)
- `dp_gaussian_eX`: Differential privacy Gaussian noise with epsilon=X (e.g., `dp_gaussian_e0.5`, `dp_gaussian_e1.0`)
- `dp_laplace_eX`: Differential privacy Laplace noise with epsilon=X (e.g., `dp_laplace_e0.5`, `dp_laplace_e1.0`)

**Run Full Defense Experiments:**
```bash
python scripts/run_experiments.py \
  --forgery-dir data/forgeries_v2 \
  --metadata data/forgeries_v2/metadata_v2.json \
  --data-dir data/organized \
  --output-dir results/experiments_v2 \
  --num-distractors 20 50 \
  --num-tests 50 \
  --models facenet arcface
```

This runs all defense mechanisms across both models and generates:
- Individual result JSON files
- `experiment_summary.json` - Aggregated results
- `results_table.csv` - Comparison table
- `ANALYSIS_REPORT.md` - Detailed analysis

### 7. View Results

```bash
# View detailed results
cat results/eval_results.json

# Quick summary
python -c "
import json
with open('results/eval_results.json') as f:
    data = json.load(f)
    m = data['metrics']
    print(f\"Top-1 Accuracy: {m['top1_accuracy']:.2%}\")
    print(f\"Total Tests: {m['total_tests']}\")
    print(f\"Correct: {m['top1_correct_count']}\")
"
```

**Results Explanation:**

The output JSON contains comprehensive metrics:

```json
{
  "metrics": {
    "total_tests": 50,              // Number of test sets evaluated
    "successful_tests": 50,          // Tests completed without errors
    "top1_accuracy": 0.92,           // % of correct Top-1 predictions
    "top1_correct_count": 46,        // Number of correct predictions
    "precision": 0.92,               // True Positives / (True Positives + False Positives)
    "recall": 0.92,                  // True Positives / (True Positives + False Negatives)
    "f1_score": 0.92,                // Harmonic mean of precision and recall
    "avg_similarity": 0.65,          // Average cosine similarity across all predictions
    "std_similarity": 0.16,          // Standard deviation of similarities
    "min_similarity": -0.01,         // Lowest similarity score
    "max_similarity": 0.91,          // Highest similarity score
    "avg_similarity_correct": 0.70,  // Average similarity for CORRECT predictions
    "std_similarity_correct": 0.15,  // Std dev for correct predictions
    "avg_similarity_incorrect": 0.38,// Average similarity for INCORRECT predictions
    "std_similarity_incorrect": 0.02 // Std dev for incorrect predictions
  }
}
```

**Key Metrics Interpretation:**

1. **Top-1 Accuracy** (e.g., 0.92 = 92%)
   - Measures: How often the attacker correctly identifies the target's real image
   - **Higher is WORSE for privacy** ⚠️
   - 92% = Attack succeeds 92% of the time
   - <10% = Strong privacy protection ✅
   - >90% = Privacy protection failed ❌

2. **Precision & Recall** (e.g., 0.92)
   - Measures: Classification quality of the attack
   - **Higher is WORSE for privacy** ⚠️
   - Values close to 1.0 = Very reliable attack
   - Values close to 0.1 = Unreliable attack (good for privacy)

3. **F1 Score** (e.g., 0.92)
   - Measures: Overall attack effectiveness (harmonic mean of precision/recall)
   - **Higher is WORSE for privacy** ⚠️
   - >0.9 = Highly effective attack ❌
   - <0.3 = Ineffective attack ✅

4. **Similarity Scores**
   - `avg_similarity_correct` (e.g., 0.70): Average similarity for successful re-identifications
   - `avg_similarity_incorrect` (e.g., 0.38): Average similarity for failed attempts
   - **Large gap (>0.3) is WORSE for privacy** ⚠️
   - Large gap = Model confidently distinguishes real from forged
   - Small gap (<0.1) = Model struggles to identify real images

5. **Overall Assessment**
   - **92% accuracy with 0.32 similarity gap**:
     - ❌ Privacy protection FAILED
     - ❌ Forgery-based anonymization is ineffective
     - ❌ Face recognition easily defeats the protection
     - 🚨 Risk: Attackers can re-identify individuals 92% of the time

   - **Ideal privacy protection** (what we want but don't have):
     - ✅ Top-1 Accuracy: <10%
     - ✅ Similarity gap: <0.1
     - ✅ F1 Score: <0.3
     - ✅ Meaning: Attacker cannot reliably distinguish real from forged images

## Research Applications

This tool supports research in:

1. **Privacy Attack Evaluation**: Test face recognition robustness against adversarial inputs
2. **Multimodal LLM Security**: Evaluate vision-language models on manipulated faces
3. **Synthetic Data Generation**: Create privacy-preserving datasets
4. **Attribute Manipulation**: Study controllable face generation
5. **Identity Preservation**: Analyze trade-offs between realism and identity retention
