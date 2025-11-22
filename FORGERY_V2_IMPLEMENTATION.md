# Forgery V2 Implementation for Final Report

## Overview

This document describes the Forgery V2 pipeline implementation for systematic parameter exploration in the final report.

## Key Features

### 1. Parameter Grid Exploration
- **Denoising Strength**: [0.2, 0.4, 0.6, 0.8] - Trade-off between identity preservation and variation
- **Inference Steps**: [20, 40, 60] - Quality vs. speed trade-off
- **Forgery Counts**: [1, 3, 5, 9] - Number of forgeries per parameter set
- **Total Combinations**: 12 parameter sets per identity (3 steps × 4 denoise values)

### 2. Quality Metrics

**Automatic Metrics** (calculated during generation):
- **Sharpness**: Laplacian variance - measures edge definition
- **Brightness**: Mean pixel intensity - exposure/lighting
- **Contrast**: Pixel intensity standard deviation - dynamic range

**Post-Processing Metrics** (analyzed separately):
- **CLIP Similarity**: Measures semantic similarity between real and forged images
- **Quality Score**: Combined metric (sharpness + contrast normalized)

### 3. Metadata Tracking

Complete experiment metadata saved in `metadata_v2.json`:
```json
{
  "experiment_config": {...},
  "summary": {...},
  "results": [
    {
      "real_image": "000001.jpg",
      "identity_id": "2880",
      "prompt": "...",
      "parameter_sets": [
        {
          "denoising_strength": 0.4,
          "inference_steps": 40,
          "num_forgeries": 3,
          "forgeries": [
            {
              "filename": "000001_forg_d40_s40_n3_s123456.png",
              "seed": 123456,
              "quality_metrics": {...}
            }
          ]
        }
      ]
    }
  ]
}
```

### 4. Visualization Pipeline

**Side-by-Side Comparisons**:
- Real image vs. 8 forgeries in 3×3 grid
- Generated for first N identities (configurable via `--viz-samples`)
- Saved to `visualizations/` directory

**Quality Analysis Plots**:
- Quality score vs. denoising strength (line plots with error bars)
- CLIP similarity vs. parameters (line plots with error bars)
- Heatmaps showing quality across parameter space
- Distribution plots for each metric

## Usage Examples

### Step 1: Generate Forgeries with Parameter Grid

```bash
# Full parameter exploration for 50 identities
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

**Expected Output:**
- Total forgeries: 50 identities × 12 param sets × avg(1+3+5+9)/4 = ~10,800 images
- Runtime: ~30-40 seconds per image = ~90-120 GPU-hours for full run
- Storage: ~16-22 GB

### Step 2: Analyze Quality Metrics

```bash
# With CLIP similarity calculation
python scripts/analyze_forgery_quality_v2.py \
  --metadata data/forgeries_v2/metadata_v2.json \
  --output-dir results/quality_analysis_v2 \
  --use-clip
```

**Output Files:**
- `forgery_quality_analysis.csv` - Per-image metrics
- `quality_summary.csv` - Aggregated by parameter set
- `quality_visualizations/quality_vs_denoise.png`
- `quality_visualizations/clip_similarity_vs_denoise.png`
- `quality_visualizations/quality_heatmap.png`
- `quality_visualizations/metric_distributions.png`

### Step 3: Evaluate Attack Success Rates

Use existing evaluation pipeline with V2 forgeries:

```bash
# Prepare test sets from V2 forgeries
python scripts/prepare_shuffled_sets_v2.py \
  --metadata data/forgeries_v2/metadata_v2.json \
  --output data/shuffled_sets_v2

# Evaluate with FaceNet and ArcFace
python scripts/evaluate_attack.py \
  --test-sets data/shuffled_sets_v2/test_sets_metadata.json \
  --model facenet \
  --output results/eval_facenet_v2.json

python scripts/evaluate_attack.py \
  --test-sets data/shuffled_sets_v2/test_sets_metadata.json \
  --model arcface \
  --output results/eval_arcface_v2.json
```

## Parameter Space Insights

### Denoising Strength Effects

| Strength | Identity Preservation | Visual Variation | Attack Effectiveness |
|----------|----------------------|------------------|---------------------|
| 0.2      | Very High            | Minimal          | Expected: 98-99%    |
| 0.4      | High                 | Moderate         | Expected: 96-98%    |
| 0.6      | Moderate             | Significant      | Expected: 85-95%    |
| 0.8      | Low                  | High             | Expected: 60-80%    |

### Inference Steps Effects

| Steps | Generation Time | Quality          | Artifacts        |
|-------|----------------|------------------|------------------|
| 20    | ~15s/image     | Good             | Occasional       |
| 40    | ~25s/image     | Very Good        | Rare             |
| 60    | ~35s/image     | Excellent        | Minimal          |

### Forgery Count Effects

| Count | Use Case                          | Statistical Power |
|-------|-----------------------------------|-------------------|
| 1     | Minimal attack scenario           | Low               |
| 3     | Standard attack (baseline)        | Medium            |
| 5     | Enhanced attack                   | High              |
| 9     | Maximum variation exploration     | Very High         |

## Expected Results for Final Report

### Hypothesis 1: Quality-Attack Trade-off
**Prediction**: Higher denoising strength → Lower quality → Lower attack success
**Analysis**: Plot attack success rate vs. quality score, stratified by denoising strength

### Hypothesis 2: Quantity Amplification
**Prediction**: More forgeries per identity → Higher aggregate attack success
**Analysis**: Compare attack success with 1, 3, 5, 9 forgeries per identity

### Hypothesis 3: Inference Step Threshold
**Prediction**: Quality plateaus after ~40 steps, diminishing returns above 60
**Analysis**: Quality metrics vs. inference steps, cost-benefit analysis

### Hypothesis 4: Identity-Dependent Vulnerability
**Prediction**: Some identities more vulnerable than others
**Analysis**: Stratify by demographic attributes (Male/Female, Young/Old)

## Data Analysis Pipeline

### 1. Load and Aggregate Results

```python
import json
import pandas as pd

# Load V2 metadata
with open('data/forgeries_v2/metadata_v2.json') as f:
    data = json.load(f)

# Load quality analysis
quality_df = pd.read_csv('results/quality_analysis_v2/forgery_quality_analysis.csv')

# Load attack evaluation
with open('results/eval_arcface_v2.json') as f:
    attack_results = json.load(f)
```

### 2. Merge Datasets

```python
# Merge quality metrics with attack results
merged = pd.merge(
    quality_df,
    attack_results_df,
    left_on=['identity_id', 'forged_image'],
    right_on=['identity_id', 'prediction_image']
)
```

### 3. Generate Visualizations

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Attack success rate vs. denoising strength
plt.figure(figsize=(10, 6))
for steps in [20, 40, 60]:
    subset = merged[merged['inference_steps'] == steps]
    grouped = subset.groupby('denoising_strength')['attack_success'].mean()
    plt.plot(grouped.index, grouped.values, marker='o', label=f'{steps} steps')

plt.xlabel('Denoising Strength')
plt.ylabel('Attack Success Rate')
plt.title('Attack Effectiveness vs. Forgery Quality')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('figures/attack_vs_denoise.png', dpi=300)
```

## Timeline (Week 1-2 of Final Report)

### Week 1
- **Day 1-2**: Generate forgeries for 50 identities with full parameter grid (~50 GPU-hours)
- **Day 3**: Run quality analysis with CLIP similarity
- **Day 4-5**: Prepare test sets and run evaluation pipeline
- **Day 6-7**: Initial analysis and visualization generation

### Week 2
- **Day 1-2**: Expand to 100-150 identities if time permits
- **Day 3-4**: Comprehensive statistical analysis
- **Day 5-6**: Generate all figures for paper
- **Day 7**: Draft results section with insights

## Expected Deliverables

1. **Dataset**: 10,000-15,000 forgeries across parameter space
2. **Metadata**: Complete JSON with all parameters and metrics
3. **Visualizations**: 
   - 10+ side-by-side comparison grids
   - 6-8 quality metric plots
   - 8-10 attack evaluation plots
4. **Analysis**: CSV files with per-image and aggregate metrics
5. **Code**: Reproducible pipeline with documentation

## Storage and Computational Requirements

- **Storage**: ~20-25 GB for images + ~500 MB for metadata
- **GPU Time**: 80-120 hours on V100 (or 40-60 hours on A100)
- **RAM**: 16-32 GB recommended for quality analysis
- **Python Packages**: transformers, diffusers, torch, PIL, cv2, pandas, matplotlib, seaborn

## Troubleshooting

### Out of Memory
- Reduce batch size or process fewer identities in parallel
- Use gradient checkpointing: `pipe.enable_attention_slicing()`
- Lower resolution if needed (resize inputs to 384×384)

### Slow CLIP Analysis
- Process in batches of 100 images
- Use GPU for CLIP inference
- Skip CLIP for preliminary analysis (use quality_score only)

### Missing Forgeries
- Check that identity_id exists in organized directory
- Verify metadata file paths are correct
- Ensure sufficient disk space

## References

- Stable Diffusion 2.1: https://huggingface.co/stabilityai/stable-diffusion-2-1
- CLIP: https://huggingface.co/openai/clip-vit-base-patch32
- CelebA Dataset: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
