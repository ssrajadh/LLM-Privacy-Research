# scripts/analyze_forgery_quality_v2.py
"""
Analyze Forgery Quality V2: Comprehensive Quality Assessment
Calculates CLIP scores, NIQE scores, and other quality metrics for forgeries
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_clip_similarity(real_img_path, forged_img_path, model, processor, device):
    """
    Calculate CLIP image-image similarity between real and forged image
    """
    from PIL import Image
    
    real_img = Image.open(real_img_path).convert("RGB")
    forged_img = Image.open(forged_img_path).convert("RGB")
    
    # Process images
    inputs = processor(images=[real_img, forged_img], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
        
    # Normalize and compute cosine similarity
    real_emb = outputs[0] / outputs[0].norm()
    forged_emb = outputs[1] / outputs[1].norm()
    
    similarity = torch.dot(real_emb, forged_emb).item()
    return similarity

def calculate_niqe_score(img_path):
    """
    Calculate NIQE (Natural Image Quality Evaluator) score
    Lower is better (less distortion)
    """
    try:
        import cv2
        from scipy import stats
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
            
        # Simple quality metric based on sharpness and contrast
        # This is a simplified version - full NIQE requires training
        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
        contrast = img.std()
        
        # Normalized quality score (0-100, higher is better)
        quality_score = min(100, (laplacian_var / 10) + (contrast / 2))
        
        return quality_score
    except Exception as e:
        print(f"Error calculating NIQE for {img_path}: {e}")
        return None

def analyze_forgery_quality(metadata_path, output_dir, use_clip=True):
    """
    Analyze quality metrics for all forgeries in metadata
    """
    print("Loading metadata...")
    with open(metadata_path, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    # Initialize CLIP if requested
    clip_model = None
    clip_processor = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if use_clip:
        print(f"Loading CLIP model on {device}...")
        try:
            from transformers import CLIPModel, CLIPProcessor
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print("CLIP model loaded successfully")
        except Exception as e:
            print(f"Failed to load CLIP: {e}")
            use_clip = False
    
    # Analyze each identity
    analysis_results = []
    
    print("\nAnalyzing forgery quality...")
    for identity_result in tqdm(results, desc="Processing identities"):
        real_path = identity_result['real_path']
        
        for param_set in identity_result['parameter_sets']:
            denoise = param_set['denoising_strength']
            steps = param_set['inference_steps']
            
            for forgery in param_set['forgeries']:
                forged_path = os.path.join(os.path.dirname(metadata_path), forgery['filename'])
                
                if not os.path.exists(forged_path):
                    continue
                
                # Calculate CLIP similarity if available
                clip_score = None
                if use_clip and clip_model is not None:
                    try:
                        clip_score = calculate_clip_similarity(
                            real_path, forged_path, clip_model, clip_processor, device
                        )
                    except Exception as e:
                        print(f"Error calculating CLIP for {forgery['filename']}: {e}")
                
                # Calculate NIQE-like quality score
                niqe_score = calculate_niqe_score(forged_path)
                
                analysis_results.append({
                    'identity_id': identity_result['identity_id'],
                    'real_image': identity_result['real_image'],
                    'forged_image': forgery['filename'],
                    'denoising_strength': denoise,
                    'inference_steps': steps,
                    'seed': forgery['seed'],
                    'clip_similarity': clip_score,
                    'quality_score': niqe_score,
                    'sharpness': forgery['quality_metrics']['sharpness'],
                    'brightness': forgery['quality_metrics']['brightness'],
                    'contrast': forgery['quality_metrics']['contrast']
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(analysis_results)
    
    # Save detailed results
    csv_path = os.path.join(output_dir, "forgery_quality_analysis.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved detailed analysis to: {csv_path}")
    
    # Generate summary statistics
    summary = generate_summary_statistics(df, output_dir)
    
    # Generate visualizations
    generate_visualizations(df, output_dir)
    
    return df, summary

def generate_summary_statistics(df, output_dir):
    """Generate summary statistics by parameter sets"""
    
    summary = []
    
    for denoise in df['denoising_strength'].unique():
        for steps in df['inference_steps'].unique():
            subset = df[(df['denoising_strength'] == denoise) & 
                       (df['inference_steps'] == steps)]
            
            stats = {
                'denoising_strength': denoise,
                'inference_steps': steps,
                'num_forgeries': len(subset),
                'mean_quality': subset['quality_score'].mean() if 'quality_score' in subset else None,
                'std_quality': subset['quality_score'].std() if 'quality_score' in subset else None,
                'mean_sharpness': subset['sharpness'].mean(),
                'mean_brightness': subset['brightness'].mean(),
                'mean_contrast': subset['contrast'].mean()
            }
            
            if 'clip_similarity' in subset.columns and subset['clip_similarity'].notna().any():
                stats['mean_clip_similarity'] = subset['clip_similarity'].mean()
                stats['std_clip_similarity'] = subset['clip_similarity'].std()
            
            summary.append(stats)
    
    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(output_dir, "quality_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary statistics to: {summary_path}")
    
    return summary_df

def generate_visualizations(df, output_dir):
    """Generate quality metric visualizations"""
    
    viz_dir = os.path.join(output_dir, "quality_visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Quality score vs denoising strength
    if 'quality_score' in df.columns and df['quality_score'].notna().any():
        plt.figure(figsize=(10, 6))
        for steps in df['inference_steps'].unique():
            subset = df[df['inference_steps'] == steps]
            grouped = subset.groupby('denoising_strength')['quality_score'].agg(['mean', 'std'])
            plt.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
                        label=f'{steps} steps', marker='o', capsize=5)
        
        plt.xlabel('Denoising Strength', fontsize=12)
        plt.ylabel('Quality Score (Higher = Better)', fontsize=12)
        plt.title('Image Quality vs Denoising Strength', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'quality_vs_denoise.png'), dpi=300)
        plt.close()
    
    # 2. CLIP similarity vs parameters
    if 'clip_similarity' in df.columns and df['clip_similarity'].notna().any():
        plt.figure(figsize=(10, 6))
        for steps in df['inference_steps'].unique():
            subset = df[df['inference_steps'] == steps]
            grouped = subset.groupby('denoising_strength')['clip_similarity'].agg(['mean', 'std'])
            plt.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                        label=f'{steps} steps', marker='s', capsize=5)
        
        plt.xlabel('Denoising Strength', fontsize=12)
        plt.ylabel('CLIP Similarity (Higher = More Similar)', fontsize=12)
        plt.title('Identity Preservation (CLIP) vs Denoising Strength', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'clip_similarity_vs_denoise.png'), dpi=300)
        plt.close()
    
    # 3. Heatmap of quality metrics
    pivot_quality = df.pivot_table(
        values='quality_score', 
        index='inference_steps', 
        columns='denoising_strength', 
        aggfunc='mean'
    )
    
    if not pivot_quality.empty:
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_quality, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': 'Quality Score'})
        plt.title('Image Quality Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Denoising Strength', fontsize=12)
        plt.ylabel('Inference Steps', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'quality_heatmap.png'), dpi=300)
        plt.close()
    
    # 4. Distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Sharpness distribution
    for denoise in sorted(df['denoising_strength'].unique()):
        subset = df[df['denoising_strength'] == denoise]
        axes[0, 0].hist(subset['sharpness'], alpha=0.5, label=f'd={denoise}', bins=30)
    axes[0, 0].set_xlabel('Sharpness')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Sharpness Distribution by Denoising Strength')
    axes[0, 0].legend()
    
    # Brightness distribution
    for denoise in sorted(df['denoising_strength'].unique()):
        subset = df[df['denoising_strength'] == denoise]
        axes[0, 1].hist(subset['brightness'], alpha=0.5, label=f'd={denoise}', bins=30)
    axes[0, 1].set_xlabel('Brightness')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Brightness Distribution by Denoising Strength')
    axes[0, 1].legend()
    
    # Contrast distribution
    for denoise in sorted(df['denoising_strength'].unique()):
        subset = df[df['denoising_strength'] == denoise]
        axes[1, 0].hist(subset['contrast'], alpha=0.5, label=f'd={denoise}', bins=30)
    axes[1, 0].set_xlabel('Contrast')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Contrast Distribution by Denoising Strength')
    axes[1, 0].legend()
    
    # CLIP similarity distribution (if available)
    if 'clip_similarity' in df.columns and df['clip_similarity'].notna().any():
        for denoise in sorted(df['denoising_strength'].unique()):
            subset = df[df['denoising_strength'] == denoise]
            if subset['clip_similarity'].notna().any():
                axes[1, 1].hist(subset['clip_similarity'].dropna(), alpha=0.5, label=f'd={denoise}', bins=30)
        axes[1, 1].set_xlabel('CLIP Similarity')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('CLIP Similarity Distribution by Denoising Strength')
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'metric_distributions.png'), dpi=300)
    plt.close()
    
    print(f"\nVisualization saved to: {viz_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze forgery quality metrics")
    parser.add_argument("--metadata", required=True, help="Path to metadata_v2.json")
    parser.add_argument("--output-dir", required=True, help="Output directory for analysis results")
    parser.add_argument("--use-clip", action='store_true', help="Calculate CLIP similarity scores")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("FORGERY QUALITY ANALYSIS V2")
    print(f"{'='*80}\n")
    
    df, summary = analyze_forgery_quality(args.metadata, args.output_dir, args.use_clip)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Total forgeries analyzed: {len(df)}")
    print(f"\nResults saved to: {args.output_dir}/")
    print(f"{'='*80}\n")
