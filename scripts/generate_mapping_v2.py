#!/usr/bin/env python3
# Author: Youngju Choi
"""
Generate mapping.json from metadata_v2.json for forgeries_v2

This script:
1. Reads metadata_v2.json
2. For each real image, randomly selects 9 forged images
3. Creates mapping.json in the same format as forgeries/mapping.json
"""

import json
import random
from pathlib import Path

def generate_mapping_v2(metadata_path, output_path, forgery_dir, num_forgeries=9, seed=42):
    """
    Generate mapping.json from metadata_v2.json
    
    Args:
        metadata_path: Path to metadata_v2.json
        output_path: Path to save mapping.json
        forgery_dir: Directory containing actual forged images
        num_forgeries: Number of forged images to select per real image (use 999 for all)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    forgery_dir = Path(forgery_dir)
    
    print(f"Reading metadata from: {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    mapping = []
    
    results = metadata.get('results', [])
    print(f"Processing {len(results)} real images...")
    print(f"Checking for actual files in: {forgery_dir}")
    
    for result in results:
        real_image = result['real_image']
        
        # Collect all forged images across all parameter sets
        all_forgeries = []
        for param_set in result.get('parameter_sets', []):
            for forgery in param_set.get('forgeries', []):
                filename = forgery['filename']
                # Check if file actually exists
                if (forgery_dir / filename).exists():
                    all_forgeries.append({
                        'filename': filename,
                        'seed': forgery['seed'],
                        'denoise_strength': forgery['denoise_strength'],
                        'inference_steps': forgery['inference_steps'],
                        'cfg_scale': forgery['cfg_scale']
                    })
        
        if len(all_forgeries) == 0:
            print(f"  {real_image}: ⚠️  No forged images found!")
            continue
        
        # Randomly select num_forgeries from all available
        if num_forgeries < 999 and len(all_forgeries) > num_forgeries:
            selected_forgeries = random.sample(all_forgeries, num_forgeries)
        else:
            selected_forgeries = all_forgeries
        
        # Create mapping entry
        mapping_entry = {
            'real': real_image,
            'forged': [f['filename'] for f in selected_forgeries],
            'prompt': result.get('prompt', ''),
            'seeds': [f['seed'] for f in selected_forgeries],
            'model': metadata['experiment_config']['model'],
            'cfg_scale': metadata['experiment_config']['cfg_scale'],
            'denoising_strengths': [f['denoise_strength'] for f in selected_forgeries],
            'inference_steps': [f['inference_steps'] for f in selected_forgeries]
        }
        
        mapping.append(mapping_entry)
        
        print(f"  {real_image}: {len(all_forgeries)} exist → {len(selected_forgeries)} selected")
    
    # Save mapping
    print(f"\nSaving mapping to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"✅ Done! Created mapping for {len(mapping)} real images")
    print(f"   Total forged images in mapping: {sum(len(m['forged']) for m in mapping)}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate mapping.json from metadata_v2.json')
    parser.add_argument('--metadata', type=str, 
                       default='data/forgeries_v2/metadata_v2.json',
                       help='Path to metadata_v2.json')
    parser.add_argument('--forgery-dir', type=str,
                       default='data/forgeries_v2',
                       help='Directory containing actual forged images')
    parser.add_argument('--output', type=str,
                       default='data/forgeries_v2/mapping.json',
                       help='Path to save mapping.json')
    parser.add_argument('--num-forgeries', type=int, default=9,
                       help='Number of forged images to select per real image (use 999 for all)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    generate_mapping_v2(args.metadata, args.output, args.forgery_dir, 
                       args.num_forgeries, args.seed)

