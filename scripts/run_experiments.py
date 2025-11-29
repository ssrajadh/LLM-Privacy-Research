#!/usr/bin/env python3
"""
Automated Experiment Runner for Open-Set Identity Re-identification

This script automates running experiments across multiple configurations:
- Models: FaceNet, ArcFace
- Identities: N=20, N=50
- Defense mechanisms: 15+ configurations

Usage:
    # Run all experiments
    python scripts/run_experiments.py \
        --data-dir data/organized \
        --output-dir results/openset_experiments
    
    # Run specific subset
    python scripts/run_experiments.py \
        --data-dir data/organized \
        --output-dir results/test \
        --models facenet \
        --num-identities 20 \
        --num-tests 10 \
        --defenses none clip_1.0 noise_0.1
"""

import os
import json
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import itertools

from defense_mechanisms import create_defense_configs


def run_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single experiment configuration.
    
    Args:
        config: Experiment configuration dictionary
        
    Returns:
        Experiment results including timing and status
    """
    # Build command
    cmd = [
        'python', 'scripts/evaluate_attack_openset.py',
        '--data-dir', config['data_dir'],
        '--model', config['model'],
        '--defense', config['defense'],
        '--num-identities', str(config['num_identities']),
        '--num-tests', str(config['num_tests']),
        '--device', config['device'],
        '--seed', str(config['seed']),
        '--output', config['output_file']
    ]
    cmd.extend([
        '--forgery-dir', config['forgery_dir'],
        '--forgery-mapping', config['forgery_mapping'],
        '--num-forged-per-real', str(config['num_forged_per_real'])
    ])
    
    if config.get('save_detailed', False):
        cmd.append('--save-detailed')
    
    print("\n" + "="*80)
    print(f"Running experiment:")
    print(f"  Model: {config['model']}")
    print(f"  Defense: {config['defense']}")
    print(f"  Identities: {config['num_identities']}")
    print(f"  Tests: {config['num_tests']}")
    print(f"  Output: {config['output_file']}")
    print("="*80)
    
    # Check if output already exists
    if Path(config['output_file']).exists():
        if config.get('skip_existing', True):
            print(f"⏭️  Output file already exists, skipping...")
            
            # Load existing results
            with open(config['output_file'], 'r') as f:
                existing_results = json.load(f)
            
            return {
                'config': config,
                'status': 'skipped',
                'metrics': existing_results.get('metrics', {}),
                'runtime': 0,
                'timestamp': existing_results.get('timestamp', 'unknown')
            }
    
    # Run experiment
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        runtime = time.time() - start_time
        
        # Load results
        with open(config['output_file'], 'r') as f:
            results = json.load(f)
        
        print(f"✅ Success! Runtime: {runtime:.1f}s")
        print(f"   Top-1 Accuracy: {results['metrics']['top1_accuracy']:.4f}")
        
        return {
            'config': config,
            'status': 'success',
            'metrics': results['metrics'],
            'runtime': runtime,
            'timestamp': datetime.now().isoformat()
        }
        
    except subprocess.CalledProcessError as e:
        runtime = time.time() - start_time
        print(f"❌ Failed! Runtime: {runtime:.1f}s")
        print(f"   Error: {e.stderr[:200]}")
        
        return {
            'config': config,
            'status': 'failed',
            'error': str(e),
            'stderr': e.stderr[:500],
            'runtime': runtime,
            'timestamp': datetime.now().isoformat()
        }
    
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout after {3600}s")
        
        return {
            'config': config,
            'status': 'timeout',
            'runtime': 3600,
            'timestamp': datetime.now().isoformat()
        }


def generate_experiment_configs(args) -> List[Dict[str, Any]]:
    """
    Generate all experiment configurations.
    
    Args:
        args: Command line arguments
        
    Returns:
        List of experiment configuration dictionaries
    """
    # Get available defenses
    all_defense_configs = create_defense_configs()
    
    # Filter defenses if specified
    if args.defenses:
        defense_names = args.defenses
    else:
        # Default: comprehensive set
        defense_names = [
            'none',
            'clip_1.0',
            'noise_0.01', 'noise_0.05', 'noise_0.1', 'noise_0.2',
            'dp_laplace_e1.0', 'dp_laplace_e0.5',
            'dp_gaussian_e1.0', 'dp_gaussian_e0.5',
        ]
    
    # Validate defense names
    for name in defense_names:
        if name not in all_defense_configs:
            raise ValueError(f"Unknown defense: {name}")
    
    # Generate all combinations
    configs = []
    
    for model, defense, num_ids in itertools.product(
        args.models,
        defense_names,
        args.num_identities
    ):
        # Generate output filename
        output_filename = f"openset_{model}_{defense}_n{num_ids}_t{args.num_tests}_s{args.seed}.json"
        output_path = Path(args.output_dir) / output_filename
        
        config = {
            'data_dir': args.data_dir,
            'model': model,
            'defense': defense,
            'num_identities': num_ids,
            'num_tests': args.num_tests,
            'device': args.device,
            'forgery_dir': args.forgery_dir,
            'forgery_mapping': args.forgery_mapping,
            'num_forged_per_real': args.num_forged_per_real,
            'seed': args.seed,
            'output_file': str(output_path),
            'save_detailed': args.save_detailed,
            'skip_existing': not args.overwrite
        }
        
        configs.append(config)
    
    return configs


def save_summary(results: List[Dict], output_dir: str):
    """
    Save experiment summary with all results.
    
    Args:
        results: List of experiment results
        output_dir: Output directory
    """
    summary_path = Path(output_dir) / 'experiment_summary.json'
    
    # Calculate aggregate statistics
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    skipped = [r for r in results if r['status'] == 'skipped']
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_experiments': len(results),
        'successful': len(successful),
        'failed': len(failed),
        'skipped': len(skipped),
        'total_runtime': sum(r['runtime'] for r in results),
        'experiments': results
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Total experiments: {len(results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Skipped: {len(skipped)}")
    print(f"Total runtime: {summary['total_runtime']:.1f}s ({summary['total_runtime']/3600:.2f}h)")
    print(f"\nSummary saved to: {summary_path}")
    
    # Create results table
    create_results_table(results, output_dir)


def create_results_table(results: List[Dict], output_dir: str):
    """
    Create a CSV table with all experiment results.
    
    Args:
        results: List of experiment results
        output_dir: Output directory
    """
    import csv
    
    table_path = Path(output_dir) / 'results_table.csv'
    
    # Extract results for successful experiments
    rows = []
    for r in results:
        if r['status'] != 'success':
            continue
        
        config = r['config']
        metrics = r.get('metrics', {})
        
        row = {
            'model': config['model'],
            'defense': config['defense'],
            'num_identities': config['num_identities'],
            'num_tests': config['num_tests'],
            'top1_accuracy': metrics.get('top1_accuracy', 0),
            'mrr': metrics.get('mrr', 0),
            'avg_similarity': metrics.get('avg_similarity', 0),
            'avg_similarity_correct': metrics.get('avg_similarity_correct', 0),
            'avg_similarity_incorrect': metrics.get('avg_similarity_incorrect', 0),
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'f1_score': metrics.get('f1_score', 0),
            'runtime': r['runtime'],
        }
        rows.append(row)
    
    if not rows:
        print("No successful experiments to create table")
        return
    
    # Write CSV
    with open(table_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Results table saved to: {table_path}")
    
    # Print key results
    print("\n" + "="*80)
    print("KEY RESULTS")
    print("="*80)
    
    # Group by model and defense
    for model in ['facenet', 'arcface']:
        model_results = [r for r in rows if r['model'] == model]
        if not model_results:
            continue
        
        print(f"\n{model.upper()}:")
        print(f"{'Defense':<20} {'N':<5} {'Top-1 Acc':<12} {'MRR':<8}")
        print("-" * 50)
        
        for row in sorted(model_results, key=lambda x: (x['defense'], x['num_identities'])):
            print(f"{row['defense']:<20} {row['num_identities']:<5} "
                  f"{row['top1_accuracy']:.4f}       "
                  f"{row['mrr']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Automated experiment runner for open-set evaluation'
    )
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory with organized identity images')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for all experiment results')
    
    # Experiment parameters
    parser.add_argument('--models', nargs='+', default=['facenet', 'arcface'],
                       choices=['facenet', 'arcface'],
                       help='Models to evaluate')
    parser.add_argument('--num-identities', nargs='+', type=int, default=[20, 50],
                       help='Number of identities to test (can specify multiple)')
    parser.add_argument('--num-tests', type=int, default=50,
                       help='Number of test cases per configuration')
    parser.add_argument('--defenses', nargs='+', default=None,
                       help='Defense mechanisms to test (default: comprehensive set)')
    
    # Runtime parameters
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing results')
    parser.add_argument('--save-detailed', action='store_true',
                       help='Save detailed per-test results')
    parser.add_argument('--forgery-dir', type=str, default='data/forgeries',
                       help='Directory containing forged images')
    parser.add_argument('--forgery-mapping', type=str,
                       default='data/forgeries/mapping.json',
                       help='Path to forgery mapping file')
    parser.add_argument('--num-forged-per-real', type=int, default=9,
                       help='Number of forged images to sample per real image at runtime')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print experiment configurations without running')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate experiment configurations
    print("Generating experiment configurations...")
    configs = generate_experiment_configs(args)
    
    print(f"\nTotal experiments to run: {len(configs)}")
    print(f"  Models: {args.models}")
    print(f"  Identities: {args.num_identities}")
    print(f"  Tests per config: {args.num_tests}")
    
    if args.dry_run:
        print("\nDry run - configurations:")
        for i, config in enumerate(configs, 1):
            print(f"\n{i}. {config['model']} / {config['defense']} / N={config['num_identities']}")
            print(f"   Output: {config['output_file']}")
        return
    
    # Run experiments
    print("\nStarting experiments...")
    start_time = time.time()
    
    results = []
    for i, config in enumerate(configs, 1):
        print(f"\n{'='*80}")
        print(f"Experiment {i}/{len(configs)}")
        print(f"{'='*80}")
        
        result = run_experiment(config)
        results.append(result)
        
        # Save intermediate summary after each experiment
        save_summary(results, args.output_dir)
    
    total_time = time.time() - start_time
    
    # Final summary
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*80)
    print(f"Total time: {total_time:.1f}s ({total_time/3600:.2f}h)")
    
    save_summary(results, args.output_dir)


if __name__ == '__main__':
    main()

