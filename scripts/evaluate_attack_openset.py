#!/usr/bin/env python3
# Author: Youngju Choi
"""
Open-Set Identity Re-identification Attack Evaluation with Defenses

This script evaluates embedding-based attacks in an open-set scenario with:
- Multiple identities (N=20, 50, etc.)
- Both ArcFace and FaceNet backbones
- Multiple defense mechanisms (clipping, noise, DP)

Usage:
    # Single experiment
    python scripts/evaluate_attack_openset.py \
        --data-dir data/organized \
        --forgery-dir data/forgeries3 \
        --model facenet \
        --defense none \
        --num-identities 20 \
        --output results/openset_facenet_none_n20.json

    # Batch experiments (see run_experiments.py)
"""

import os
import json
import argparse
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import torch

import sys
sys.path.append(str(Path(__file__).parent))
from embedding_attack import FaceEmbeddingExtractor
from defense_mechanisms import EmbeddingDefense, DefenseConfig, create_defense_configs


class OpenSetEvaluator:
    """
    Evaluates identity re-identification attacks in an open-set scenario.
    
    Open-set means the attacker must identify the correct identity from a large
    pool of candidates, including multiple real and forged images per identity.
    """
    
    def __init__(self, 
                 extractor: FaceEmbeddingExtractor,
                 defense: Optional[EmbeddingDefense] = None,
                 forgery_dir: Optional[str] = None,
                 forgery_mapping_path: Optional[str] = None,
                 num_forged_per_real: int = 9):
        """
        Initialize open-set evaluator.
        
        Args:
            extractor: Face embedding extractor
            defense: Optional defense mechanism to apply to embeddings
            forgery_dir: Directory with forged images
            forgery_mapping_path: Path to forgery mapping JSON
            num_forged_per_real: Number of forged images to sample per real image at runtime
        """
        self.extractor = extractor
        self.defense = defense
        self.embedding_cache = {}
        self.forgery_dir = Path(forgery_dir) if forgery_dir else None
        self.forgery_mapping = self._load_forgery_mapping(forgery_mapping_path) \
            if forgery_mapping_path else {}
        self.use_forged = bool(self.forgery_mapping and self.forgery_dir and self.forgery_dir.exists())
        self.num_forged_per_real = num_forged_per_real
    
    def get_embedding(self, image_path: str, 
                     apply_defense: bool = True,
                     use_cache: bool = True) -> Optional[np.ndarray]:
        """
        Extract and optionally protect embedding.
        
        Args:
            image_path: Path to image
            apply_defense: Whether to apply defense mechanism
            use_cache: Whether to use cached embeddings
            
        Returns:
            Protected embedding or None if extraction failed
        """
        # Cache key includes defense state
        cache_key = (image_path, apply_defense)
        
        if use_cache and cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # Extract raw embedding
        emb = self.extractor.extract_embedding(image_path)
        
        if emb is None:
            return None
        
        # Apply defense if specified
        if apply_defense and self.defense is not None:
            emb = self.defense.apply(emb)
        
        if use_cache:
            self.embedding_cache[cache_key] = emb
        
        return emb
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def _load_forgery_mapping(self, mapping_path: str) -> Dict[str, List[str]]:
        """
        Load forgery mapping from JSON and return dict real -> forged list.
        """
        if not mapping_path or not os.path.exists(mapping_path):
            return {}

        with open(mapping_path, 'r') as f:
            data = json.load(f)

        mapping = {}
        for entry in data:
            real = entry.get('real')
            forged = entry.get('forged') or entry.get('forgeries') or []
            if real:
                mapping[real] = [str(fg) for fg in forged]

        return mapping

    def _collect_forged_images(self, base_images: List[str]) -> List[str]:
        """
        Given a list of real image paths, fetch corresponding forged paths if available.
        Randomly samples num_forged_per_real images from the available forgeries.
        """
        if not self.use_forged:
            return []

        forged_candidates = []
        for img_path in base_images:
            basename = Path(img_path).name
            if basename not in self.forgery_mapping:
                continue

            forged_list = self.forgery_mapping[basename]
            
            # Randomly sample num_forged_per_real from available forgeries
            if len(forged_list) > self.num_forged_per_real:
                forged_list = random.sample(forged_list, self.num_forged_per_real)
            
            for forged_name in forged_list:
                forged_path = self.forgery_dir / forged_name
                if forged_path.exists():
                    forged_candidates.append(str(forged_path))

        return forged_candidates
    
    def create_test_set(self,
                       target_identity_dir: str,
                       distractor_identity_dirs: List[str],
                       num_query: int = 1,
                       num_target_candidates: int = 4,
                       num_distractor_candidates: int = 4) -> Optional[Dict]:
        """
        Create a single open-set test case.
        
        Args:
            target_identity_dir: Directory of target identity
            distractor_identity_dirs: List of distractor identity directories
            num_query: Number of query images for the target
            num_target_candidates: Number of candidate images from target
            num_distractor_candidates: Number of candidates per distractor
            
        Returns:
            Test set dictionary or None if creation failed
        """
        target_images = list(Path(target_identity_dir).glob('*.jpg')) + \
                       list(Path(target_identity_dir).glob('*.png'))
        target_images = [str(p) for p in target_images]
        
        if len(target_images) < num_query + num_target_candidates:
            return None
        
        # Sample query and target candidate images
        random.shuffle(target_images)
        query_images = target_images[:num_query]
        target_candidates = target_images[num_query:num_query + num_target_candidates]
        
        target_forged_candidates = self._collect_forged_images(target_candidates)

        # Sample distractor candidates
        all_candidates = target_candidates.copy()
        if target_forged_candidates:
            all_candidates.extend(target_forged_candidates)
        
        for dist_dir in distractor_identity_dirs:
            dist_images = list(Path(dist_dir).glob('*.jpg')) + \
                         list(Path(dist_dir).glob('*.png'))
            dist_images = [str(p) for p in dist_images]
            
            if len(dist_images) >= num_distractor_candidates:
                random.shuffle(dist_images)
                all_candidates.extend(dist_images[:num_distractor_candidates])
                dist_forged = self._collect_forged_images(dist_images[:num_distractor_candidates])
                if dist_forged:
                    all_candidates.extend(dist_forged)
        
        # Shuffle candidates
        random.shuffle(all_candidates)
        
        target_identity_id = Path(target_identity_dir).name
        
        return {
            'target_identity_id': target_identity_id,
            'query_images': query_images,
            'target_candidates': target_candidates,
            'target_forged_candidates': target_forged_candidates,
            'candidate_images': all_candidates,
            'num_target_candidates': len(target_candidates),
            'num_total_candidates': len(all_candidates),
            'num_distractors': len(distractor_identity_dirs)
        }
    
    def evaluate_test_set(self, test_set: Dict) -> Optional[Dict]:
        """
        Evaluate a single test set.
        
        Args:
            test_set: Test set dictionary from create_test_set
            
        Returns:
            Evaluation results or None if evaluation failed
        """
        # Extract query embeddings
        query_embeddings = []
        for img_path in test_set['query_images']:
            emb = self.get_embedding(img_path, apply_defense=False)  # Query not defended
            if emb is not None:
                query_embeddings.append(emb)
        
        if len(query_embeddings) == 0:
            return None
        
        # Average query embeddings
        query_embedding = np.mean(query_embeddings, axis=0)
        
        # Extract candidate embeddings (with defense applied)
        similarities = []
        for img_path in test_set['candidate_images']:
            emb = self.get_embedding(img_path, apply_defense=True)
            if emb is not None:
                sim = self.compute_similarity(query_embedding, emb)
                similarities.append((img_path, sim))
        
        if len(similarities) == 0:
            return None
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Check if top-1 is correct (belongs to target identity)
        top1_image = similarities[0][0]
        top1_correct = top1_image in test_set['target_candidates']
        
        # Calculate mean reciprocal rank (MRR)
        ranks = []
        for i, (img, _) in enumerate(similarities):
            if img in test_set['target_candidates']:
                ranks.append(i + 1)  # 1-indexed rank
        
        mrr = 1.0 / ranks[0] if ranks else 0.0
        
        return {
            'target_identity_id': test_set['target_identity_id'],
            'num_candidates': len(similarities),
            'num_target_candidates': test_set['num_target_candidates'],
            'num_distractors': test_set['num_distractors'],
            'top1_prediction': top1_image,
            'top1_similarity': float(similarities[0][1]),
            'top1_correct': top1_correct,
            'mrr': mrr,
            'first_correct_rank': ranks[0] if ranks else None,
            'all_similarities': [(img, float(sim)) for img, sim in similarities]
        }
    
    def run_evaluation(self,
                      identity_dirs: List[str],
                      num_tests: int = 50,
                      num_distractors: int = 19,
                      show_progress: bool = True) -> List[Dict]:
        """
        Run full open-set evaluation.
        
        Args:
            identity_dirs: List of identity directories
            num_tests: Number of test cases to create
            num_distractors: Number of distractor identities per test
            show_progress: Whether to show progress bar
            
        Returns:
            List of evaluation results
        """
        if len(identity_dirs) < num_distractors + 1:
            raise ValueError(f"Need at least {num_distractors + 1} identities, got {len(identity_dirs)}")
        
        results = []
        iterator = range(num_tests)
        if show_progress:
            iterator = tqdm(iterator, desc="Evaluating test sets")
        
        for _ in iterator:
            # Sample target and distractors
            sampled = random.sample(identity_dirs, num_distractors + 1)
            target_dir = sampled[0]
            distractor_dirs = sampled[1:]
            
            # Create test set
            test_set = self.create_test_set(
                target_dir,
                distractor_dirs,
                num_query=1,
                num_target_candidates=4,
                num_distractor_candidates=4
            )
            
            if test_set is None:
                continue
            
            # Evaluate test set
            result = self.evaluate_test_set(test_set)
            if result is not None:
                results.append(result)
        
        return results


def calculate_metrics(results: List[Dict]) -> Dict:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Dictionary of metrics
    """
    if not results:
        return {}
    
    total = len(results)
    
    # Accuracy metrics
    top1_correct = sum(1 for r in results if r['top1_correct'])
    
    top1_accuracy = top1_correct / total
    
    # MRR
    mrr = np.mean([r['mrr'] for r in results])
    
    # Similarity statistics
    similarities = [r['top1_similarity'] for r in results]
    
    correct_sims = [r['top1_similarity'] for r in results if r['top1_correct']]
    incorrect_sims = [r['top1_similarity'] for r in results if not r['top1_correct']]
    
    metrics = {
        'total_tests': total,
        'top1_accuracy': float(top1_accuracy),
        'top1_correct_count': top1_correct,
        'mrr': float(mrr),
        
        'avg_similarity': float(np.mean(similarities)),
        'std_similarity': float(np.std(similarities)),
        'min_similarity': float(np.min(similarities)),
        'max_similarity': float(np.max(similarities)),
    }
    
    if correct_sims:
        metrics['avg_similarity_correct'] = float(np.mean(correct_sims))
        metrics['std_similarity_correct'] = float(np.std(correct_sims))
    
    if incorrect_sims:
        metrics['avg_similarity_incorrect'] = float(np.mean(incorrect_sims))
        metrics['std_similarity_incorrect'] = float(np.std(incorrect_sims))
    
    # Precision, Recall, F1
    true_positives = top1_correct
    false_positives = total - top1_correct
    false_negatives = total - top1_correct
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics.update({
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score)
    })
    
    return metrics


def load_identity_directories(data_dir: str, min_images: int = 10) -> List[str]:
    """
    Load identity directories with sufficient images.
    
    Args:
        data_dir: Root directory with identity subdirectories
        min_images: Minimum number of images required per identity
        
    Returns:
        List of identity directory paths
    """
    data_path = Path(data_dir)
    identity_dirs = []
    
    for d in data_path.iterdir():
        if not d.is_dir():
            continue
        
        images = list(d.glob('*.jpg')) + list(d.glob('*.png'))
        if len(images) >= min_images:
            identity_dirs.append(str(d))
    
    return identity_dirs


def print_metrics(metrics: Dict, defense_name: str = ""):
    """Print evaluation metrics in a formatted way."""
    print("\n" + "="*60)
    print(f"EVALUATION METRICS{' - ' + defense_name if defense_name else ''}")
    print("="*60)
    
    print(f"\nAccuracy:")
    print(f"  Top-1: {metrics['top1_accuracy']:.4f} ({metrics['top1_correct_count']}/{metrics['total_tests']})")
    print(f"  MRR:   {metrics['mrr']:.4f}")
    
    print(f"\nSimilarity Statistics:")
    print(f"  Overall: {metrics['avg_similarity']:.4f} ± {metrics['std_similarity']:.4f}")
    
    if 'avg_similarity_correct' in metrics:
        print(f"  Correct: {metrics['avg_similarity_correct']:.4f} ± {metrics['std_similarity_correct']:.4f}")
    
    if 'avg_similarity_incorrect' in metrics:
        print(f"  Incorrect: {metrics['avg_similarity_incorrect']:.4f} ± {metrics['std_similarity_incorrect']:.4f}")
    
    print(f"\nClassification Metrics:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Open-set identity re-identification evaluation with defenses'
    )
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory with organized identity images')
    parser.add_argument('--forgery-dir', type=str, default='data/forgeries',
                       help='Directory with forged images (optional)')
    parser.add_argument('--forgery-mapping', type=str,
                       default='data/forgeries/mapping.json',
                       help='Path to forgery mapping JSON (optional)')
    parser.add_argument('--num-forged-per-real', type=int, default=9,
                       help='Number of forged images to sample per real image at runtime')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='facenet', 
                       choices=['facenet', 'arcface'],
                       help='Face recognition model')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    
    # Defense parameters
    parser.add_argument('--defense', type=str, default='none',
                       help='Defense mechanism name (see defense_mechanisms.py)')
    parser.add_argument('--list-defenses', action='store_true',
                       help='List available defense mechanisms and exit')
    
    # Evaluation parameters
    parser.add_argument('--num-identities', type=int, default=20,
                       help='Total number of identities to use (N)')
    parser.add_argument('--num-tests', type=int, default=50,
                       help='Number of test cases to evaluate')
    parser.add_argument('--num-distractors', type=int, default=None,
                       help='Number of distractor identities per test (default: N-1)')
    parser.add_argument('--min-images', type=int, default=10,
                       help='Minimum images per identity')
    
    # Output parameters
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file for results')
    parser.add_argument('--save-detailed', action='store_true',
                       help='Save detailed per-test results')
    
    # Random seed
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # List defenses if requested
    if args.list_defenses:
        configs = create_defense_configs()
        print("\nAvailable Defense Mechanisms:")
        print("="*60)
        for name, config in configs.items():
            defense = EmbeddingDefense(config)
            info = defense.get_info()
            print(f"\n{name}:")
            print(f"  Mechanism: {info['mechanism']}")
            if 'privacy_guarantee' in info:
                print(f"  Privacy: {info['privacy_guarantee']}")
        return
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Print configuration
    print("="*60)
    print("Open-Set Identity Re-identification Evaluation")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Defense: {args.defense}")
    print(f"Number of identities: {args.num_identities}")
    print(f"Number of tests: {args.num_tests}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print()
    
    # Load identity directories
    print("Loading identity directories...")
    all_identity_dirs = load_identity_directories(args.data_dir, args.min_images)
    print(f"Found {len(all_identity_dirs)} identities with >= {args.min_images} images")
    
    if len(all_identity_dirs) < args.num_identities:
        print(f"Warning: Only {len(all_identity_dirs)} identities available, requested {args.num_identities}")
        args.num_identities = len(all_identity_dirs)
    
    # Sample identities
    identity_dirs = random.sample(all_identity_dirs, args.num_identities)
    print(f"Sampled {len(identity_dirs)} identities for evaluation")
    
    # Set num_distractors
    if args.num_distractors is None:
        args.num_distractors = args.num_identities - 1
    
    # Initialize model
    print(f"\nInitializing {args.model} model...")
    extractor = FaceEmbeddingExtractor(model_name=args.model, device=args.device)
    
    # Initialize defense
    defense = None
    if args.defense != 'none':
        configs = create_defense_configs()
        if args.defense not in configs:
            raise ValueError(f"Unknown defense: {args.defense}. Use --list-defenses to see available options.")
        
        defense_config = configs[args.defense]
        defense = EmbeddingDefense(defense_config)
        print(f"\nDefense mechanism: {args.defense}")
        print(json.dumps(defense.get_info(), indent=2))
    
    # Initialize evaluator
    evaluator = OpenSetEvaluator(
        extractor,
        defense,
        forgery_dir=args.forgery_dir,
        forgery_mapping_path=args.forgery_mapping,
        num_forged_per_real=args.num_forged_per_real
    )
    
    # Run evaluation
    print("\nRunning evaluation...")
    results = evaluator.run_evaluation(
        identity_dirs,
        num_tests=args.num_tests,
        num_distractors=args.num_distractors,
        show_progress=True
    )
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Print metrics
    print_metrics(metrics, args.defense)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'args': vars(args),
        'metrics': metrics,
        'defense_info': defense.get_info() if defense else {'mechanism': 'none'}
    }
    
    if args.save_detailed:
        # Remove detailed similarities to save space
        for r in results:
            if 'all_similarities' in r:
                r['all_similarities'] = r['all_similarities'][:10]  # Keep top 10 only
        output_data['results'] = results
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()

