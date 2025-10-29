#!/usr/bin/env python3
"""
Evaluate embedding-based identity re-identification attack on shuffled test sets

This script evaluates the attack performance on pre-generated shuffled sets.

Usage:
    python scripts/evaluate_attack.py \
        --test-sets data/shuffled_sets/test_sets_metadata.json \
        --model facenet \
        --output results/evaluation_results.json
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import torch

import sys
sys.path.append(str(Path(__file__).parent))
from embedding_attack import FaceEmbeddingExtractor


class AttackEvaluator:
    
    def __init__(self, extractor: FaceEmbeddingExtractor):
        self.extractor = extractor
        self.embedding_cache = {}
    
    def get_embedding(self, image_path: str, use_cache: bool = True) -> np.ndarray:
        if use_cache and image_path in self.embedding_cache:
            return self.embedding_cache[image_path]
        
        emb = self.extractor.extract_embedding(image_path)
        if emb is not None and use_cache:
            self.embedding_cache[image_path] = emb
        
        return emb
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    
    def evaluate_test_set(self, test_set: Dict) -> Dict:
        query_embeddings = []
        for img_path in test_set['query_images']:
            emb = self.get_embedding(img_path)
            if emb is not None:
                query_embeddings.append(emb)
        
        if len(query_embeddings) == 0:
            return None
        
        query_embedding = np.mean(query_embeddings, axis=0)
        
        similarities = []
        for img_path in test_set['candidate_images']:
            emb = self.get_embedding(img_path)
            if emb is not None:
                sim = self.compute_similarity(query_embedding, emb)
                similarities.append((img_path, sim))
        
        if len(similarities) == 0:
            return None
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        top1_image = similarities[0][0]
        top1_similarity = similarities[0][1]
        
        is_correct = top1_image in test_set['real_images']
        
        return {
            'set_id': test_set.get('set_id', -1),
            'target_identity_id': test_set['target_identity_id'],
            'top1_prediction': top1_image,
            'top1_similarity': float(top1_similarity),
            'top1_correct': is_correct,
            'all_predictions': [(img, float(sim)) for img, sim in similarities]
        }
    
    def evaluate_all(self, test_sets: List[Dict], show_progress: bool = True) -> List[Dict]:
        results = []
        iterator = tqdm(test_sets, desc="Evaluating test sets") if show_progress else test_sets
        
        for test_set in iterator:
            result = self.evaluate_test_set(test_set)
            if result is not None:
                results.append(result)
        
        return results


def calculate_comprehensive_metrics(results: List[Dict]) -> Dict:
    if not results:
        return {}
    
    total = len(results)
    
    top1_correct = sum(1 for r in results if r['top1_correct'])
    
    metrics = {
        'total_tests': total,
        'successful_tests': sum(1 for r in results if r is not None),
        'top1_accuracy': top1_correct / total if total > 0 else 0.0,
        'top1_correct_count': top1_correct
    }
    
    true_positives = top1_correct
    false_positives = total - top1_correct
    false_negatives = total - top1_correct
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics.update({
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    })
    
    similarities = [r['top1_similarity'] for r in results]
    metrics.update({
        'avg_similarity': float(np.mean(similarities)),
        'std_similarity': float(np.std(similarities)),
        'min_similarity': float(np.min(similarities)),
        'max_similarity': float(np.max(similarities))
    })
    
    correct_sims = [r['top1_similarity'] for r in results if r['top1_correct']]
    incorrect_sims = [r['top1_similarity'] for r in results if not r['top1_correct']]
    
    if correct_sims:
        metrics['avg_similarity_correct'] = float(np.mean(correct_sims))
        metrics['std_similarity_correct'] = float(np.std(correct_sims))
    
    if incorrect_sims:
        metrics['avg_similarity_incorrect'] = float(np.mean(incorrect_sims))
        metrics['std_similarity_incorrect'] = float(np.std(incorrect_sims))
    
    return metrics


def print_metrics(metrics: Dict):
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    print(f"\nTest Set Statistics:")
    print(f"  Total tests: {metrics.get('total_tests', 0)}")
    print(f"  Successful tests: {metrics.get('successful_tests', 0)}")
    
    print(f"\nTop-1 Accuracy:")
    print(f"  {metrics.get('top1_accuracy', 0):.4f} ({metrics.get('top1_correct_count', 0)}/{metrics.get('total_tests', 0)})")
    
    print(f"\nClassification Metrics:")
    print(f"  Precision: {metrics.get('precision', 0):.4f}")
    print(f"  Recall: {metrics.get('recall', 0):.4f}")
    print(f"  F1 Score: {metrics.get('f1_score', 0):.4f}")
    
    print(f"\nSimilarity Statistics:")
    print(f"  Average: {metrics.get('avg_similarity', 0):.4f} ± {metrics.get('std_similarity', 0):.4f}")
    print(f"  Range: [{metrics.get('min_similarity', 0):.4f}, {metrics.get('max_similarity', 0):.4f}]")
    
    if 'avg_similarity_correct' in metrics:
        print(f"\n  Correct predictions: {metrics['avg_similarity_correct']:.4f} ± {metrics['std_similarity_correct']:.4f}")
    if 'avg_similarity_incorrect' in metrics:
        print(f"  Incorrect predictions: {metrics['avg_similarity_incorrect']:.4f} ± {metrics['std_similarity_incorrect']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate identity re-identification attack')
    parser.add_argument('--test-sets', type=str, required=True,
                       help='Path to test sets metadata JSON file')
    parser.add_argument('--model', type=str, default='facenet', choices=['facenet', 'arcface'],
                       help='Face recognition model to use')
    parser.add_argument('--output', type=str, default='results/evaluation_results.json',
                       help='Output file for results')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--max-sets', type=int, default=None,
                       help='Maximum number of test sets to evaluate (for testing)')
    parser.add_argument('--save-predictions', action='store_true',
                       help='Save detailed predictions to output file')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Identity Re-identification Attack Evaluation")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Test sets: {args.test_sets}")
    print(f"Device: {args.device}")
    print()
    
    print("Loading test sets...")
    with open(args.test_sets, 'r') as f:
        test_sets = json.load(f)
    
    if args.max_sets is not None:
        test_sets = test_sets[:args.max_sets]
        print(f"Limited to {args.max_sets} test sets for evaluation")
    
    print(f"Loaded {len(test_sets)} test sets")
    
    print(f"\nInitializing {args.model} model...")
    extractor = FaceEmbeddingExtractor(model_name=args.model, device=args.device)
    
    evaluator = AttackEvaluator(extractor)
    
    print("\nRunning evaluation...")
    results = evaluator.evaluate_all(test_sets)
    
    metrics = calculate_comprehensive_metrics(results)
    
    print_metrics(metrics)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'args': vars(args),
        'metrics': metrics,
    }
    
    if args.save_predictions:
        output_data['results'] = results
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()

