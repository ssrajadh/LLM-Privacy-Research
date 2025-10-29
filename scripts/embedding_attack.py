#!/usr/bin/env python3
"""
Embedding-based Identity Re-identification Attack

This script implements a baseline attack using pre-trained face recognition models
(FaceNet, ArcFace) to perform identity re-identification on shuffled image sets.

Usage:
    python scripts/embedding_attack.py \
        --data-dir data/organized \
        --model facenet \
        --num-identities 10 \
        --output results/baseline_attack.json

Models supported:
    - facenet: FaceNet (InceptionResnetV1) trained on VGGFace2
    - arcface: ArcFace (ResNet) via InsightFace
"""

import os
import argparse
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import torch
import torch.nn.functional as F
from PIL import Image
import cv2

try:
    from facenet_pytorch import InceptionResnetV1, MTCNN
    FACENET_AVAILABLE = True
except ImportError:
    FACENET_AVAILABLE = False
    print("Warning: facenet-pytorch not available. Install with: pip install facenet-pytorch")

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except (ImportError, ValueError) as e:
    INSIGHTFACE_AVAILABLE = False
    print(f"Warning: insightface not available ({type(e).__name__}). Will use FaceNet only.")


class FaceEmbeddingExtractor:
    
    def __init__(self, model_name='facenet', device='cuda'):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        if model_name == 'facenet':
            if not FACENET_AVAILABLE:
                raise ImportError("facenet-pytorch is required for FaceNet model")
            self._init_facenet()
        elif model_name == 'arcface':
            if not INSIGHTFACE_AVAILABLE:
                raise ImportError("insightface is required for ArcFace model")
            self._init_arcface()
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _init_facenet(self):
        print("Loading FaceNet model (InceptionResnetV1 trained on VGGFace2)...")
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709,
            post_process=True, device=self.device
        )
        print("FaceNet model loaded successfully")
    
    def _init_arcface(self):
        print("Loading ArcFace model via InsightFace...")
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0 if self.device == 'cuda' else -1, det_size=(640, 640))
        print("ArcFace model loaded successfully")
    
    def extract_embedding(self, image_path: str) -> Optional[np.ndarray]:
        try:
            if self.model_name == 'facenet':
                return self._extract_facenet(image_path)
            elif self.model_name == 'arcface':
                return self._extract_arcface(image_path)
        except Exception as e:
            print(f"Error extracting embedding from {image_path}: {e}")
            return None
    
    def _extract_facenet(self, image_path: str) -> Optional[np.ndarray]:
        img = Image.open(image_path).convert('RGB')
        
        face = self.mtcnn(img)
        if face is None:
            return None
        
        with torch.no_grad():
            face = face.unsqueeze(0).to(self.device)
            embedding = self.model(face)
            embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding.cpu().numpy()[0]
    
    def _extract_arcface(self, image_path: str) -> Optional[np.ndarray]:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        faces = self.app.get(img)
        if len(faces) == 0:
            return None
        
        face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        embedding = face.normed_embedding
        
        return embedding
    
    def extract_embeddings_batch(self, image_paths: List[str], 
                                 show_progress: bool = True) -> Dict[str, np.ndarray]:
        embeddings = {}
        iterator = tqdm(image_paths, desc="Extracting embeddings") if show_progress else image_paths
        
        for img_path in iterator:
            emb = self.extract_embedding(img_path)
            if emb is not None:
                embeddings[img_path] = emb
        
        return embeddings


class IdentityReidentificationAttack:
    
    def __init__(self, extractor: FaceEmbeddingExtractor):
        self.extractor = extractor
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    
    def predict_identity(self, query_embedding: np.ndarray, 
                        candidate_embeddings: Dict[str, np.ndarray],
                        top_k: int = 1) -> List[Tuple[str, float]]:
        similarities = []
        for img_path, emb in candidate_embeddings.items():
            sim = self.compute_similarity(query_embedding, emb)
            similarities.append((img_path, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def attack_single_identity(self, identity_dir: str, 
                               num_forge: int = 9,
                               num_query: int = 1) -> Dict:
        images = [str(p) for p in Path(identity_dir).glob('*.jpg')]
        if len(images) < num_query + 1:
            return None
        
        random.shuffle(images)
        query_images = images[:num_query]
        target_image = images[num_query]
        
        query_embeddings = []
        for img in query_images:
            emb = self.extractor.extract_embedding(img)
            if emb is not None:
                query_embeddings.append(emb)
        
        if len(query_embeddings) == 0:
            return None
        
        query_embedding = np.mean(query_embeddings, axis=0)
        
        target_embedding = self.extractor.extract_embedding(target_image)
        if target_embedding is None:
            return None
        
        candidate_images = [target_image]
        
        # TODO: Add forged images from other identities
        # will use a simple shuffled set from the same identity for now
        other_images = images[num_query + 1:]
        if len(other_images) >= num_forge:
            candidate_images.extend(other_images[:num_forge])
        
        candidate_embeddings = {}
        for img in candidate_images:
            emb = self.extractor.extract_embedding(img)
            if emb is not None:
                candidate_embeddings[img] = emb
        
        predictions = self.predict_identity(query_embedding, candidate_embeddings, top_k=len(candidate_embeddings))
        
        top1_correct = predictions[0][0] == target_image if predictions else False
        
        return {
            'identity_dir': identity_dir,
            'query_images': query_images,
            'target_image': target_image,
            'candidate_images': candidate_images,
            'predictions': [(p, float(s)) for p, s in predictions],
            'top1_prediction': predictions[0][0] if predictions else None,
            'top1_similarity': float(predictions[0][1]) if predictions else 0.0,
            'top1_correct': top1_correct
        }


def load_identity_directories(data_dir: str, num_identities: Optional[int] = None,
                              min_images: int = 10) -> List[str]:
    data_path = Path(data_dir)
    identity_dirs = [str(d) for d in data_path.iterdir() if d.is_dir()]
    
    filtered_dirs = []
    for d in identity_dirs:
        num_imgs = len(list(Path(d).glob('*.jpg')))
        if num_imgs >= min_images:
            filtered_dirs.append(d)
    
    print(f"Found {len(filtered_dirs)} identities with >= {min_images} images")
    
    if num_identities is not None and len(filtered_dirs) > num_identities:
        filtered_dirs = random.sample(filtered_dirs, num_identities)
        print(f"Sampled {num_identities} identities for testing")
    
    return filtered_dirs


def calculate_metrics(results: List[Dict]) -> Dict:
    if not results:
        return {}
    
    total = len(results)
    correct = sum(1 for r in results if r['top1_correct'])
    
    top1_accuracy = correct / total if total > 0 else 0.0
    
    true_positives = correct
    false_positives = total - correct
    false_negatives = total - correct
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    avg_top1_similarity = np.mean([r['top1_similarity'] for r in results])
    
    return {
        'total_tests': total,
        'correct_predictions': correct,
        'top1_accuracy': top1_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'avg_top1_similarity': float(avg_top1_similarity)
    }


def main():
    parser = argparse.ArgumentParser(description='Embedding-based Identity Re-identification Attack')
    parser.add_argument('--data-dir', type=str, default='data/organized',
                       help='Directory with organized identity images')
    parser.add_argument('--model', type=str, default='facenet', choices=['facenet', 'arcface'],
                       help='Face recognition model to use')
    parser.add_argument('--num-identities', type=int, default=10,
                       help='Number of identities to test (None for all)')
    parser.add_argument('--min-images', type=int, default=10,
                       help='Minimum images per identity')
    parser.add_argument('--num-forge', type=int, default=9,
                       help='Number of forged images per test')
    parser.add_argument('--num-query', type=int, default=1,
                       help='Number of query images for known embedding')
    parser.add_argument('--output', type=str, default='results/baseline_attack.json',
                       help='Output file for results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Embedding-based Identity Re-identification Attack")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Data directory: {args.data_dir}")
    print(f"Number of identities: {args.num_identities}")
    print(f"Random seed: {args.seed}")
    print()
    
    extractor = FaceEmbeddingExtractor(model_name=args.model, device=args.device)
    
    attack = IdentityReidentificationAttack(extractor)
    
    identity_dirs = load_identity_directories(
        args.data_dir, 
        num_identities=args.num_identities,
        min_images=args.min_images
    )
    
    print("\nRunning attacks...")
    results = []
    for identity_dir in tqdm(identity_dirs, desc="Processing identities"):
        result = attack.attack_single_identity(
            identity_dir,
            num_forge=args.num_forge,
            num_query=args.num_query
        )
        if result is not None:
            results.append(result)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    metrics = calculate_metrics(results)
    print(f"Total tests: {metrics['total_tests']}")
    print(f"Correct predictions: {metrics['correct_predictions']}")
    print(f"\nTop-1 Accuracy: {metrics['top1_accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Average Top-1 Similarity: {metrics['avg_top1_similarity']:.4f}")
    
    output_data = {
        'args': vars(args),
        'metrics': metrics,
        'results': results
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()

