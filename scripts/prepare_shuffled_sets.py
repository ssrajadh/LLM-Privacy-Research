#!/usr/bin/env python3
"""
Prepare shuffled test sets for identity re-identification attacks

This script creates test scenarios where real images are mixed with forged/other
identity images to simulate the attack scenario described in the research.

Usage:
    python scripts/prepare_shuffled_sets.py \
        --data-dir data/organized \
        --output data/shuffled_sets \
        --num-sets 100 \
        --set-size 10
"""

import os
import json
import random
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm


class ShuffledSetGenerator:
    """Generate shuffled test sets with real and forged images"""
    
    def __init__(self, data_dir: str, forgery_dir: str = None, forgery_mapping: str = None, 
                 min_images_per_identity: int = 10):
        self.data_dir = Path(data_dir)
        self.forgery_dir = Path(forgery_dir) if forgery_dir else None
        self.min_images = min_images_per_identity
        self.identity_dirs = self._load_identities()
        print(f"Loaded {len(self.identity_dirs)} identities with >= {min_images_per_identity} images")
        
        # Load forgery mapping
        self.forgery_mapping = {}
        self.image_path_lookup = {}  # basename -> full path mapping
        if forgery_mapping and os.path.exists(forgery_mapping):
            self._load_forgery_mapping(forgery_mapping)
            self._build_image_lookup()
            print(f"Loaded {len(self.forgery_mapping)} forged image mappings")
    
    def _load_identities(self) -> List[Dict]:
        """Load all identity directories and their images"""
        identities = []
        for id_dir in self.data_dir.iterdir():
            if not id_dir.is_dir():
                continue
            
            images = list(id_dir.glob('*.jpg'))
            if len(images) >= self.min_images:
                identities.append({
                    'id': id_dir.name,
                    'dir': str(id_dir),
                    'images': [str(img) for img in images]
                })
        
        return identities
    
    def _load_forgery_mapping(self, mapping_file: str):
        """Load mapping from real images to forged images (supports V1 and V2 formats)"""
        with open(mapping_file, 'r') as f:
            data = json.load(f)
        
        # Check if this is V2 format (has 'results' key)
        if isinstance(data, dict) and 'results' in data:
            # V2 format: nested structure with parameter_sets
            for result in data['results']:
                real_img = result['real_image']
                forged_imgs = []
                # Collect all forgeries across all parameter sets
                for param_set in result['parameter_sets']:
                    for forgery in param_set['forgeries']:
                        forged_imgs.append(forgery['filename'])
                self.forgery_mapping[real_img] = forged_imgs
        else:
            # V1 format: simple list of mappings
            for item in data:
                real_img = item['real']
                forged_imgs = item['forged']
                # Store with just filename as key
                self.forgery_mapping[real_img] = forged_imgs
    
    def _build_image_lookup(self):
        """Build lookup table: basename -> {path, identity_id}"""
        for identity in self.identity_dirs:
            for img_path in identity['images']:
                basename = Path(img_path).name
                self.image_path_lookup[basename] = {
                    'path': img_path,
                    'identity_id': identity['id'],
                    'identity': identity
                }
    
    def create_shuffled_set_from_real(self, real_filename: str, img_info: Dict, num_query: int = 1) -> Dict:
        """
        Create a shuffled test set for a specific real image
        Uses ONLY the forged images specified in mapping.json
        """
        target_identity = img_info['identity']
        real_img_path = img_info['path']
        
        # Get query images from same identity (exclude the real image)
        available_for_query = [img for img in target_identity['images'] if img != real_img_path]
        if len(available_for_query) < num_query:
            return None
        
        query_images = random.sample(available_for_query, num_query)
        real_images = [real_img_path]
        
        # Get forged images from mapping
        forged_images = []
        if real_filename in self.forgery_mapping:
            forged_list = self.forgery_mapping[real_filename]
            forged_paths = [str(self.forgery_dir / f) for f in forged_list]
            forged_images.extend(forged_paths)
        
        # Combine and shuffle
        candidate_images = real_images + forged_images
        random.shuffle(candidate_images)
        
        return {
            'target_identity_id': target_identity['id'],
            'query_images': query_images,
            'real_images': real_images,
            'forged_images': forged_images,
            'candidate_images': candidate_images,
            'num_real': 1,
            'num_forged': len(forged_images)
        }
    
    def create_shuffled_set(self, 
                           target_identity: Dict,
                           num_real: int = 1,
                           num_forged: int = 9,
                           num_query: int = 1) -> Dict:
        """
        Create a single shuffled set for testing
        
        Args:
            target_identity: Identity dict with images
            num_real: Number of real images to include (usually 1)
            num_forged: Number of forged/other identity images
            num_query: Number of query images for the known embedding
            
        Returns:
            Dict containing:
                - target_identity_id
                - query_images: List of paths for known user
                - real_images: List of paths for real images to find
                - forged_images: List of paths for forged/other images
                - shuffled_images: Combined list of all candidate images
        """
        if len(target_identity['images']) < num_query + num_real:
            return None
        
        sampled_images = random.sample(target_identity['images'], num_query + num_real)
        query_images = sampled_images[:num_query]
        real_images = sampled_images[num_query:num_query + num_real]
        
        # Sample forged images
        forged_images = []
        
        if self.forgery_mapping and self.forgery_dir:
            # ONLY use generated forged images from mapping.json
            for real_img_path in real_images:
                real_filename = Path(real_img_path).name
                
                if real_filename in self.forgery_mapping:
                    # Get forged images for this real image
                    forged_list = self.forgery_mapping[real_filename]
                    # Convert to full paths
                    forged_paths = [str(self.forgery_dir / f) for f in forged_list]
                    
                    # Use ALL forged images from mapping (typically 3 per real image)
                    forged_images.extend(forged_paths)
        else:
            # Fallback: if no forgery mapping, use random images from other identities
            other_identities = [id_dict for id_dict in self.identity_dirs 
                               if id_dict['id'] != target_identity['id']]
            
            for _ in range(num_forged):
                other_id = random.choice(other_identities)
                forged_img = random.choice(other_id['images'])
                forged_images.append(forged_img)
        
        candidate_images = real_images + forged_images
        random.shuffle(candidate_images)
        
        return {
            'target_identity_id': target_identity['id'],
            'query_images': query_images,
            'real_images': real_images,
            'forged_images': forged_images,
            'candidate_images': candidate_images,
            'num_real': num_real,
            'num_forged': num_forged
        }
    
    def generate_test_sets(self, 
                          num_sets: int,
                          num_real: int = 1,
                          num_forged: int = 9,
                          num_query: int = 1) -> List[Dict]:
        """Generate multiple shuffled test sets"""
        test_sets = []
        
        # If using forgery mapping, only use images that have forged versions
        if self.forgery_mapping:
            # Get all real images that have forged versions
            available_real_images = []
            for real_filename in self.forgery_mapping.keys():
                if real_filename in self.image_path_lookup:
                    available_real_images.append(real_filename)
            
            if not available_real_images:
                print("ERROR: No real images from mapping found in data directory!")
                return test_sets
            
            print(f"Using {len(available_real_images)} real images from forgery mapping")
            
            # Generate test sets using only mapped real images
            used_images = set()
            for i in tqdm(range(num_sets), desc="Generating test sets"):
                # Pick a real image that hasn't been used yet (if possible)
                available = [img for img in available_real_images if img not in used_images]
                if not available:
                    # If all used, reset
                    available = available_real_images
                    used_images.clear()
                
                real_filename = random.choice(available)
                used_images.add(real_filename)
                
                img_info = self.image_path_lookup[real_filename]
                test_set = self.create_shuffled_set_from_real(
                    real_filename,
                    img_info,
                    num_query=num_query
                )
                
                if test_set is not None:
                    test_set['set_id'] = i
                    test_sets.append(test_set)
        else:
            # Original behavior: random identities
            for i in tqdm(range(num_sets), desc="Generating test sets"):
                target_identity = random.choice(self.identity_dirs)
                
                test_set = self.create_shuffled_set(
                    target_identity,
                    num_real=num_real,
                    num_forged=num_forged,
                    num_query=num_query
                )
                
                if test_set is not None:
                    test_set['set_id'] = i
                    test_sets.append(test_set)
        
        return test_sets
    
    def save_test_sets(self, test_sets: List[Dict], output_dir: str, copy_images: bool = False):
        """Save test sets to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        metadata_file = output_path / 'test_sets_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(test_sets, f, indent=2)
        
        print(f"Saved test sets metadata to: {metadata_file}")
        
        if copy_images:
            print("Copying images to test set directories...")
            for test_set in tqdm(test_sets, desc="Copying images"):
                set_dir = output_path / f"set_{test_set['set_id']:04d}"
                set_dir.mkdir(exist_ok=True)
                
                query_dir = set_dir / 'query'
                candidate_dir = set_dir / 'candidates'
                query_dir.mkdir(exist_ok=True)
                candidate_dir.mkdir(exist_ok=True)
                
                for i, img_path in enumerate(test_set['query_images']):
                    dst = query_dir / f"query_{i}_{Path(img_path).name}"
                    shutil.copy2(img_path, dst)
                    
                for i, img_path in enumerate(test_set['candidate_images']):
                    is_real = img_path in test_set['real_images']
                    label = 'real' if is_real else 'forged'
                    dst = candidate_dir / f"candidate_{i:02d}_{label}_{Path(img_path).name}"
                    shutil.copy2(img_path, dst)


def main():
    parser = argparse.ArgumentParser(description='Prepare shuffled test sets for identity re-identification')
    parser.add_argument('--data-dir', type=str, default='data/organized',
                       help='Directory with organized identity images')
    parser.add_argument('--forgery-dir', type=str, default=None,
                       help='Directory containing generated forged images (optional)')
    parser.add_argument('--forgery-mapping', type=str, default=None,
                       help='JSON file mapping real images to forged images (optional)')
    parser.add_argument('--output', type=str, default='data/shuffled_sets',
                       help='Output directory for test sets')
    parser.add_argument('--num-sets', type=int, default=100,
                       help='Number of test sets to generate')
    parser.add_argument('--num-real', type=int, default=1,
                       help='Number of real images per set')
    parser.add_argument('--num-forged', type=int, default=9,
                       help='Number of forged images per set')
    parser.add_argument('--num-query', type=int, default=1,
                       help='Number of query images per set')
    parser.add_argument('--min-images', type=int, default=10,
                       help='Minimum images per identity to include')
    parser.add_argument('--copy-images', action='store_true',
                       help='Copy images to test set directories (requires more disk space)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print("="*60)
    print("Shuffled Test Set Generation")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output}")
    print(f"Number of sets: {args.num_sets}")
    print(f"Images per set: {args.num_real} real + {args.num_forged} forged")
    print(f"Query images: {args.num_query}")
    print()
    
    generator = ShuffledSetGenerator(
        args.data_dir, 
        forgery_dir=args.forgery_dir,
        forgery_mapping=args.forgery_mapping,
        min_images_per_identity=args.min_images
    )
    
    test_sets = generator.generate_test_sets(
        num_sets=args.num_sets,
        num_real=args.num_real,
        num_forged=args.num_forged,
        num_query=args.num_query
    )
    
    print(f"\nGenerated {len(test_sets)} test sets")
    
    generator.save_test_sets(test_sets, args.output, copy_images=args.copy_images)
    
    print("\nDone!")


if __name__ == '__main__':
    main()

