#!/usr/bin/env python3
"""
Defense Mechanisms for Face Recognition Systems

This module implements various defense mechanisms to protect against
identity re-identification attacks on face embeddings:

1. Embedding Clipping: Clip embedding values to a maximum norm
2. Gaussian Noise Injection: Add calibrated Gaussian noise to embeddings
3. Differential Privacy (DP): Add DP-style perturbations with privacy guarantees

Usage:
    from defense_mechanisms import EmbeddingDefense
    
    defense = EmbeddingDefense(mechanism='gaussian_noise', noise_scale=0.1)
    protected_embedding = defense.apply(original_embedding)
"""

import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class DefenseConfig:
    """Configuration for defense mechanisms"""
    mechanism: str  # 'none', 'clip', 'gaussian_noise', 'dp_laplace', 'dp_gaussian'
    
    # Clipping parameters
    clip_norm: float = 1.0
    
    # Gaussian noise parameters
    noise_scale: float = 0.1  # Standard deviation of noise
    
    # DP parameters
    epsilon: float = 1.0  # Privacy budget
    delta: float = 1e-5  # DP delta parameter (for Gaussian mechanism)
    sensitivity: float = 2.0  # L2 sensitivity of embeddings
    
    # Combined defense parameters
    apply_preprocessing_clip: bool = False  # Clip before adding noise
    preprocessing_clip_norm: float = 1.0


class EmbeddingDefense:
    """
    Implements defense mechanisms for face recognition embeddings.
    
    Supports multiple defense strategies:
    - Embedding clipping: Limits the L2 norm of embeddings
    - Gaussian noise: Adds calibrated Gaussian noise
    - DP-Laplace: Differential privacy with Laplace mechanism
    - DP-Gaussian: Differential privacy with Gaussian mechanism
    """
    
    def __init__(self, config: DefenseConfig):
        """
        Initialize defense mechanism.
        
        Args:
            config: DefenseConfig object with defense parameters
        """
        self.config = config
        self.mechanism = config.mechanism
        
        # Validate mechanism
        valid_mechanisms = ['none', 'clip', 'gaussian_noise', 'dp_laplace', 'dp_gaussian']
        if self.mechanism not in valid_mechanisms:
            raise ValueError(f"Unknown mechanism: {self.mechanism}. Must be one of {valid_mechanisms}")
        
        # Pre-compute DP noise scales if applicable
        if self.mechanism == 'dp_laplace':
            self.noise_scale = self._compute_laplace_scale()
        elif self.mechanism == 'dp_gaussian':
            self.noise_scale = self._compute_gaussian_scale()
    
    def _compute_laplace_scale(self) -> float:
        """
        Compute Laplace noise scale for (ε, 0)-DP.
        
        Scale = sensitivity / ε
        """
        return self.config.sensitivity / self.config.epsilon
    
    def _compute_gaussian_scale(self) -> float:
        """
        Compute Gaussian noise scale for (ε, δ)-DP.
        
        Using the standard Gaussian mechanism formula:
        σ² ≥ 2 * ln(1.25/δ) * sensitivity² / ε²
        """
        numerator = 2 * np.log(1.25 / self.config.delta) * (self.config.sensitivity ** 2)
        denominator = self.config.epsilon ** 2
        variance = numerator / denominator
        return np.sqrt(variance)
    
    def clip_embedding(self, embedding: np.ndarray, max_norm: float) -> np.ndarray:
        """
        Clip embedding to maximum L2 norm.
        
        Args:
            embedding: Input embedding vector
            max_norm: Maximum allowed L2 norm
            
        Returns:
            Clipped embedding
        """
        norm = np.linalg.norm(embedding)
        if norm > max_norm:
            return embedding * (max_norm / norm)
        return embedding.copy()
    
    def add_gaussian_noise(self, embedding: np.ndarray, scale: float) -> np.ndarray:
        """
        Add Gaussian noise to embedding.
        
        Args:
            embedding: Input embedding vector
            scale: Standard deviation of Gaussian noise
            
        Returns:
            Noisy embedding
        """
        noise = np.random.normal(0, scale, size=embedding.shape)
        return embedding + noise
    
    def add_laplace_noise(self, embedding: np.ndarray, scale: float) -> np.ndarray:
        """
        Add Laplace noise to embedding (for DP).
        
        Args:
            embedding: Input embedding vector
            scale: Scale parameter (b) of Laplace distribution
            
        Returns:
            Noisy embedding with DP guarantee
        """
        noise = np.random.laplace(0, scale, size=embedding.shape)
        return embedding + noise
    
    def apply(self, embedding: np.ndarray) -> np.ndarray:
        """
        Apply defense mechanism to embedding.
        
        Args:
            embedding: Original embedding vector
            
        Returns:
            Protected embedding
        """
        if embedding is None:
            return None
        
        # Make a copy to avoid modifying original
        protected = embedding.copy()
        
        # Optional preprocessing clip (useful before adding noise)
        if self.config.apply_preprocessing_clip:
            protected = self.clip_embedding(protected, self.config.preprocessing_clip_norm)
        
        # Apply primary defense mechanism
        if self.mechanism == 'none':
            return protected
        
        elif self.mechanism == 'clip':
            return self.clip_embedding(protected, self.config.clip_norm)
        
        elif self.mechanism == 'gaussian_noise':
            noisy = self.add_gaussian_noise(protected, self.config.noise_scale)
            # Optionally renormalize after adding noise
            # noisy = noisy / np.linalg.norm(noisy)  # Uncomment if needed
            return noisy
        
        elif self.mechanism == 'dp_laplace':
            return self.add_laplace_noise(protected, self.noise_scale)
        
        elif self.mechanism == 'dp_gaussian':
            return self.add_gaussian_noise(protected, self.noise_scale)
        
        return protected
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the defense mechanism.
        
        Returns:
            Dictionary with defense configuration and computed parameters
        """
        info = {
            'mechanism': self.mechanism,
            'config': {
                'clip_norm': self.config.clip_norm,
                'noise_scale': self.config.noise_scale,
                'epsilon': self.config.epsilon,
                'delta': self.config.delta,
                'sensitivity': self.config.sensitivity,
                'apply_preprocessing_clip': self.config.apply_preprocessing_clip,
            }
        }
        
        if self.mechanism in ['dp_laplace', 'dp_gaussian']:
            info['computed_noise_scale'] = self.noise_scale
            
            if self.mechanism == 'dp_laplace':
                info['privacy_guarantee'] = f"({self.config.epsilon}, 0)-DP"
            else:
                info['privacy_guarantee'] = f"({self.config.epsilon}, {self.config.delta})-DP"
        
        return info


def create_defense_configs() -> Dict[str, DefenseConfig]:
    """
    Create a suite of defense configurations for experiments.
    
    Returns:
        Dictionary mapping defense names to DefenseConfig objects
    """
    configs = {
        # Baseline: No defense
        'none': DefenseConfig(mechanism='none'),
        
        # Embedding clipping variants
        'clip_0.5': DefenseConfig(mechanism='clip', clip_norm=0.5),
        'clip_1.0': DefenseConfig(mechanism='clip', clip_norm=1.0),
        'clip_2.0': DefenseConfig(mechanism='clip', clip_norm=2.0),
        
        # Gaussian noise variants
        'noise_0.01': DefenseConfig(mechanism='gaussian_noise', noise_scale=0.01),
        'noise_0.05': DefenseConfig(mechanism='gaussian_noise', noise_scale=0.05),
        'noise_0.1': DefenseConfig(mechanism='gaussian_noise', noise_scale=0.1),
        'noise_0.2': DefenseConfig(mechanism='gaussian_noise', noise_scale=0.2),
        'noise_0.5': DefenseConfig(mechanism='gaussian_noise', noise_scale=0.5),
        
        # DP-Laplace variants (epsilon values)
        'dp_laplace_e1.0': DefenseConfig(
            mechanism='dp_laplace', 
            epsilon=1.0, 
            sensitivity=2.0
        ),
        'dp_laplace_e0.5': DefenseConfig(
            mechanism='dp_laplace', 
            epsilon=0.5, 
            sensitivity=2.0
        ),
        'dp_laplace_e0.1': DefenseConfig(
            mechanism='dp_laplace', 
            epsilon=0.1, 
            sensitivity=2.0
        ),
        
        # DP-Gaussian variants (epsilon, delta)-DP
        'dp_gaussian_e1.0': DefenseConfig(
            mechanism='dp_gaussian', 
            epsilon=1.0, 
            delta=1e-5,
            sensitivity=2.0
        ),
        'dp_gaussian_e0.5': DefenseConfig(
            mechanism='dp_gaussian', 
            epsilon=0.5, 
            delta=1e-5,
            sensitivity=2.0
        ),
        'dp_gaussian_e0.1': DefenseConfig(
            mechanism='dp_gaussian', 
            epsilon=0.1, 
            delta=1e-5,
            sensitivity=2.0
        ),
        
        # Combined defenses: Clip + Noise
        'clip+noise_0.1': DefenseConfig(
            mechanism='gaussian_noise',
            noise_scale=0.1,
            apply_preprocessing_clip=True,
            preprocessing_clip_norm=1.0
        ),
    }
    
    return configs


if __name__ == '__main__':
    """Test defense mechanisms"""
    print("Testing Defense Mechanisms")
    print("=" * 60)
    
    # Create a dummy embedding
    np.random.seed(42)
    embedding = np.random.randn(512)
    embedding = embedding / np.linalg.norm(embedding)  # Normalize
    
    print(f"Original embedding norm: {np.linalg.norm(embedding):.4f}")
    print()
    
    # Test each defense
    configs = create_defense_configs()
    
    for name, config in configs.items():
        defense = EmbeddingDefense(config)
        protected = defense.apply(embedding)
        
        # Compute similarity between original and protected
        similarity = np.dot(embedding, protected) / (
            np.linalg.norm(embedding) * np.linalg.norm(protected)
        )
        
        print(f"Defense: {name}")
        print(f"  Protected norm: {np.linalg.norm(protected):.4f}")
        print(f"  Similarity to original: {similarity:.4f}")
        
        info = defense.get_info()
        if 'privacy_guarantee' in info:
            print(f"  Privacy: {info['privacy_guarantee']}")
        
        print()

