"""Central configuration settings for EUTERPE."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class PathConfig:
    """Directory and file path locations."""
    root_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data")
    raw_data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "raw")
    processed_data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "processed")
    splits_data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "splits")
    checkpoints_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "models" / "checkpoints")
    docs_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "docs")


@dataclass(frozen=True)
class DataConfig:
    """Data processing, cleaning, and filtering parameters."""
    min_user_scrobbles: int = 50
    min_track_scrobbles: int = 5
    duplicate_time_delta_seconds: int = 30
    confidence_alpha: float = 40.0
    
    # Chronological split ratios
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    
    # Minimum test events required for warm-start evaluation
    min_test_interactions_warm: int = 5


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture and training hyperparameters."""
    latent_dim: int = 64
    als_regularization: float = 0.05
    als_iterations: int = 25
    
    bpr_learning_rate: float = 0.05
    bpr_regularization: float = 0.005
    bpr_epochs: int = 50
    bpr_negative_samples: int = 5
    
    neural_learning_rate: float = 1e-3
    neural_batch_size: int = 512
    neural_epochs: int = 30
    neural_temperature: float = 0.07


@dataclass(frozen=True)
class DiscoveryConfig:
    """Discovery metrics and reranking parameters."""
    top_k: int = 20
    candidate_m: int = 100
    mmr_lambdas: List[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8, 1.0])
    default_mmr_lambda: float = 0.6
    bootstrap_iterations: int = 1000
    significance_alpha: float = 0.01


@dataclass(frozen=True)
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)


default_config = Config()
