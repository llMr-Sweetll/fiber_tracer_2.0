"""
Configuration management for the Fiber Tracer application.

Author: Chandrashekhar Hegde
"""

import os
import json
import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional, List
import logging
import multiprocessing as mp

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for image processing parameters."""
    chunk_size: int = 100
    scale_factor: int = 1
    num_workers: Optional[int] = None
    median_disk_size: int = 2
    gaussian_sigma: float = 1.0
    clahe_clip_limit: float = 0.03
    
    def __post_init__(self):
        if self.num_workers is None:
            self.num_workers = mp.cpu_count()


@dataclass
class SegmentationConfig:
    """Configuration for segmentation parameters."""
    block_size: int = 51
    min_object_size: int = 500
    adaptive_threshold: bool = True
    fill_holes: bool = True
    use_structure_tensor: bool = False  # Enable Structure Tensor enhancement
    

@dataclass
class FiberAnalysisConfig:
    """Configuration for fiber analysis parameters."""
    min_diameter: float = 10.0  # micrometers
    max_diameter: float = 50.0  # micrometers
    voxel_size: float = 1.1  # micrometers
    angle_bins: List[float] = field(default_factory=lambda: list(range(0, 181, 15)))
    calculate_tortuosity: bool = True
    calculate_orientation: bool = True
    orientation_method: str = 'structure_tensor'  # 'pca' or 'structure_tensor'


@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters."""
    generate_heatmap: bool = True
    generate_histogram: bool = True
    generate_3d_visualization: bool = True
    use_mayavi: bool = False  # Default to False due to installation issues
    use_plotly: bool = True  # Modern alternative
    colormap: str = 'viridis'
    figure_dpi: int = 150
    save_format: str = 'png'


@dataclass
class Config:
    """Main configuration class containing all sub-configurations."""
    data_dir: str
    output_dir: str
    num_images: Optional[int] = None
    sort_order: str = 'Name (Ascending)'
    log_level: str = 'INFO'
    
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    analysis: FiberAnalysisConfig = field(default_factory=FiberAnalysisConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """Load configuration from a JSON or YAML file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                data = yaml.safe_load(f)
            elif config_path.endswith('.json'):
                data = json.load(f)
            else:
                raise ValueError("Configuration file must be .json, .yaml, or .yml")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Config':
        """Create configuration from a dictionary."""
        # Extract main configuration
        main_config = {
            'data_dir': data.get('data_dir'),
            'output_dir': data.get('output_dir'),
            'num_images': data.get('num_images'),
            'sort_order': data.get('sort_order', 'Name (Ascending)'),
            'log_level': data.get('log_level', 'INFO')
        }
        
        # Extract sub-configurations
        if 'processing' in data:
            main_config['processing'] = ProcessingConfig(**data['processing'])
        if 'segmentation' in data:
            main_config['segmentation'] = SegmentationConfig(**data['segmentation'])
        if 'analysis' in data:
            main_config['analysis'] = FiberAnalysisConfig(**data['analysis'])
        if 'visualization' in data:
            main_config['visualization'] = VisualizationConfig(**data['visualization'])
        
        return cls(**main_config)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'data_dir': self.data_dir,
            'output_dir': self.output_dir,
            'num_images': self.num_images,
            'sort_order': self.sort_order,
            'log_level': self.log_level,
            'processing': asdict(self.processing),
            'segmentation': asdict(self.segmentation),
            'analysis': asdict(self.analysis),
            'visualization': asdict(self.visualization)
        }
    
    def save(self, config_path: str):
        """Save configuration to a file."""
        data = self.to_dict()
        
        with open(config_path, 'w') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                yaml.dump(data, f, default_flow_style=False)
            elif config_path.endswith('.json'):
                json.dump(data, f, indent=2)
            else:
                raise ValueError("Configuration file must be .json, .yaml, or .yml")
        
        logger.info(f"Configuration saved to {config_path}")
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        errors = []
        
        # Check required paths
        if not self.data_dir:
            errors.append("data_dir is required")
        elif not os.path.exists(self.data_dir):
            errors.append(f"data_dir does not exist: {self.data_dir}")
        
        if not self.output_dir:
            errors.append("output_dir is required")
        
        # Check numeric parameters
        if self.processing.chunk_size <= 0:
            errors.append("chunk_size must be positive")
        
        if self.processing.scale_factor < 1:
            errors.append("scale_factor must be at least 1")
        
        if self.analysis.min_diameter <= 0:
            errors.append("min_diameter must be positive")
        
        if self.analysis.max_diameter <= self.analysis.min_diameter:
            errors.append("max_diameter must be greater than min_diameter")
        
        if self.analysis.voxel_size <= 0:
            errors.append("voxel_size must be positive")
        
        if errors:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            return False
        
        return True
