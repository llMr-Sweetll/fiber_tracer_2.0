"""
Core module for fiber tracer - orchestrates the entire pipeline.
"""

import os
import logging
import sys
from typing import Optional, List, Dict, Any
import numpy as np
import json
from datetime import datetime

from .config import Config
from .preprocessing import (
    ImagePreprocessor, 
    VolumeBuilder, 
    get_sorted_tiff_files,
    validate_images
)
from .segmentation import (
    FiberSegmenter,
    AdvancedSegmenter,
    track_fibers_across_slices
)
from .analysis import (
    FiberAnalyzer,
    FiberConnectivityAnalyzer,
    FiberProperties
)
from .visualization import FiberVisualizer
from .utils import (
    suppress_warnings,
    setup_matplotlib_backend,
    check_dependencies,
    ProgressLogger,
    format_time,
    format_bytes
)
from .ascii_art import animate_startup, show_completion

logger = logging.getLogger(__name__)

# Suppress harmless warnings
suppress_warnings()
setup_matplotlib_backend()


class FiberTracer:
    """Main class orchestrating the fiber tracing pipeline."""
    
    def __init__(self, config: Config):
        """
        Initialize FiberTracer with configuration.
        
        Args:
            config: Config object with all parameters
        """
        self.config = config
        
        # Initialize components
        self.preprocessor = ImagePreprocessor(config.processing)
        self.segmenter = FiberSegmenter(config.segmentation)
        self.analyzer = FiberAnalyzer(config.analysis)
        self.visualizer = FiberVisualizer(config.visualization)
        
        # Data containers
        self.tiff_files: List[str] = []
        self.volume: Optional[np.ndarray] = None
        self.binary_volume: Optional[np.ndarray] = None
        self.labeled_volume: Optional[np.ndarray] = None
        self.fibers: List[FiberProperties] = []
        self.statistics: Dict[str, Any] = {}
        
        # Setup output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = os.path.join(
            self.config.output_dir, 
            f'fiber_tracer_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger.info("=" * 60)
        logger.info("Fiber Tracer Started")
        logger.info("=" * 60)
        logger.info(f"Output directory: {self.config.output_dir}")
        logger.info(f"Log file: {log_file}")
    
    def run(self) -> bool:
        """
        Run the complete fiber tracing pipeline.
        
        Returns:
            Success status
        """
        try:
            # Validate configuration
            if not self.config.validate():
                logger.error("Configuration validation failed")
                return False
            
            # Save configuration
            config_path = os.path.join(self.config.output_dir, 'config.json')
            self.config.save(config_path)
            
            # Execute pipeline steps
            logger.info("Starting fiber tracing pipeline")
            
            # Step 1: Load and validate images
            if not self._load_images():
                return False
            
            # Step 2: Build 3D volume
            if not self._build_volume():
                return False
            
            # Step 3: Segment fibers
            if not self._segment_fibers():
                return False
            
            # Step 4: Label connected components
            if not self._label_fibers():
                return False
            
            # Step 5: Analyze fibers
            if not self._analyze_fibers():
                return False
            
            # Step 6: Generate visualizations
            if not self._generate_visualizations():
                return False
            
            # Step 7: Save results
            if not self._save_results():
                return False
            
            logger.info("=" * 60)
            logger.info("Fiber Tracer Completed Successfully")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"Fatal error in pipeline: {e}", exc_info=True)
            return False
    
    def _load_images(self) -> bool:
        """Load and validate TIFF images."""
        try:
            logger.info("Loading TIFF images")
            
            # Get sorted file list
            self.tiff_files = get_sorted_tiff_files(
                self.config.data_dir,
                self.config.sort_order,
                self.config.num_images
            )
            
            # Validate images
            valid, errors = validate_images(self.tiff_files[:min(5, len(self.tiff_files))])
            
            if not valid:
                for error in errors:
                    logger.error(error)
                return False
            
            logger.info(f"Found {len(self.tiff_files)} valid TIFF images")
            return True
            
        except Exception as e:
            logger.error(f"Error loading images: {e}")
            return False
    
    def _build_volume(self) -> bool:
        """Build 3D volume from images."""
        try:
            logger.info("Building 3D volume from images")
            
            # Create volume builder
            builder = VolumeBuilder(self.config.output_dir)
            
            # Build volume
            self.volume, volume_shape = builder.build_volume_chunked(
                self.tiff_files,
                self.preprocessor,
                self.config.processing.chunk_size,
                self.config.processing.num_workers
            )
            
            logger.info(f"Volume shape: {volume_shape}")
            logger.info(f"Volume size: {np.prod(volume_shape) * 4 / (1024**3):.2f} GB")
            
            # Cleanup temporary files
            builder.cleanup()
            
            return True
            
        except Exception as e:
            logger.error(f"Error building volume: {e}")
            return False
    
    def _segment_fibers(self) -> bool:
        """Segment fibers in the volume."""
        try:
            logger.info("Segmenting fibers")
            
            # Choose segmentation method
            method = 'adaptive'  # Can be made configurable
            
            # Perform segmentation
            self.binary_volume = self.segmenter.segment_volume(
                self.volume, 
                method=method
            )
            
            # Optional: Apply advanced segmentation
            if hasattr(self.config.segmentation, 'use_advanced') and self.config.segmentation.use_advanced:
                advanced_segmenter = AdvancedSegmenter(self.config.segmentation)
                self.binary_volume = advanced_segmenter.segment_multiscale(self.volume)
            
            # Calculate segmentation statistics
            fiber_pixels = np.sum(self.binary_volume)
            total_pixels = self.binary_volume.size
            
            logger.info(f"Segmentation complete")
            logger.info(f"Fiber pixels: {fiber_pixels:,} ({100*fiber_pixels/total_pixels:.2f}%)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during segmentation: {e}")
            return False
    
    def _label_fibers(self) -> bool:
        """Label connected components as individual fibers."""
        try:
            logger.info("Labeling connected components")
            
            # Label fibers
            self.labeled_volume, num_fibers = self.segmenter.label_fibers(
                self.binary_volume,
                connectivity=26
            )
            
            # Optional: Track fibers across slices
            if hasattr(self.config.segmentation, 'track_fibers') and self.config.segmentation.track_fibers:
                self.labeled_volume = track_fibers_across_slices(
                    self.labeled_volume,
                    overlap_threshold=0.5
                )
            
            logger.info(f"Labeled {num_fibers} fibers")
            
            return True
            
        except Exception as e:
            logger.error(f"Error labeling fibers: {e}")
            return False
    
    def _analyze_fibers(self) -> bool:
        """Analyze fiber properties."""
        try:
            logger.info("Analyzing fiber properties")
            
            # Analyze fibers
            self.fibers = self.analyzer.analyze_fibers(
                self.labeled_volume,
                self.binary_volume,
                grayscale_volume=self.volume
            )
            
            if not self.fibers:
                logger.warning("No fibers found matching criteria")
                return False
            
            # Calculate statistics
            self.statistics = self.analyzer.calculate_statistics(self.fibers)
            
            # Calculate volume fraction
            volume_fraction = self.analyzer.calculate_volume_fraction(self.binary_volume)
            self.statistics['volume_fraction'] = volume_fraction
            
            # Analyze connectivity
            connectivity_analyzer = FiberConnectivityAnalyzer(self.config.analysis.voxel_size)
            connectivity_metrics = connectivity_analyzer.analyze_connectivity(
                self.labeled_volume,
                distance_threshold=20.0
            )
            self.statistics['connectivity'] = connectivity_metrics
            
            # Log summary statistics
            logger.info(f"Analysis complete: {len(self.fibers)} fibers analyzed")
            logger.info(f"Volume fraction: {volume_fraction:.2f}%")
            logger.info(f"Mean length: {self.statistics['length_stats']['mean']:.1f} μm")
            logger.info(f"Mean diameter: {self.statistics['diameter_stats']['mean']:.1f} μm")
            
            return True
            
        except Exception as e:
            logger.error(f"Error analyzing fibers: {e}")
            return False
    
    def _generate_visualizations(self) -> bool:
        """Generate all visualizations."""
        try:
            logger.info("Generating visualizations")
            
            # Generate visualizations
            self.visualizer.generate_all_visualizations(
                self.fibers,
                self.labeled_volume,
                self.config.output_dir
            )
            
            logger.info("Visualizations complete")
            return True
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            return False
    
    def _save_results(self) -> bool:
        """Save analysis results."""
        try:
            logger.info("Saving results")
            
            # Save fiber properties
            properties_path = os.path.join(self.config.output_dir, 'fiber_properties.csv')
            self.analyzer.save_results(self.fibers, properties_path, format='csv')
            
            # Save statistics
            stats_path = os.path.join(self.config.output_dir, 'statistics.json')
            with open(stats_path, 'w') as f:
                json.dump(self.statistics, f, indent=2, default=str)
            
            # Save volume fraction
            vf_path = os.path.join(self.config.output_dir, 'volume_fraction.txt')
            with open(vf_path, 'w') as f:
                f.write(f"Fiber Volume Fraction: {self.statistics['volume_fraction']:.2f}%\n")
                f.write(f"Total Fibers: {len(self.fibers)}\n")
            
            # Generate summary report
            self._generate_summary_report()
            
            logger.info(f"Results saved to {self.config.output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return False
    
    def _generate_summary_report(self):
        """Generate a text summary report."""
        report_path = os.path.join(self.config.output_dir, 'summary_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("FIBER ANALYSIS SUMMARY REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Directory: {self.config.data_dir}\n")
            f.write(f"Number of Images: {len(self.tiff_files)}\n")
            f.write(f"Voxel Size: {self.config.analysis.voxel_size} μm\n\n")
            
            f.write("-" * 40 + "\n")
            f.write("FIBER STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Fibers: {len(self.fibers)}\n")
            f.write(f"Volume Fraction: {self.statistics['volume_fraction']:.2f}%\n\n")
            
            if 'length_stats' in self.statistics:
                f.write("Length (μm):\n")
                f.write(f"  Mean: {self.statistics['length_stats']['mean']:.1f}\n")
                f.write(f"  Std: {self.statistics['length_stats']['std']:.1f}\n")
                f.write(f"  Min: {self.statistics['length_stats']['min']:.1f}\n")
                f.write(f"  Max: {self.statistics['length_stats']['max']:.1f}\n\n")
            
            if 'diameter_stats' in self.statistics:
                f.write("Diameter (μm):\n")
                f.write(f"  Mean: {self.statistics['diameter_stats']['mean']:.1f}\n")
                f.write(f"  Std: {self.statistics['diameter_stats']['std']:.1f}\n")
                f.write(f"  Min: {self.statistics['diameter_stats']['min']:.1f}\n")
                f.write(f"  Max: {self.statistics['diameter_stats']['max']:.1f}\n\n")
            
            if 'orientation_stats' in self.statistics:
                f.write("Orientation (degrees):\n")
                f.write(f"  Mean: {self.statistics['orientation_stats']['mean']:.1f}\n")
                f.write(f"  Std: {self.statistics['orientation_stats']['std']:.1f}\n\n")
            
            if 'class_distribution' in self.statistics:
                f.write("Fiber Classification:\n")
                for class_name, count in self.statistics['class_distribution'].items():
                    f.write(f"  {class_name}: {count}\n")
                f.write("\n")
            
            if 'connectivity' in self.statistics:
                conn = self.statistics['connectivity']
                f.write("Connectivity Analysis:\n")
                f.write(f"  Total Connections: {conn.get('total_connections', 0)}\n")
                f.write(f"  Avg Connections/Fiber: {conn.get('average_connections_per_fiber', 0):.1f}\n")
                f.write(f"  Isolated Fibers: {conn.get('isolated_fibers', 0)}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 60 + "\n")
        
        logger.info(f"Summary report saved to {report_path}")


def run_from_args(args) -> bool:
    """
    Run fiber tracer from command-line arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Success status
    """
    # Create configuration from arguments
    config = Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_images=args.num_images,
        sort_order=args.sort_order,
        log_level=args.log_level
    )
    
    # Update sub-configurations from arguments
    config.processing.chunk_size = args.chunk_size
    config.processing.scale_factor = args.scale_factor
    config.processing.num_workers = args.num_workers
    
    config.analysis.voxel_size = args.voxel_size
    config.analysis.min_diameter = args.min_diameter
    config.analysis.max_diameter = args.max_diameter
    
    # Create and run tracer
    tracer = FiberTracer(config)
    return tracer.run()
