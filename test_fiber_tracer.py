#!/usr/bin/env python3
"""
Test script for the Fiber Tracer application.
This script creates synthetic test data and runs the fiber tracer pipeline.

Author: Mr. Sweet
Contact: hegde.g.chandrashekhar@gmail.com
"""

import os
import sys
import numpy as np
from pathlib import Path
import argparse
from skimage import io, draw, filters
import logging

# Add the parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from fiber_tracer import FiberTracer, Config
from fiber_tracer.config import (
    ProcessingConfig, 
    SegmentationConfig, 
    FiberAnalysisConfig, 
    VisualizationConfig
)
from fiber_tracer.ascii_art import animate_startup, show_completion

logger = logging.getLogger(__name__)


def create_synthetic_fiber_data(output_dir: str, num_slices: int = 50, 
                               image_size: tuple = (256, 256),
                               num_fibers: int = 20):
    """
    Create synthetic TIFF images with fiber-like structures for testing.
    
    Args:
        output_dir: Directory to save test images
        num_slices: Number of image slices to create
        image_size: Size of each image (height, width)
        num_fibers: Approximate number of fibers to generate
    """
    print(f"Creating synthetic fiber data in {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize 3D volume
    volume = np.zeros((num_slices, *image_size), dtype=np.float32)
    
    # Generate random fiber paths
    for fiber_id in range(num_fibers):
        # Random fiber parameters
        start_x = np.random.randint(10, image_size[1] - 10)
        start_y = np.random.randint(10, image_size[0] - 10)
        start_z = np.random.randint(0, num_slices // 3)
        
        # Fiber direction (with some randomness)
        direction_x = np.random.randn() * 0.3
        direction_y = np.random.randn() * 0.3
        direction_z = 1.0 + np.random.randn() * 0.2
        
        # Fiber diameter
        radius = np.random.uniform(2, 5)
        
        # Fiber length
        length = np.random.randint(num_slices // 2, num_slices)
        
        # Generate fiber path
        current_x = start_x
        current_y = start_y
        
        for z in range(start_z, min(start_z + length, num_slices)):
            # Add some tortuosity
            current_x += direction_x + np.random.randn() * 0.5
            current_y += direction_y + np.random.randn() * 0.5
            
            # Keep within bounds
            current_x = np.clip(current_x, radius, image_size[1] - radius)
            current_y = np.clip(current_y, radius, image_size[0] - radius)
            
            # Draw circular cross-section
            rr, cc = draw.disk((int(current_y), int(current_x)), radius, shape=image_size)
            
            # Add intensity variation
            intensity = np.random.uniform(0.8, 1.0)
            volume[z, rr, cc] = np.maximum(volume[z, rr, cc], intensity)
    
    # Add noise and background
    for z in range(num_slices):
        # Add minimal noise
        noise = np.random.randn(*image_size) * 0.01
        volume[z] += noise
        
        # Add minimal background intensity
        volume[z] += 0.01
        
        # Clip values
        volume[z] = np.clip(volume[z], 0, 1)
        
        # Convert to uint16 for saving
        img_uint16 = (volume[z] * 65535).astype(np.uint16)
        
        # Save as TIFF
        filename = os.path.join(output_dir, f'fiber_slice_{z:04d}.tif')
        io.imsave(filename, img_uint16)
    
    print(f"Created {num_slices} synthetic TIFF images")
    return output_dir


def test_basic_pipeline():
    """Test the basic fiber tracer pipeline with synthetic data."""
    # Show test mode animation
    animate_startup(test_mode=True)
    
    print("\n" + "="*60)
    print("FIBER TRACER TEST - Basic Pipeline")
    print("="*60)
    
    # Create test directories
    test_dir = Path("test_run")
    data_dir = test_dir / "test_data"
    output_dir = test_dir / "test_output"
    
    # Create synthetic data
    create_synthetic_fiber_data(
        str(data_dir),
        num_slices=30,
        image_size=(128, 128),
        num_fibers=10
    )
    
    # Create configuration
    config = Config(
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        log_level='INFO'
    )
    
    # Adjust parameters for test data
    config.processing.chunk_size = 10
    config.processing.scale_factor = 1
    config.processing.median_disk_size = 1
    config.processing.gaussian_sigma = 0.5
    
    config.segmentation.block_size = 21
    config.segmentation.min_object_size = 10
    config.segmentation.adaptive_threshold = False  # Use Otsu for clean synthetic data
    
    config.analysis.min_diameter = 2.0
    config.analysis.max_diameter = 30.0
    config.analysis.voxel_size = 1.0
    
    config.visualization.use_mayavi = False  # Avoid Mayavi dependency
    config.visualization.use_plotly = True
    
    # Run fiber tracer
    print("\nRunning fiber tracer...")
    tracer = FiberTracer(config)
    success = tracer.run()
    
    if success:
        print("\n✓ Test completed successfully!")
        print(f"Results saved to: {output_dir}")
        
        # Check output files
        expected_files = [
            'fiber_properties.csv',
            'statistics.json',
            'volume_fraction.txt',
            'summary_report.txt',
            'config.json'
        ]
        
        print("\nChecking output files:")
        for filename in expected_files:
            filepath = output_dir / filename
            if filepath.exists():
                print(f"  ✓ {filename}")
            else:
                print(f"  ✗ {filename} (missing)")
        
        # Print summary statistics
        if tracer.fibers:
            print(f"\nSummary:")
            print(f"  Total fibers detected: {len(tracer.fibers)}")
            if tracer.statistics:
                print(f"  Volume fraction: {tracer.statistics.get('volume_fraction', 0):.2f}%")
    else:
        print("\n✗ Test failed!")
    
    return success


def test_with_config_file():
    """Test using a configuration file."""
    print("\n" + "="*60)
    print("FIBER TRACER TEST - Configuration File")
    print("="*60)
    
    # Create test directories
    test_dir = Path("test_run_config")
    data_dir = test_dir / "test_data"
    output_dir = test_dir / "test_output"
    
    # Create synthetic data
    create_synthetic_fiber_data(
        str(data_dir),
        num_slices=20,
        image_size=(100, 100),
        num_fibers=8
    )
    
    # Create configuration file
    config_file = test_dir / "test_config.yaml"
    config_content = f"""
# Test configuration for fiber tracer
data_dir: "{str(data_dir).replace(os.sep, '/')}"
output_dir: "{str(output_dir).replace(os.sep, '/')}"
num_images: 20
sort_order: "Name (Ascending)"
log_level: "INFO"

processing:
  chunk_size: 10
  scale_factor: 1
  median_disk_size: 1
  gaussian_sigma: 0.5
  clahe_clip_limit: 0.03

segmentation:
  block_size: 21
  min_object_size: 30
  adaptive_threshold: true
  fill_holes: true

analysis:
  min_diameter: 5.0
  max_diameter: 25.0
  voxel_size: 1.0
  calculate_tortuosity: true
  calculate_orientation: true

visualization:
  generate_heatmap: true
  generate_histogram: true
  generate_3d_visualization: true
  use_mayavi: false
  use_plotly: true
  colormap: "viridis"
  figure_dpi: 100
"""
    
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"Created config file: {config_file}")
    
    # Load and run with config file
    config = Config.from_file(str(config_file))
    
    print("\nRunning fiber tracer with config file...")
    tracer = FiberTracer(config)
    success = tracer.run()
    
    if success:
        print("\n✓ Config file test completed successfully!")
    else:
        print("\n✗ Config file test failed!")
    
    return success


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("RUNNING ALL FIBER TRACER TESTS")
    print("="*60)
    
    results = []
    
    # Test 1: Basic pipeline
    try:
        results.append(("Basic Pipeline", test_basic_pipeline()))
    except Exception as e:
        print(f"Basic pipeline test failed with error: {e}")
        results.append(("Basic Pipeline", False))
    
    # Test 2: Config file
    try:
        results.append(("Config File", test_with_config_file()))
    except Exception as e:
        print(f"Config file test failed with error: {e}")
        results.append(("Config File", False))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
    
    return all_passed


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Test the Fiber Tracer application')
    parser.add_argument('--test', type=str, default='all',
                       choices=['all', 'basic', 'config'],
                       help='Which test to run')
    
    args = parser.parse_args()
    
    if args.test == 'all':
        success = run_all_tests()
    elif args.test == 'basic':
        success = test_basic_pipeline()
    elif args.test == 'config':
        success = test_with_config_file()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
