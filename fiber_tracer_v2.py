#!/usr/bin/env python3
"""
Fiber Tracing CLI - Robust Version
"""

import argparse
import sys
import os
from pathlib import Path

# Add the parent directory to path to import the package
sys.path.insert(0, str(Path(__file__).parent))

try:
    from fiber_tracer import FiberTracer, Config
    from fiber_tracer.ascii_art import animate_startup, show_completion
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import fiber_tracer package: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Fiber Tracer V2 CLI")
    
    parser.add_argument('--data_dir', required=True, help="Input directory")
    parser.add_argument('--output_dir', required=True, help="Output directory")
    parser.add_argument('--voxel_size', type=float, default=1.1, help="Voxel size")
    parser.add_argument('--log_level', default="INFO", help="Log level")
    parser.add_argument('--config', help="Config file path")
    
    args = parser.parse_args()
    
    animate_startup()
    
    try:
        if args.config:
            config = Config.from_file(args.config)
            # Override if provided
            if args.data_dir: config.data_dir = args.data_dir
            if args.output_dir: config.output_dir = args.output_dir
        else:
            config = Config(
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                log_level=args.log_level
            )
            config.analysis.voxel_size = args.voxel_size
            
        print(f"Starting analysis on: {config.data_dir}")
        print(f"Output directory: {config.output_dir}")
        
        tracer = FiberTracer(config)
        success = tracer.run()
        
        show_completion(success)
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"Fatal Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
