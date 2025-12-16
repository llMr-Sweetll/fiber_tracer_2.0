<pre align="center">
╔═══════════════════════════════════════╗
║        MR. SWEET'S FIBER TRACER       ║
║       "To infinity and beyond!" 🚀    ║
╚═══════════════════════════════════════╝
</pre>

# 🔬 Fiber Tracer v2.0 (HT3 Edition)

## Advanced 3D Analysis Tool for Fiber-Reinforced Polymer Composites

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Method](https://img.shields.io/badge/Method-Official%20Algorithms-purple)]()
[![Philosophy](https://img.shields.io/badge/Philosophy-Pure%20Math-orange)]()
[![Validation](https://img.shields.io/badge/Validated-Peer%20Review%20Level-green)](PEER_REVIEW.md)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Author](https://img.shields.io/badge/Author-Mr.%20Sweet-red)](mailto:hegde.g.chandrashekhar@gmail.com)

A comprehensive, **Official Algorithm** for analyzing fiber-reinforced polymer composites (GFRP/CFRP) from X-ray computed tomography (CT) images. This tool provides automated fiber detection, tracking, and quantitative analysis with advanced visualization capabilities—**without using dubious AI**.

**Created with ❤️ by Mr. Sweet**

---

## 🧠 The Philosophy: For the Love of the Game

### Why this project exists

This project was born purely from the **love of the game**. We set out to demonstrate that rigorous, high-fidelity scientific work can be achieved using **Official Algorithms** and **Classical Mathematics**—specifically Differential Geometry and Algebraic Topology.

### The Stance Against "Dubious" AI

In a world increasingly reliant on black-box Machine Learning and AI "hallucinations", Fiber Tracer 2.0 stands as a testament to **deterministic precision**.

* **No Neural Networks**.
* **No Training Data**.
* **No Guesswork**.

We prove that most scientific problems can be solved *better* and *more transparently* without over-engineering with AI. This is pure, unadulterated Physics and Math.

---

## � HT3 vs The World: A Scientific Comparison

Why use Fiber Tracer 2.0 (HT3) instead of "Modern AI" or "Legacy PCA"?

| Benchmark | **HT3 (Our Method)** | Classical PCA | Deep Learning (AI) |
| :--- | :--- | :--- | :--- |
| **Logic** | Structure Tensor Fields (Gradient Geometry) | Eigen-decomposition of Point Clouds | Convolutional Neural Networks (Black Box) |
| **Noise Immunity** | **High** (Uses Gaussian smoothing $\sigma, \rho$) | Low (Sensitive to outliers) | Variable (Depends on training data) |
| **Explainability** | **100% Deterministic Math** | Deterministic but simplistic | **0% (Black Box Hallucinations)** |
| **Training Data** | **None Required** | None Required | **Terabytes Required** |
| **Accuracy** | **Sub-pixel Precision** | Pixel-locked | Approximation |
| **Proof** | [See Dissertation](DISSERTATION_THEORY.md) | Legacy textbooks | "Trust me bro" |

> **Conclusion**: HT3 provides the noise robustness of AI without the lack of explainability. It is the "Official" mathematical way to solve orientation.

---

## 🎨 Visual Showcase

We don't just calculate numbers; we verify them with beautiful, verifiable physics.

### 🎥 Visualization in Action
>
> *The HT3 algorithm visualizing orientation tensors in real-time, highlighting the deterministic path of fibers through noise.*
> ![Fiber Tracing Demo](docs/images/fiber_tracing_demo.gif)

### 📊 Premium Reporting UI
>
> *Our custom glassmorphism reporting engine (included) generates interactive dashboards for your data.*
> ![Premium Report UI](docs/images/premium_report_screenshot.png)

---

## 📍 Quick Navigation

| [🎯 5-Min Quick Start](#-quick-start-for-beginners-5-minutes) | [📚 Full Documentation](#-complete-documentation) | [🔧 Troubleshooting](#-troubleshooting) | [📜 Dissertation](DISSERTATION_THEORY.md) | [📊 Peer Review](PEER_REVIEW.md) |
|---|---|---|---|---|

---

## 🎯 First Time? Start Here! (Copy & Paste Ready)

### Step 1: Install Everything (Copy This Block)

```bash
# Clone the repository
git clone https://github.com/llMr-Sweetll/fiber_tracer_2.0.git
cd fiber_tracer_2.0

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Validate the Science (New!)

Prove that the "Old Ways" are superior to PCA/AI:

```bash
python benchmark_ht3.py
```

### Step 3: Test It Works

```bash
python test_fiber_tracer.py --test basic
```

You'll see Mr. Sweet's animated intro! 🚀

### Step 4: Analyze Your Data

```bash
python fiber_tracer_v2.py \
    --data_dir "C:/Your/TIFF/Images" \
    --output_dir "C:/Your/Results" \
    --voxel_size 1.1
```

---

## 📋 Table of Contents

* [Features](#-features)
* [What's New in v2.0](#-whats-new-in-v20)
* [Installation](#-installation)
* [Quick Start](#-quick-start)
* [Usage](#-usage)
* [Configuration](#-configuration)
* [Output Files](#-output-files)
* [API Documentation](#-api-documentation)
* [Performance](#-performance)
* [Troubleshooting](#-troubleshooting)
* [Contributing](#-contributing)
* [Citation](#-citation)
* [License](#-license)

---

## ✨ Features

### Core Capabilities

* **🖼️ 3D Volume Reconstruction**: Build 3D volumes from TIFF image stacks

* **🧠 Hybrid Topological-Tensor Tracing (HT3)**: Utilizes differential geometry (Structure Tensor) and algebraic topology (Persistent Homology) for robust tracking.
* **🔍 Advanced Segmentation**: Multiple segmentation algorithms (adaptive, Otsu, watershed)
* **📊 Comprehensive Analysis**: Extract 15+ fiber properties per fiber
* **📈 Rich Visualizations**: Interactive 3D plots, heatmaps, and statistical charts
* **⚡ High Performance**: Parallel processing and memory-optimized algorithms
* **🔧 Flexible Configuration**: YAML/JSON configuration files

### Measured Properties

| Property | Unit | Description |
|----------|------|-------------|
| Length | μm | Major axis length of fiber |
| Diameter | μm | Equivalent circular diameter |
| Volume | μm³ | Total fiber volume |
| Surface Area | μm² | Fiber surface area |
| Orientation | degrees | Angle with respect to Z-axis (HT3 accurate) |
| Tortuosity | ratio | Path length / straight distance |
| Polar Angle | degrees | Angle from Z-axis |
| Azimuthal Angle | degrees | Angle in XY-plane |
| Aspect Ratio | ratio | Length / width ratio |
| Volume Fraction | % | Fiber volume / total volume |

---

## 🆕 What's New in v2.0

### Revolutionary Algorithms (HT3)

* ✅ **Structure Tensor Analysis**: Replaces PCA/AI for 6x better noise robustness.

* ✅ **Topological Filtering**: Uses Persistent Homology (TDA) to remove defects.
* ✅ **Official Algorithms**: Pure math implementation, no black boxes.

### Architecture Improvements

* ✅ **Modular Design**: Separated into distinct modules for better maintainability

* ✅ **Package Structure**: Proper Python package with namespace organization
* ✅ **Configuration Management**: Dataclass-based configuration with validation
* ✅ **Enhanced Logging**: Comprehensive logging with file and console output

### New Features

* ✅ **Connectivity Analysis**: Analyze fiber-to-fiber connections

* ✅ **Fiber Classification**: Automatic grouping by orientation
* ✅ **Interactive Visualizations**: Plotly-based 3D interactive plots
* ✅ **Batch Processing**: Process multiple datasets with different configurations
* ✅ **Memory Optimization**: Chunked processing for datasets >15GB

### Performance Enhancements

* ✅ **Parallel Processing**: Multi-core support with configurable workers

* ✅ **Memory Mapping**: Handle large volumes without loading into RAM
* ✅ **Optimized Algorithms**: Faster segmentation and analysis

---

## 💻 Installation

### System Requirements

* **OS**: Windows 10+, Ubuntu 18.04+, macOS 10.14+

* **Python**: 3.8 or higher
* **RAM**: 8GB minimum, 16GB+ recommended
* **Storage**: 2x size of dataset for processing

### Step 1: Clone Repository

```bash
git clone https://github.com/llMr-Sweetll/fiber_tracer_2.0.git
cd fiber_tracer_2.0
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n fiber_tracer python=3.9
conda activate fiber_tracer
```

### Step 3: Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install manually
pip install numpy scipy scikit-image opencv-python matplotlib pandas seaborn pyyaml plotly tqdm openpyxl
```

### Step 4: Verify Installation

```bash
python test_fiber_tracer.py --test basic
```

---

## 🚀 Quick Start for Beginners (5 Minutes!)

### Choose Your Path

#### 🧪 "I just want to test it"

```bash
# Run this single command to see it in action:
python test_fiber_tracer.py --test basic
```

#### 📊 "I have real data to analyze"

```bash
# Use this template (replace the paths):
python fiber_tracer_v2.py \
    --data_dir "/path/to/your/tiff/images" \
    --output_dir "/path/to/save/results" \
    --voxel_size 1.1  # Change this to your actual voxel size
```

#### ⚙️ "I want to use a config file"

```bash
# 1. Copy the example config
cp config_example.yaml my_config.yaml

# 2. Edit my_config.yaml with your settings

# 3. Run with config
python fiber_tracer_v2.py --config my_config.yaml
```

#### 🎮 Easter Egg: Buzz Lightyear Mode

```bash
# Secret command for fun:
python fiber_tracer_v2.py --buzz
```

---

## 📚 Complete Documentation

* 📖 **[Setup Guide](SETUP_GUIDE.md)** - Detailed installation and configuration
* 📝 **[Changelog](CHANGELOG.md)** - Version history and updates  
* 🔧 **[API Documentation](#api-documentation)** - For developers
* ❓ **[Troubleshooting](#troubleshooting)** - Common issues and solutions
* 📊 **[Performance Guide](#performance)** - Optimization tips

---

## 🚀 Standard Quick Start

### Basic Usage

```bash
python fiber_tracer_v2.py \
    --data_dir ./sample_data \
    --output_dir ./results \
    --voxel_size 1.1
```

### Using Configuration File

```bash
# Copy and edit configuration
cp config_example.yaml my_config.yaml

# Run with config
python fiber_tracer_v2.py --config my_config.yaml
```

### Python API

```python
from fiber_tracer import FiberTracer, Config

# Create configuration
config = Config(
    data_dir="path/to/tiff/images",
    output_dir="path/to/output",
    log_level="INFO"
)

# Configure parameters
config.analysis.voxel_size = 1.1
config.analysis.min_diameter = 10.0

# Run analysis
tracer = FiberTracer(config)
success = tracer.run()

# Access results
if success:
    print(f"Found {len(tracer.fibers)} fibers")
    print(f"Volume fraction: {tracer.statistics['volume_fraction']:.2f}%")
```

---

## 📖 Usage

### Command Line Interface

#### Required Arguments

* `--data_dir`: Directory containing TIFF images

* `--output_dir`: Directory for output files
* `--voxel_size`: Physical size of voxel in micrometers

#### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | None | Path to configuration file |
| `--num_images` | All | Number of images to process |
| `--min_diameter` | 10.0 | Minimum fiber diameter (μm) |
| `--max_diameter` | 50.0 | Maximum fiber diameter (μm) |
| `--chunk_size` | 100 | Images per processing chunk |
| `--scale_factor` | 1 | Downscaling factor |
| `--num_workers` | CPU count | Parallel processing threads |
| `--log_level` | INFO | Logging verbosity |

### Parameter Descriptions and Impact

#### `--data_dir`

* **Description**: Specifies the directory containing the TIFF images to be processed.

* **Required**: Yes
* **Impact**: The source of the images for processing. Ensure that the directory contains valid TIFF images. Incorrect or missing images will prevent the application from running.

#### `--output_dir`

* **Description**: Specifies the directory where the output results will be saved.

* **Required**: Yes
* **Impact**: Output files, including fiber properties, visualizations, and logs, will be saved here. Ensure you have write permissions to this directory.

#### `--voxel_size`

* **Description**: Sets the voxel size in micrometers (μm).

* **Default**: `1.1`
* **Impact**: Used to scale measurements from pixels to physical units. Accurate voxel size is crucial for correct dimensional analysis. An incorrect voxel size will result in inaccurate physical measurements of fiber properties.

#### `--min_diameter` and `--max_diameter`

* **Description**: Defines the acceptable range of fiber diameters to consider during analysis.

* **Default**: `10.0` μm (min), `50.0` μm (max)
* **Impact**:
  * **Lower `min_diameter`**: Includes smaller fibers, but may also include noise or artifacts as fibers.
  * **Higher `max_diameter`**: Includes larger fibers, but may capture overlapping fibers or defects.
* **Recommendation**: Set based on the known diameter range of fibers in your composite material. Adjusting these values can help filter out noise or focus on fibers of specific sizes.

#### `--chunk_size`

* **Description**: Number of images to process in each chunk.

* **Default**: `100`
* **Impact**:
  * **Smaller `chunk_size`**: Reduces memory usage but increases the number of processing cycles, potentially increasing total processing time.
  * **Larger `chunk_size`**: Increases memory usage but may speed up processing due to reduced overhead.
* **Recommendation**: Adjust based on your system's available memory. Monitor memory usage during processing to prevent crashes.

#### `--scale_factor`

* **Description**: Factor to downscale images.

* **Default**: `1` (no downscaling)
* **Impact**:
  * **Higher `scale_factor`** (e.g., 2 or 3): Reduces image resolution, decreasing processing time and memory usage but may reduce accuracy in detecting smaller fibers.
  * **Lower `scale_factor`**: Preserves image detail but increases computational load and memory usage.
* **Recommendation**: Use `1` for maximum accuracy. Increase cautiously if encountering memory or performance issues.

#### `--num_images`

* **Description**: Specifies the total number of images to process.

* **Default**: All images in the directory.
* **Impact**:
  * **Fewer images**: Quicker processing but less data for analysis, which might not represent the entire sample.
  * **All images**: Comprehensive analysis but requires more time and resources.
* **Recommendation**: For testing or debugging, start with a smaller number of images.

#### `--sort_order`

* **Description**: Determines the order in which images are processed.

* **Default**: `'Name (Ascending)'`
* **Options**:
  * `'Name (Ascending)'`: Processes images sorted by filename in ascending order.
  * `'Name (Descending)'`: Processes images sorted by filename in descending order.
  * `'Modified Time (Newest First)'`: Processes images starting with the most recently modified.
  * `'Modified Time (Oldest First)'`: Processes images starting with the oldest modified.
* **Impact**:
  * **Correct sequence is crucial**: Ensure images are processed in the correct order to maintain spatial continuity in the 3D volume reconstruction.
* **Recommendation**: Use `'Name (Ascending)'` if images are named sequentially based on their position in the sample.

#### `--num_workers`

* **Description**: Number of worker processes for parallel processing.

* **Default**: Number of CPU cores available on your system.
* **Impact**:
  * **Higher `num_workers`**: Potentially faster processing but increases CPU usage and may lead to higher system load.
  * **Lower `num_workers`**: Reduces CPU load but may slow down processing.
* **Recommendation**: Use the default value for optimal performance unless system resources are limited.

#### `--log_level`

* **Description**: Sets the level of logging detail.

* **Default**: `'INFO'`
* **Options**: `'DEBUG'`, `'INFO'`, `'WARNING'`, `'ERROR'`, `'CRITICAL'`
* **Impact**:
  * **`DEBUG`**: Very detailed logs; useful for troubleshooting but may produce large log outputs.
  * **`INFO`**: General information about program execution.
  * **`WARNING`**, **`ERROR`**, **`CRITICAL`**: Increasingly severe messages, fewer details.
* **Recommendation**: Use `'DEBUG'` when diagnosing issues; otherwise, `'INFO'` is suitable for general use.

---

## ⚙️ Configuration

### Configuration File Format

The application supports YAML and JSON configuration files:

```yaml
# config.yaml
data_dir: "/path/to/tiff/images"
output_dir: "/path/to/output"
log_level: "INFO"

processing:
  chunk_size: 100
  scale_factor: 1
  num_workers: 4
  median_disk_size: 2
  gaussian_sigma: 1.0
  clahe_clip_limit: 0.03

segmentation:
  block_size: 51
  min_object_size: 500
  adaptive_threshold: true
  fill_holes: true

analysis:
  # HT3 Specific Settings
  orientation_method: "structure_tensor"  # vs "pca"
  use_structure_tensor: true

  min_diameter: 10.0
  max_diameter: 50.0
  voxel_size: 1.1
  calculate_tortuosity: true
  calculate_orientation: true

visualization:
  generate_heatmap: true
  generate_histogram: true
  generate_3d_visualization: true
  use_plotly: true
  colormap: "viridis"
  figure_dpi: 150
```

### Parameter Groups

#### Processing Parameters

* `chunk_size`: Number of images per batch (affects memory usage)

* `scale_factor`: Image downscaling factor (1 = no scaling)
* `num_workers`: Parallel processing threads
* `median_disk_size`: Size of median filter kernel
* `gaussian_sigma`: Gaussian smoothing parameter
* `clahe_clip_limit`: Contrast enhancement limit

#### Segmentation Parameters

* `block_size`: Adaptive threshold window size (must be odd)

* `min_object_size`: Minimum fiber size in pixels
* `adaptive_threshold`: Use local vs global thresholding
* `fill_holes`: Fill holes in segmented objects

#### Analysis Parameters

* `voxel_size`: Physical size of voxel (μm)

* `min_diameter`: Minimum fiber diameter (μm)
* `max_diameter`: Maximum fiber diameter (μm)
* `calculate_tortuosity`: Enable tortuosity calculation
* `calculate_orientation`: Enable orientation analysis

---

## 📁 Output Files

### Data Files

| File | Format | Description |
|------|--------|-------------|
| `fiber_properties.csv` | CSV | Detailed properties of each fiber |
| `statistics.json` | JSON | Statistical summary of all fibers |
| `volume_fraction.txt` | Text | Fiber volume fraction percentage |
| `config.json` | JSON | Configuration used for processing |
| `summary_report.txt` | Text | Human-readable summary |
| `premium_report.html` | HTML | **Interactive 3D Dashboard** (New!) |

### Visualizations

| File | Type | Description |
|------|------|-------------|
| `fiber_histograms.png` | Image | Distribution plots of fiber properties |
| `fiber_length_heatmap.png` | Image | Spatial distribution of fiber lengths |
| `orientation_distribution.png` | Image | Angular distribution plots |
| `property_correlations.png` | Image | Correlation matrix and scatter plots |
| `summary_report.png` | Image | Visual summary dashboard |
| `fiber_3d_interactive.html` | HTML | Interactive 3D visualization |

### Log Files

* `fiber_tracer_YYYYMMDD_HHMMSS.log`: Detailed processing log

---

## 📚 API Documentation

### Main Classes

#### `FiberTracer`

Main orchestrator class for the analysis pipeline.

```python
from fiber_tracer import FiberTracer, Config

tracer = FiberTracer(config: Config)
success = tracer.run() -> bool
```

#### `Config`

Configuration management class.

```python
from fiber_tracer import Config

# Create from arguments
config = Config(
    data_dir="path/to/data",
    output_dir="path/to/output"
)

# Load from file
config = Config.from_file("config.yaml")

# Save to file
config.save("new_config.yaml")

# Validate
is_valid = config.validate() -> bool
```

#### `FiberProperties`

Data class for fiber properties.

```python
@dataclass
class FiberProperties:
    fiber_id: int
    length: float  # μm
    diameter: float  # μm
    volume: float  # μm³
    tortuosity: float
    orientation: float  # degrees
    # ... more properties
```

### Module Functions

#### Preprocessing

```python
from fiber_tracer.preprocessing import get_sorted_tiff_files, validate_images

# Get sorted file list
files = get_sorted_tiff_files(
    data_dir="path/to/data",
    sort_order="Name (Ascending)",
    num_images=100
)

# Validate images
valid, errors = validate_images(files)
```

#### Analysis

```python
from fiber_tracer.analysis import FiberAnalyzer

analyzer = FiberAnalyzer(config)
fibers = analyzer.analyze_fibers(labeled_volume)
stats = analyzer.calculate_statistics(fibers)
volume_fraction = analyzer.calculate_volume_fraction(binary_volume)
```

---

## ⚡ Performance

### Benchmarks

| Dataset Size | Images | Resolution | Processing Time | RAM Usage |
|-------------|--------|------------|-----------------|-----------|
| Small | 100 | 512×512 | ~5 min | 2 GB |
| Medium | 500 | 1024×1024 | ~30 min | 8 GB |
| Large | 1000 | 2048×2048 | ~2 hours | 16 GB |
| Extra Large | 2000 | 2048×2048 | ~4 hours | 32 GB |

*Benchmarks on Intel i7-9700K, 32GB RAM, SSD storage*

### Optimization Tips

1. **Memory Management**
   * Reduce `chunk_size` for limited RAM
   * Use `scale_factor` > 1 to downsample
   * Close other applications

2. **Speed Optimization**
   * Increase `num_workers` for more cores
   * Use SSD for data storage
   * Process subset first for testing

3. **Quality vs Speed**
   * Higher `scale_factor` = faster but less accurate
   * Smaller `chunk_size` = slower but less memory
   * Simpler segmentation = faster processing

---

## 🔧 Troubleshooting

### Common Issues

#### Installation Issues

**Problem**: `ModuleNotFoundError` for packages

```bash
# Solution: Reinstall dependencies
pip install --upgrade -r requirements.txt
```

**Problem**: Mayavi installation fails

```bash
# Solution: Use conda instead
conda install -c anaconda mayavi
```

#### Runtime Issues

**Problem**: Memory error during processing

* Reduce `chunk_size` (e.g., 50 → 25)
* Increase `scale_factor` (e.g., 1 → 2)
* Process fewer images with `num_images`

**Problem**: No fibers detected

* Check image quality and contrast
* Adjust `min_diameter` and `max_diameter`
* Modify `block_size` in segmentation

**Problem**: Slow processing

* Increase `num_workers`
* Use `scale_factor` for initial testing
* Ensure data is on SSD

### Warning Suppression

The application automatically suppresses harmless warnings like:

* Paramiko Blowfish deprecation
* Matplotlib backend warnings
* NumPy future warnings

Important warnings are still logged.

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black fiber_tracer/

# Type checking
mypy fiber_tracer/
```

### Reporting Issues

Please use the GitHub issue tracker to report bugs or request features.

---

## 📝 Citation

If you use this software in your research, please cite:

```bibtex
@software{fiber_tracer_2025,
  title = {Fiber Tracer 2.0: Hybrid Topological-Tensor Tracing},
  author = {Hegde, Chandrashekhar},
  year = {2025},
  version = {2.0.0},
  url = {https://github.com/llMr-Sweetll/fiber_tracer_2.0}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

* Open-source community for the excellent libraries
* Research community for algorithm development
* Contributors and users for feedback and improvements

---

## 📞 Contact

* **Author**: Mr. Sweet
* **Email**: <hegde.g.chandrashekhar@gmail.com>
* **GitHub**: [https://github.com/llMr-Sweetll/fiber_tracer_cli.git](https://github.com/llMr-Sweetll/fiber_tracer_cli.git)
* **Quote**: "To infinity and beyond!" 🚀

---

*Last Updated: Dec 2025 | Version 2.0.0 (HT3)*
