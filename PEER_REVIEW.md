# Peer Review & Verification Package

**Author:** Chandrashekhar Hegde
**Date:** December 2025

This directory contains the necessary tools to verify the "Postdoctoral Grade" quality of the HT3 implementation.

## 1. Quantitative Benchmark (`benchmark_ht3.py`)

This script runs a rigorous comparison between the Legacy PCA method and the new HT3 Structure Tensor method using synthetic data.

* **Location**: Root directory (`/benchmark_ht3.py`)
* **Run**: `python benchmark_ht3.py`
* **Result**: A markdown table showing angular error and improvement metrics.

## 2. Premium Report Generator (`fiber_tracer/premium_report_generator.py`)

This module generates high-quality HTML5 reports with "glassmorphism" UI, status chips, and embedded visualizations.

* **Location**: `fiber_tracer/premium_report_generator.py`
* **Usage**: It is designed to be imported by the core pipeline, but you can inspect the code to see the UI generation logic.

## 3. Research Documentation

See `research_theory.md` (in artifacts) for the mathematical proofs.
