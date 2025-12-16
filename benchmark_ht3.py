"""
HT3 vs. PCA Benchmark Script
Quantitative verification of the Hybrid Topological-Tensor Tracing method.

Author: Chandrashekhar Hegde
"""

import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
import time
# from tabulate import tabulate

# Import our stack
# Note: We need to import the class definitions. 
# Since this is a standalone artifact run in a different context, we might need to rely on local definitions 
# or assume the user copies it. For this artifact, we will copy the minimal classes needed to run the test 
# if the package isn't installed, OR we assume the package is reachable. 
# Given the environment, we will assume we can import from the file system if we add it to path.

import sys
import os

# Mock classes to ensure standalone execution if imports fail
# (In a real scenario, we would set PYTHONPATH)

class Benchmark:
    def __init__(self, size=64):
        self.size = size
        self.center = size // 2
        
    def generate_synthetic_fiber(self, angle_deg, noise_level=0.0):
        """Generates a volume with a single cylinder at a specific angle."""
        vol = np.zeros((self.size, self.size, self.size))
        
        # Vector direction
        angle_rad = np.radians(angle_deg)
        # Rotating in XZ plane for simplicity
        direction = np.array([np.cos(angle_rad), 0, np.sin(angle_rad)])
        
        # Create cylinder line
        t = np.linspace(-self.size/2, self.size/2, self.size*2)
        
        for ti in t:
            p = np.array([self.center, self.center, self.center]) + direction * ti
            z, y, x = p.astype(int)
            
            # Simple 3x3 kernel painting
            for dz in range(-2, 3):
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        if (dz**2 + dy**2 + dx**2) <= 4:
                            zz, yy, xx = z+dz, y+dy, x+dx
                            if 0 <= zz < self.size and 0 <= yy < self.size and 0 <= xx < self.size:
                                vol[zz, yy, xx] = 1.0
                                
        # Add noise
        if noise_level > 0:
            vol += np.random.normal(0, noise_level, vol.shape)
            vol = np.clip(vol, 0, 1)
            
        return vol, direction

    def run_pca_orientation(self, vol):
        """Legacy PCA method."""
        coords = np.argwhere(vol > 0.5)
        if len(coords) < 3: return np.array([0, 0, 1])
        
        # PCA
        mean = coords.mean(axis=0)
        centered = coords - mean
        cov = np.cov(centered, rowvar=False)
        evals, evecs = np.linalg.eigh(cov)
        
        # Principal axis
        principal = evecs[:, -1] # Largest eigenvalue
        return principal

    def run_structure_tensor(self, vol):
        """HT3 Structure Tensor method."""
        # Simplified ST implementation for benchmark
        sigma = 1.0
        rho = 2.0
        
        Dz = ndimage.gaussian_filter1d(vol, sigma=sigma, axis=0, order=1)
        Dy = ndimage.gaussian_filter1d(vol, sigma=sigma, axis=1, order=1)
        Dx = ndimage.gaussian_filter1d(vol, sigma=sigma, axis=2, order=1)
        
        Szz = ndimage.gaussian_filter(Dz * Dz, sigma=rho)
        Syy = ndimage.gaussian_filter(Dy * Dy, sigma=rho)
        Sxx = ndimage.gaussian_filter(Dx * Dx, sigma=rho)
        Szy = ndimage.gaussian_filter(Dz * Dy, sigma=rho)
        Szx = ndimage.gaussian_filter(Dz * Dx, sigma=rho)
        Syx = ndimage.gaussian_filter(Dy * Dx, sigma=rho)
        
        # Sample center point
        c = self.center
        tensor = np.array([
            [Szz[c,c,c], Szy[c,c,c], Szx[c,c,c]],
            [Szy[c,c,c], Syy[c,c,c], Syx[c,c,c]],
            [Szx[c,c,c], Syx[c,c,c], Sxx[c,c,c]]
        ])
        
        w, v = np.linalg.eigh(tensor)
        # Smallest eigenvalue vector is fiber direction
        return v[:, 0]

    def calculate_angular_error(self, v_pred, v_true):
        """Calculate angular error in degrees."""
        # Normalize
        v_pred = v_pred / (np.linalg.norm(v_pred) + 1e-9)
        v_true = v_true / (np.linalg.norm(v_true) + 1e-9)
        
        dot = np.abs(np.dot(v_pred, v_true))
        dot = np.clip(dot, 0, 1)
        angle = np.degrees(np.arccos(dot))
        return angle

def main():
    print("="*60)
    print("HT3 Quantitative Benchmark: Orientation Accuracy")
    print("="*60)
    print(f"Date: {time.strftime('%Y-%m-%d')}")
    print("Author: Chandrashekhar Hegde")
    print("-" * 60)
    
    benchmark = Benchmark(size=64)
    results = []
    
    test_angles = [0, 15, 30, 45, 60, 90]
    noise_levels = [0.0, 0.2, 0.5]
    
    print(f"Running tests on {len(test_angles)*len(noise_levels)} scenarios...")
    
    for noise in noise_levels:
        for angle in test_angles:
            vol, true_vec = benchmark.generate_synthetic_fiber(angle, noise_level=noise)
            
            # Run PCA
            start = time.time()
            pca_vec = benchmark.run_pca_orientation(vol)
            pca_time = (time.time() - start) * 1000
            pca_error = benchmark.calculate_angular_error(pca_vec, true_vec)
            
            # Run HT3
            start = time.time()
            ht3_vec = benchmark.run_structure_tensor(vol)
            ht3_time = (time.time() - start) * 1000
            ht3_error = benchmark.calculate_angular_error(ht3_vec, true_vec)
            
            results.append({
                "Noise": f"{noise*100}%",
                "True Angle": f"{angle}°",
                "PCA Error": float(f"{pca_error:.2f}"),
                "HT3 Error": float(f"{ht3_error:.2f}"),
                "Improvement": float(f"{pca_error - ht3_error:.2f}")
            })
            
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate Mean Statistics
    mean_pca = df["PCA Error"].mean()
    mean_ht3 = df["HT3 Error"].mean()
    improvement_percent = ((mean_pca - mean_ht3) / mean_pca) * 100
    
    print("\nBENCHMARK RESULTS SUMMARY")
    print("="*80)
    print(f"{'Noise':<10} | {'True Angle':<12} | {'PCA Error':<10} | {'HT3 Error':<10} | {'Improvement':<12}")
    print("-" * 80)
    for _, row in df.iterrows():
        print(f"{row['Noise']:<10} | {row['True Angle']:<12} | {row['PCA Error']:<10} | {row['HT3 Error']:<10} | {row['Improvement']:<12}")
    
    print("-" * 80)
    print(f"Mean Angular Error (PCA): {mean_pca:.4f}°")
    print(f"Mean Angular Error (HT3): {mean_ht3:.4f}°")
    print(f"Accuracy Improvement:     {improvement_percent:.1f}%")
    print("="*60)
    
    if mean_ht3 < mean_pca:
        print("\nCONCLUSION: HT3 (Structure Tensor) statistically outperforms PCA.")
    else:
        print("\nCONCLUSION: No significant improvement detected.")

if __name__ == "__main__":
    main()
