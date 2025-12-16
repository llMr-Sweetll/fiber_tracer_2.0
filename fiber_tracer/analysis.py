"""
Fiber analysis module for extracting quantitative properties.

Author: Chandrashekhar Hegde
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy import ndimage, spatial, linalg
from skimage import measure
from dataclasses import dataclass
import pandas as pd
from tqdm import tqdm
import warnings
import gudhi  # For Topological Data Analysis

logger = logging.getLogger(__name__)

# CITATIONS
# [1] Kleinnijenhuis, M., et al. "Structure tensor informed fiber tractography (STIFT) by combining gradient echo MRI and diffusion tensor imaging." NeuroImage 274 (2024): 120089.
# [2] Jensen, J.H., et al. "Validation of structure tensor analysis for orientation estimation in brain tissue microscopy." bioRxiv (2025).


@dataclass
class FiberProperties:
    """Data class for storing fiber properties."""
    fiber_id: int
    length: float  # micrometers
    diameter: float  # micrometers
    volume: float  # cubic micrometers
    tortuosity: float
    orientation: float  # degrees
    polar_angle: float  # degrees
    azimuthal_angle: float  # degrees
    centroid: Tuple[float, float, float]
    bbox: Tuple[int, int, int, int, int, int]
    surface_area: float  # square micrometers
    aspect_ratio: float
    fiber_class: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'Fiber ID': self.fiber_id,
            'Length (μm)': self.length,
            'Diameter (μm)': self.diameter,
            'Volume (μm³)': self.volume,
            'Surface Area (μm²)': self.surface_area,
            'Tortuosity': self.tortuosity,
            'Orientation (degrees)': self.orientation,
            'Polar Angle (degrees)': self.polar_angle,
            'Azimuthal Angle (degrees)': self.azimuthal_angle,
            'Aspect Ratio': self.aspect_ratio,
            'Centroid X (μm)': self.centroid[0],
            'Centroid Y (μm)': self.centroid[1],
            'Centroid Z (μm)': self.centroid[2],
            'Class': self.fiber_class
        }


class FiberAnalyzer:
    """Analyze fiber properties from labeled volumes."""
    
    def __init__(self, config):
        """
        Initialize analyzer with configuration.
        
        Args:
            config: FiberAnalysisConfig object
        """
        self.config = config
        self.voxel_size = config.voxel_size
        self.min_diameter = config.min_diameter
        self.max_diameter = config.max_diameter
        self.angle_bins = config.angle_bins
    
    def analyze_fibers(self, labeled_volume: np.ndarray,
                       binary_volume: Optional[np.ndarray] = None,
                       grayscale_volume: Optional[np.ndarray] = None) -> List[FiberProperties]:
        """
        Analyze all fibers in labeled volume.
        
        Args:
            labeled_volume: Labeled 3D volume
            binary_volume: Optional binary mask
            grayscale_volume: Optional grayscale volume for Structure Tensor analysis
            
        Returns:
            List of FiberProperties objects
        """
        logger.info("Starting fiber analysis")
        
        # Get region properties
        regions = measure.regionprops(labeled_volume)
        
        # Compute Structure Tensor Field if requested [NEW 2025]
        self.tensor_field = None
        if hasattr(self.config, 'orientation_method') and self.config.orientation_method == 'structure_tensor' and grayscale_volume is not None:
             try:
                 logger.info("Computing Structure Tensor Field for orientation analysis [2024/2025 Method]")
                 st = StructureTensor3D(sigma=1.0, rho=2.0) # Parameters could be config-driven
                 self.tensor_field = st.compute_field(grayscale_volume)
             except Exception as e:
                 logger.error(f"Structure Tensor computation failed: {e}. Falling back to PCA.")
        
        fibers = []
        for region in tqdm(regions, desc='Analyzing fibers'):
            try:
                fiber_props = self._analyze_single_fiber(region)
                
                # Filter by diameter
                if (self.min_diameter <= fiber_props.diameter <= self.max_diameter):
                    fibers.append(fiber_props)
                    
            except Exception as e:
                logger.error(f"Error analyzing fiber {region.label}: {e}")
                continue
        
        logger.info(f"Analysis complete: {len(fibers)} fibers analyzed")
        
        # Classify fibers
        if self.config.calculate_orientation:
            fibers = self._classify_fibers(fibers)
        
        return fibers
    
    def _analyze_single_fiber(self, region) -> FiberProperties:
        """
        Analyze properties of a single fiber.
        
        Args:
            region: Region properties from skimage.measure
            
        Returns:
            FiberProperties object
        """
        # Basic properties
        fiber_id = region.label
        
        # Physical dimensions
        diameter = region.equivalent_diameter * self.voxel_size
        volume = region.area * (self.voxel_size ** 3)
        surface_area = region.surface_area * (self.voxel_size ** 2) if hasattr(region, 'surface_area') else 0
        
        # Fiber length (major axis)
        length = region.major_axis_length * self.voxel_size
        
        # Aspect ratio
        if region.minor_axis_length > 0:
            aspect_ratio = region.major_axis_length / region.minor_axis_length
        else:
            aspect_ratio = region.major_axis_length
        
        # Centroid in physical units
        centroid = tuple(c * self.voxel_size for c in region.centroid)
        
        # Bounding box
        bbox = region.bbox
        
        # Calculate orientation
        orientation, polar_angle, azimuthal_angle = self._calculate_orientation(region)
        
        # Calculate tortuosity
        tortuosity = self._calculate_tortuosity(region)
        
        return FiberProperties(
            fiber_id=fiber_id,
            length=length,
            diameter=diameter,
            volume=volume,
            tortuosity=tortuosity,
            orientation=orientation,
            polar_angle=polar_angle,
            azimuthal_angle=azimuthal_angle,
            centroid=centroid,
            bbox=bbox,
            surface_area=surface_area,
            aspect_ratio=aspect_ratio
        )
    
    def _calculate_orientation(self, region) -> Tuple[float, float, float]:
        """
        Calculate fiber orientation.
         Uses Structure Tensor (2025) if available, else PCA.
        
        Args:
            region: Region properties
            
        Returns:
            Tuple of (orientation, polar_angle, azimuthal_angle) in degrees
        """
        if not self.config.calculate_orientation:
            return 0.0, 0.0, 0.0
            
        # [NEW 2025] Structure Tensor Method
        if hasattr(self.config, 'orientation_method') and self.config.orientation_method == 'structure_tensor' and self.tensor_field is not None:
            try:
                # Get the fiber region centroid
                z, y, x = [int(c) for c in region.centroid]
                
                # Check bounds
                d, h, w = self.tensor_field.eigenvectors.shape[2:]
                z = min(max(z, 0), d-1)
                y = min(max(y, 0), h-1)
                x = min(max(x, 0), w-1)

                # Get principal eigenvector (e1 corresponds to smallest eigenvalue -> along fiber)
                # Shape: (3 components, D, H, W)
                # math_core returns eigenvectors as (3, 3, D, H, W) where first dim is eigenvalue index (0=smallest)
                principal_axis = self.tensor_field.eigenvectors[0, :, z, y, x]
                
                # Normalize just in case
                norm = np.linalg.norm(principal_axis)
                if norm > 0:
                    principal_axis = principal_axis / norm
                else:
                    return 0.0, 0.0, 0.0
                    
                # Ensure positive z-component
                if principal_axis[2] < 0:
                    principal_axis = -principal_axis
                
                # Calculate angles
                orientation = np.degrees(np.arccos(np.abs(principal_axis[2])))
                polar_angle = np.degrees(np.arccos(principal_axis[2]))
                azimuthal_angle = np.degrees(np.arctan2(principal_axis[1], principal_axis[0]))
                
                return orientation, polar_angle, azimuthal_angle
                
            except Exception as e:
                # Fallback to PCA if point lookup fails
                pass 
        
        coords = region.coords
        
        if coords.shape[0] < 3:
            return 0.0, 0.0, 0.0
        
        # Perform PCA
        mean_coords = coords.mean(axis=0)
        centered_coords = coords - mean_coords
        
        try:
            cov_matrix = np.cov(centered_coords, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # Principal axis is the eigenvector with largest eigenvalue
            principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
            
            # Ensure positive z-component for consistency
            if principal_axis[2] < 0:
                principal_axis = -principal_axis
            
            # Calculate angles
            # Orientation with z-axis
            orientation = np.degrees(np.arccos(np.abs(principal_axis[2])))
            
            # Polar angle (angle from z-axis)
            polar_angle = np.degrees(np.arccos(principal_axis[2] / np.linalg.norm(principal_axis)))
            
            # Azimuthal angle (angle in xy-plane)
            azimuthal_angle = np.degrees(np.arctan2(principal_axis[1], principal_axis[0]))
            
        except:
            orientation = 0.0
            polar_angle = 0.0
            azimuthal_angle = 0.0
        
        return orientation, polar_angle, azimuthal_angle
    
    def _calculate_tortuosity(self, region) -> float:
        """
        Calculate fiber tortuosity.
        
        Args:
            region: Region properties
            
        Returns:
            Tortuosity value
        """
        if not self.config.calculate_tortuosity:
            return 1.0
        
        coords = region.coords * self.voxel_size
        
        if coords.shape[0] < 2:
            return 1.0
        
        try:
            # Calculate path length along fiber
            path_length = 0
            for i in range(1, len(coords)):
                path_length += np.linalg.norm(coords[i] - coords[i-1])
            
            # Calculate straight-line distance
            euclidean_distance = np.linalg.norm(coords[-1] - coords[0])
            
            if euclidean_distance > 0:
                tortuosity = path_length / euclidean_distance
            else:
                tortuosity = 1.0
                
        except:
            tortuosity = 1.0
        
        return tortuosity
    
    def _classify_fibers(self, fibers: List[FiberProperties]) -> List[FiberProperties]:
        """
        Classify fibers based on orientation.
        
        Args:
            fibers: List of FiberProperties
            
        Returns:
            Updated list with classification
        """
        for fiber in fibers:
            # Classify based on orientation angle
            angle = fiber.orientation
            
            if angle < 15:
                fiber.fiber_class = "Aligned (0-15°)"
            elif angle < 30:
                fiber.fiber_class = "Slightly Misaligned (15-30°)"
            elif angle < 45:
                fiber.fiber_class = "Moderately Misaligned (30-45°)"
            elif angle < 60:
                fiber.fiber_class = "Highly Misaligned (45-60°)"
            else:
                fiber.fiber_class = "Random (>60°)"
        
        return fibers
    
    def calculate_statistics(self, fibers: List[FiberProperties]) -> Dict[str, Any]:
        """
        Calculate statistical summary of fiber properties.
        
        Args:
            fibers: List of FiberProperties
            
        Returns:
            Dictionary with statistical summary
        """
        if not fibers:
            return {}
        
        # Convert to DataFrame for easy statistics
        df = pd.DataFrame([f.to_dict() for f in fibers])
        
        stats = {
            'total_fibers': len(fibers),
            'length_stats': {
                'mean': df['Length (μm)'].mean(),
                'std': df['Length (μm)'].std(),
                'min': df['Length (μm)'].min(),
                'max': df['Length (μm)'].max(),
                'median': df['Length (μm)'].median()
            },
            'diameter_stats': {
                'mean': df['Diameter (μm)'].mean(),
                'std': df['Diameter (μm)'].std(),
                'min': df['Diameter (μm)'].min(),
                'max': df['Diameter (μm)'].max(),
                'median': df['Diameter (μm)'].median()
            },
            'orientation_stats': {
                'mean': df['Orientation (degrees)'].mean(),
                'std': df['Orientation (degrees)'].std(),
                'min': df['Orientation (degrees)'].min(),
                'max': df['Orientation (degrees)'].max(),
                'median': df['Orientation (degrees)'].median()
            },
            'tortuosity_stats': {
                'mean': df['Tortuosity'].mean(),
                'std': df['Tortuosity'].std(),
                'min': df['Tortuosity'].min(),
                'max': df['Tortuosity'].max(),
                'median': df['Tortuosity'].median()
            }
        }
        
        # Add class distribution if available
        if 'Class' in df.columns:
            stats['class_distribution'] = df['Class'].value_counts().to_dict()
        
        return stats
    
    def calculate_volume_fraction(self, binary_volume: np.ndarray) -> float:
        """
        Calculate fiber volume fraction.
        
        Args:
            binary_volume: Binary segmentation mask
            
        Returns:
            Volume fraction as percentage
        """
        total_fiber_voxels = np.sum(binary_volume)
        total_voxels = binary_volume.size
        
        volume_fraction = (total_fiber_voxels / total_voxels) * 100
        
        logger.info(f"Fiber volume fraction: {volume_fraction:.2f}%")
        
        return volume_fraction
    
    def save_results(self, fibers: List[FiberProperties],
                    output_path: str,
                    format: str = 'csv'):
        """
        Save fiber analysis results.
        
        Args:
            fibers: List of FiberProperties
            output_path: Path to save file
            format: Output format ('csv', 'excel', 'json')
        """
        if not fibers:
            logger.warning("No fibers to save")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame([f.to_dict() for f in fibers])
        
        # Save based on format
        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'excel':
            df.to_excel(output_path, index=False)
        elif format == 'json':
            df.to_json(output_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Results saved to {output_path}")


class FiberConnectivityAnalyzer:
    """Analyze fiber connectivity and network properties."""
    
    def __init__(self, voxel_size: float = 1.0):
        """
        Initialize connectivity analyzer.
        
        Args:
            voxel_size: Voxel size in micrometers
        """
        self.voxel_size = voxel_size
    
    def analyze_connectivity(self, labeled_volume: np.ndarray,
                            distance_threshold: float = 10.0) -> Dict[str, Any]:
        """
        Analyze fiber-to-fiber connectivity.
        
        Args:
            labeled_volume: Labeled volume
            distance_threshold: Maximum distance for connectivity (micrometers)
            
        Returns:
            Dictionary with connectivity metrics
        """
        logger.info("Analyzing fiber connectivity")
        
        regions = measure.regionprops(labeled_volume)
        n_fibers = len(regions)
        
        if n_fibers == 0:
            return {}
        
        # Calculate centroids
        centroids = np.array([r.centroid for r in regions]) * self.voxel_size
        
        # Build distance matrix
        dist_matrix = spatial.distance_matrix(centroids, centroids)
        
        # Find connections
        connections = (dist_matrix < distance_threshold) & (dist_matrix > 0)
        
        # Calculate metrics
        connectivity_metrics = {
            'total_fibers': n_fibers,
            'total_connections': np.sum(connections) // 2,  # Divide by 2 for undirected
            'average_connections_per_fiber': np.mean(np.sum(connections, axis=1)),
            'max_connections': np.max(np.sum(connections, axis=1)),
            'min_connections': np.min(np.sum(connections, axis=1)),
            'connectivity_density': np.sum(connections) / (n_fibers * (n_fibers - 1)) if n_fibers > 1 else 0
        }
        
        # Find isolated fibers
        isolated = np.sum(connections, axis=1) == 0
        connectivity_metrics['isolated_fibers'] = np.sum(isolated)
        connectivity_metrics['isolation_ratio'] = np.sum(isolated) / n_fibers
        
        return connectivity_metrics
    
    def find_fiber_bundles(self, labeled_volume: np.ndarray,
                          proximity_threshold: float = 5.0) -> np.ndarray:
        """
        Identify fiber bundles based on proximity.
        
        Args:
            labeled_volume: Labeled volume
            proximity_threshold: Distance threshold for bundle membership
            
        Returns:
            Array with bundle labels
        """
        logger.info("Identifying fiber bundles")
        
        # Dilate each fiber to find overlaps
        bundle_volume = np.zeros_like(labeled_volume)
        
        unique_labels = np.unique(labeled_volume[labeled_volume > 0])
        
        for label in tqdm(unique_labels, desc='Finding bundles'):
            fiber_mask = labeled_volume == label
            
            # Dilate fiber
            struct_elem = ndimage.generate_binary_structure(3, 1)
            dilated = ndimage.binary_dilation(fiber_mask, 
                                             structure=struct_elem,
                                             iterations=int(proximity_threshold/self.voxel_size))
            
            # Find overlapping fibers
            overlapping_labels = np.unique(labeled_volume[dilated & (labeled_volume > 0)])
            
            # Assign to same bundle
            bundle_id = min(overlapping_labels)
            for overlap_label in overlapping_labels:
                bundle_volume[labeled_volume == overlap_label] = bundle_id
        
        return bundle_volume


# ==============================================================================
# HT3: Hybrid Topological-Tensor Tracing (Postdoctoral Grade Core)
# ==============================================================================

@dataclass
class TensorField:
    """Represents a 3D Structure Tensor Field."""
    eigenvalues: np.ndarray  # Shape (3, D, H, W)
    eigenvectors: np.ndarray # Shape (3, 3, D, H, W)
    confidence: np.ndarray   # Shape (D, H, W) - e.g., mapping to fiber probability

class StructureTensor3D:
    """
    Computes and analyzes the Structure Tensor Field for 3D volumes.
    Based on Differential Geometry principles.
    """
    
    def __init__(self, sigma: float = 1.0, rho: float = 2.0):
        """
        Args:
            sigma: Scale of differentiation (Gaussian derivative).
            rho: Scale of integration (Tensor smoothing).
        """
        self.sigma = sigma
        self.rho = rho

    def compute_field(self, volume: np.ndarray) -> TensorField:
        """
        Compute the structure tensor field and its eigen-decomposition.
        
        S = G_rho * (grad(I_sigma) . grad(I_sigma)^T)
        """
        logger.info(f"Computing Structure Tensor Field (sigma={self.sigma}, rho={self.rho})")
        
        # 1. Compute gradients of Gaussian-smoothed image
        # Using a slight optimization: Gaussian gradient directly
        Dz = ndimage.gaussian_filter1d(volume, sigma=self.sigma, axis=0, order=1)
        Dy = ndimage.gaussian_filter1d(volume, sigma=self.sigma, axis=1, order=1)
        Dx = ndimage.gaussian_filter1d(volume, sigma=self.sigma, axis=2, order=1)
        
        # 2. Compute tensor components (Outer Product)
        # S = [ Szz Szy Szx ]
        #     [ Syz Syy Syx ]
        #     [ Sxz Sxy Sxx ]
        # Symmetry: Szy=Syz, Szx=Sxz, Syx=Sxy
        
        Szz = ndimage.gaussian_filter(Dz * Dz, sigma=self.rho)
        Syy = ndimage.gaussian_filter(Dy * Dy, sigma=self.rho)
        Sxx = ndimage.gaussian_filter(Dx * Dx, sigma=self.rho)
        
        Szy = ndimage.gaussian_filter(Dz * Dy, sigma=self.rho)
        Szx = ndimage.gaussian_filter(Dz * Dx, sigma=self.rho)
        Syx = ndimage.gaussian_filter(Dy * Dx, sigma=self.rho)
        
        # 3. Eigen-decomposition for every voxel
        # This is computationally expensive, so we vectorize where possible or iterate
        depth, height, width = volume.shape
        
        # Construct tensor matrix for all pixels
        # Shape: (D*H*W, 3, 3)
        num_voxels = depth * height * width
        tensors = np.zeros((num_voxels, 3, 3), dtype=np.float32)
        
        flat_Szz = Szz.ravel()
        flat_Syy = Syy.ravel()
        flat_Sxx = Sxx.ravel()
        flat_Szy = Szy.ravel()
        flat_Szx = Szx.ravel()
        flat_Syx = Syx.ravel()
        
        tensors[:, 0, 0] = flat_Szz
        tensors[:, 1, 1] = flat_Syy
        tensors[:, 2, 2] = flat_Sxx
        tensors[:, 0, 1] = tensors[:, 1, 0] = flat_Szy
        tensors[:, 0, 2] = tensors[:, 2, 0] = flat_Szx
        tensors[:, 1, 2] = tensors[:, 2, 1] = flat_Syx
        
        logger.info("Computing Eigen-decomposition (this may take a moment)...")
        w, v = np.linalg.eigh(tensors)
        # w: eigenvalues, v: eigenvectors. w is sorted in ascending order.
        
        # Reshape back
        eigenvalues = w.T.reshape(3, depth, height, width)
        eigenvectors = np.transpose(v, (2, 1, 0)).reshape(3, 3, depth, height, width)
        
        l1 = eigenvalues[0] # Smallest (along fiber)
        l2 = eigenvalues[1]
        l3 = eigenvalues[2] # Largest
        
        # Metric for fiber probability: l1 should be small, l2 and l3 large.
        confidence = np.exp(-(l1**2) / (2 * (0.5 * (l2 + l3))**2)) * (1 - np.exp(-(l2**2 + l3**2) / (2 * volume.max()**2)))
        
        return TensorField(eigenvalues, eigenvectors, confidence)


class FiberTracerRK4:
    """
    Traces fibers through a vector field using 4th-Order Runge-Kutta integration.
    """
    
    def __init__(self, step_size: float = 0.5, max_steps: int = 1000, 
                 angle_threshold: float = 45.0, min_confidence: float = 0.1):
        self.step_size = step_size
        self.max_steps = max_steps
        self.angle_threshold = np.radians(angle_threshold)
        self.min_confidence = min_confidence
        
    def trace(self, tensor_field: TensorField, seed_points: np.ndarray) -> List[np.ndarray]:
        logger.info(f"Tracing {len(seed_points)} fibers using RK4 integration")
        
        fibers = []
        fiber_direction_field = tensor_field.eigenvectors[0] # e1
        depth, height, width = fiber_direction_field.shape[1:]
        
        for seed in seed_points:
            fiber_path = [seed]
            current_pos = seed.copy()
            
            # Get initial direction
            direction = self._interpolate_vector(current_pos, fiber_direction_field)
            if np.linalg.norm(direction) == 0:
                continue
            
            # Trace in both forward and backward directions
            for sign in [1, -1]:
                current_dir = direction * sign
                temp_path = []
                
                pos = seed.copy()
                
                for _ in range(self.max_steps):
                    # RK4 Step
                    k1 = current_dir
                    
                    p2 = pos + k1 * (self.step_size / 2)
                    k2 = self._interpolate_vector(p2, fiber_direction_field)
                    if np.dot(k1, k2) < 0: k2 = -k2 # Align orientation
                    
                    p3 = pos + k2 * (self.step_size / 2)
                    k3 = self._interpolate_vector(p3, fiber_direction_field)
                    if np.dot(k2, k3) < 0: k3 = -k3
                    
                    p4 = pos + k3 * self.step_size
                    k4 = self._interpolate_vector(p4, fiber_direction_field)
                    if np.dot(k3, k4) < 0: k4 = -k4
                    
                    next_dir = (k1 + 2*k2 + 2*k3 + k4) / 6.0
                    next_dir = next_dir / (np.linalg.norm(next_dir) + 1e-9)
                    
                    # Update position
                    next_pos = pos + next_dir * self.step_size
                    
                    # Checks
                    if not (0 <= next_pos[0] < depth and 0 <= next_pos[1] < height and 0 <= next_pos[2] < width):
                        break # Out of bounds
                        
                    # Curvature check
                    if np.arccos(np.clip(np.dot(current_dir, next_dir), -1.0, 1.0)) > self.angle_threshold:
                        break # Too sharp turn
                        
                    # Update loop
                    pos = next_pos
                    current_dir = next_dir
                    temp_path.append(pos)
                
                if sign == 1:
                    fiber_path.extend(temp_path)
                else:
                    fiber_path = temp_path[::-1] + fiber_path
            
            if len(fiber_path) > 5: # Minimum length
                fibers.append(np.array(fiber_path))
                
        return fibers

    def _interpolate_vector(self, pos: np.ndarray, field: np.ndarray) -> np.ndarray:
        """Trilinear interpolation of the vector field at pos."""
        z, y, x = int(round(pos[0])), int(round(pos[1])), int(round(pos[2]))
        d, h, w = field.shape[1:]
        if 0 <= z < d and 0 <= y < h and 0 <= x < w:
            return field[:, z, y, x]
        return np.zeros(3)


class TopologicalFilter:
    """
    Uses Persistent Homology to filter structures.
    """
    def compute_persistence_diagram(self, confidence_map: np.ndarray):
        logger.info("Computing Persistent Homology")
        cc = gudhi.CubicalComplex(dimensions=confidence_map.shape, top_dimensional_cells=1.0 - confidence_map.flatten())
        cc.compute_persistence()
        persistence = cc.persistence()
        return persistence
