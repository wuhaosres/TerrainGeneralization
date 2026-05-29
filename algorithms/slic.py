"""
Terrain-aware SLIC superpixel segmentation algorithm.
Implements advanced segmentation with terrain feature constraints.
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from core.interfaces import ISegmentationStrategy
from core.base import TerrainProcessorBase


@dataclass
class SLICConfiguration:
    """
    Configuration container for SLIC algorithm parameters.
    Encapsulates all tunable parameters for segmentation.
    """
    num_segments: int = 200
    compactness: float = 8.0
    terrain_weight: float = 4.0
    max_iterations: int = 15
    min_region_size: int = 30
    enforce_connectivity: bool = True
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.num_segments <= 0:
            raise ValueError("num_segments must be positive")
        if self.compactness <= 0:
            raise ValueError("compactness must be positive")
        if self.terrain_weight < 0:
            raise ValueError("terrain_weight must be non-negative")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.min_region_size <= 0:
            raise ValueError("min_region_size must be positive")


class ClusterCenter:
    """
    Represents a superpixel cluster center.
    Encapsulates spatial and feature coordinates.
    """
    
    def __init__(
        self,
        y: float,
        x: float,
        lab: np.ndarray,
        terrain_feature: float
    ):
        """
        Initialize cluster center.
        
        Args:
            y: Y coordinate
            x: X coordinate
            lab: LAB color values
            terrain_feature: Terrain feature value
        """
        self.y = y
        self.x = x
        self.lab = lab
        self.terrain_feature = terrain_feature
    
    def update(
        self,
        y: float,
        x: float,
        lab: np.ndarray,
        terrain_feature: float
    ) -> None:
        """Update cluster center coordinates and features."""
        self.y = y
        self.x = x
        self.lab = lab
        self.terrain_feature = terrain_feature
    
    def to_array(self) -> np.ndarray:
        """Convert to array representation."""
        return np.array([
            self.y,
            self.x,
            self.lab[0],
            self.lab[1],
            self.lab[2],
            self.terrain_feature
        ])


class TerrainAwareSLIC(TerrainProcessorBase, ISegmentationStrategy):
    """
    Terrain-aware SLIC superpixel segmentation algorithm.
    
    This class implements the Strategy pattern for segmentation,
    incorporating terrain features to improve boundary adherence.
    
    Attributes:
        processor_type: Type identifier for the processor
        version: Version number
    """
    
    processor_type = 'segmentation'
    version = '2.0'
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SLIC processor.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self._slic_config = self._create_slic_config()
        self._cluster_centers: list = []
        self._labels: Optional[np.ndarray] = None
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        pass
    
    def _create_slic_config(self) -> SLICConfiguration:
        """Create SLIC configuration from config dict."""
        return SLICConfiguration(
            num_segments=self._config.get('num_segments', 200),
            compactness=self._config.get('compactness', 8.0),
            terrain_weight=self._config.get('terrain_weight', 4.0),
            max_iterations=self._config.get('max_iterations', 15),
            min_region_size=self._config.get('min_region_size', 30)
        )
    
    def process(
        self,
        image: np.ndarray,
        terrain_features: np.ndarray,
        structure_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Process image with terrain-aware SLIC segmentation.
        
        Args:
            image: Input RGB image
            terrain_features: Terrain feature map
            structure_mask: Optional structure line mask
            
        Returns:
            Segmentation labels array
        """
        self._slic_config.validate()
        
        H, W = image.shape[:2]
        
        lab_image = self._convert_to_lab(image)
        
        step = self._compute_grid_step(H, W)
        
        self._cluster_centers = self._initialize_centers(
            lab_image, terrain_features, step
        )
        
        labels = self._iterate_clustering(
            lab_image,
            terrain_features,
            structure_mask,
            step
        )
        
        if self._slic_config.enforce_connectivity:
            labels = self._enforce_connectivity(labels)
        
        self._labels = labels
        return labels
    
    def segment(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Implement ISegmentationStrategy interface.
        
        Args:
            image: Input image
            **kwargs: Additional parameters
            
        Returns:
            Segmentation labels
        """
        terrain_features = kwargs.get('terrain_features')
        structure_mask = kwargs.get('structure_mask')
        
        if terrain_features is None:
            raise ValueError("terrain_features is required")
        
        return self.process(image, terrain_features, structure_mask)
    
    def apply(self, data: Any, **kwargs) -> Any:
        """Implement IStrategy interface."""
        return self.segment(data, **kwargs)
    
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return "TerrainAwareSLIC"
    
    def _convert_to_lab(self, image: np.ndarray) -> np.ndarray:
        """
        Convert RGB image to LAB color space.
        
        Args:
            image: RGB image array
            
        Returns:
            LAB color space image
        """
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        return np.float32(lab_image)
    
    def _compute_grid_step(self, height: int, width: int) -> int:
        """
        Compute grid step size for initial cluster centers.
        
        Args:
            height: Image height
            width: Image width
            
        Returns:
            Grid step size
        """
        return int(np.sqrt(height * width / self._slic_config.num_segments))
    
    def _initialize_centers(
        self,
        lab_image: np.ndarray,
        terrain_features: np.ndarray,
        step: int
    ) -> list:
        """
        Initialize cluster centers on a regular grid.
        
        Args:
            lab_image: LAB color space image
            terrain_features: Terrain feature map
            step: Grid step size
            
        Returns:
            List of ClusterCenter objects
        """
        H, W = lab_image.shape[:2]
        centers = []
        
        for y in range(step // 2, H, step):
            for x in range(step // 2, W, step):
                y, x = self._perturb_center(lab_image, y, x)
                
                center = ClusterCenter(
                    y=float(y),
                    x=float(x),
                    lab=lab_image[y, x],
                    terrain_feature=terrain_features[y, x]
                )
                centers.append(center)
        
        return centers
    
    def _perturb_center(
        self,
        lab_image: np.ndarray,
        y: int,
        x: int
    ) -> Tuple[int, int]:
        """
        Perturb center to lowest gradient position.
        
        Args:
            lab_image: LAB color space image
            y: Initial Y coordinate
            x: Initial X coordinate
            
        Returns:
            Perturbed (y, x) coordinates
        """
        H, W = lab_image.shape[:2]
        
        gradient = np.zeros((3, 3))
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    gradient[dy + 1, dx + 1] = np.sum(
                        np.abs(lab_image[ny, nx] - lab_image[y, x])
                    )
        
        dy, dx = np.unravel_index(np.argmin(gradient), (3, 3))
        return y + (dy - 1), x + (dx - 1)
    
    def _iterate_clustering(
        self,
        lab_image: np.ndarray,
        terrain_features: np.ndarray,
        structure_mask: Optional[np.ndarray],
        step: int
    ) -> np.ndarray:
        """
        Perform iterative clustering.
        
        Args:
            lab_image: LAB color space image
            terrain_features: Terrain feature map
            structure_mask: Structure line mask
            step: Grid step size
            
        Returns:
            Segmentation labels
        """
        H, W = lab_image.shape[:2]
        labels = -1 * np.ones((H, W), dtype=np.int32)
        distance_map = np.full((H, W), np.inf, dtype=np.float32)
        
        for iteration in range(self._slic_config.max_iterations):
            distance_map[:] = np.inf
            
            for idx, center in enumerate(self._cluster_centers):
                self._assign_pixels(
                    lab_image,
                    terrain_features,
                    structure_mask,
                    center,
                    idx,
                    step,
                    labels,
                    distance_map
                )
            
            self._update_centers(lab_image, terrain_features, labels)
        
        return labels
    
    def _assign_pixels(
        self,
        lab_image: np.ndarray,
        terrain_features: np.ndarray,
        structure_mask: Optional[np.ndarray],
        center: ClusterCenter,
        center_idx: int,
        step: int,
        labels: np.ndarray,
        distance_map: np.ndarray
    ) -> None:
        """
        Assign pixels to nearest cluster center.
        
        Args:
            lab_image: LAB color space image
            terrain_features: Terrain feature map
            structure_mask: Structure line mask
            center: Cluster center
            center_idx: Center index
            step: Grid step size
            labels: Labels array to update
            distance_map: Distance map to update
        """
        H, W = lab_image.shape[:2]
        y, x = int(center.y), int(center.x)
        
        y1, y2 = max(0, y - step), min(H, y + step)
        x1, x2 = max(0, x - step), min(W, x + step)
        
        window_y, window_x = np.mgrid[y1:y2, x1:x2]
        pixel_lab = lab_image[window_y, window_x]
        pixel_terrain = terrain_features[window_y, window_x]
        
        color_distance = np.sum(
            (pixel_lab - center.lab) ** 2,
            axis=-1
        )
        
        spatial_distance = (
            (window_y - y) ** 2 +
            (window_x - x) ** 2
        )
        
        normalized_distance = (
            color_distance +
            (spatial_distance / step ** 2) * (self._slic_config.compactness ** 2)
        )
        
        terrain_distance = (pixel_terrain - center.terrain_feature) ** 2
        
        max_terrain = np.max(terrain_features) ** 2
        if max_terrain == 0:
            max_terrain = 1
        
        total_distance = (
            normalized_distance +
            self._slic_config.terrain_weight * (terrain_distance / max_terrain)
        )
        
        if structure_mask is not None:
            struct_lines = structure_mask[window_y, window_x] > 0
            total_distance[struct_lines] += 1e9
        
        update_mask = total_distance < distance_map[y1:y2, x1:x2]
        distance_map[y1:y2, x1:x2][update_mask] = total_distance[update_mask]
        labels[y1:y2, x1:x2][update_mask] = center_idx
    
    def _update_centers(
        self,
        lab_image: np.ndarray,
        terrain_features: np.ndarray,
        labels: np.ndarray
    ) -> None:
        """
        Update cluster centers based on assigned pixels.
        
        Args:
            lab_image: LAB color space image
            terrain_features: Terrain feature map
            labels: Current labels
        """
        for idx, center in enumerate(self._cluster_centers):
            mask = labels == idx
            
            if np.any(mask):
                coords = np.argwhere(mask)
                new_y, new_x = np.mean(coords, axis=0)
                
                new_lab = np.mean(lab_image[mask], axis=0)
                new_terrain = np.mean(terrain_features[mask])
                
                center.update(new_y, new_x, new_lab, new_terrain)
    
    def _enforce_connectivity(self, labels: np.ndarray) -> np.ndarray:
        """
        Enforce connectivity and remove small regions.
        
        Args:
            labels: Input labels
            
        Returns:
            Cleaned labels
        """
        H, W = labels.shape
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            mask = labels == label
            area = np.count_nonzero(mask)
            
            if area < self._slic_config.min_region_size:
                y_indices, x_indices = np.where(mask)
                
                for y, x in zip(y_indices, x_indices):
                    best_label = label
                    min_dist = 999
                    
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dy == 0 and dx == 0:
                                continue
                            
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < H and 0 <= nx < W:
                                if labels[ny, nx] != label:
                                    dist = dy * dy + dx * dx
                                    if dist < min_dist:
                                        min_dist = dist
                                        best_label = labels[ny, nx]
                    
                    labels[y, x] = best_label
        
        return labels
    
    def compute_curvature(self, dem: np.ndarray) -> np.ndarray:
        """
        Compute terrain curvature from DEM.
        
        Args:
            dem: Digital Elevation Model
            
        Returns:
            Curvature map
        """
        dy, dx = np.gradient(dem)
        dyy, dyx = np.gradient(dy)
        dxx, dxy = np.gradient(dx)
        
        curvature = np.abs(dyy + dxx)
        
        return cv2.normalize(
            curvature,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
            dtype=cv2.CV_32F
        )
    
    def draw_contours(
        self,
        image: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        """
        Draw segmentation contours on image.
        
        Args:
            image: Input image
            labels: Segmentation labels
            
        Returns:
            Image with contours
        """
        output = image.copy()
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            mask = np.uint8(labels == label)
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(output, contours, -1, (0, 0, 255), thickness=1)
        
        return output
