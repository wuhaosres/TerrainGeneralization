"""
Helper utilities for terrain processing.
Provides common utility functions and classes.
"""

import os
import numpy as np
import rasterio
from typing import Tuple, Optional, Dict, Any
import json


class GeoHelper:
    """
    Geographic utility functions.
    Provides helper methods for geospatial operations.
    """
    
    @staticmethod
    def get_extent(file_path: str) -> Tuple[float, float, float, float]:
        """
        Get spatial extent of raster file.
        
        Args:
            file_path: Path to raster file
            
        Returns:
            Extent tuple (xmin, ymin, xmax, ymax)
        """
        with rasterio.open(file_path) as dataset:
            bounds = dataset.bounds
            return (bounds.left, bounds.bottom, bounds.right, bounds.top)
    
    @staticmethod
    def get_center(file_path: str) -> Tuple[float, float]:
        """
        Get center coordinates of raster file.
        
        Args:
            file_path: Path to raster file
            
        Returns:
            Center coordinates (x, y)
        """
        extent = GeoHelper.get_extent(file_path)
        return (
            (extent[0] + extent[2]) / 2,
            (extent[1] + extent[3]) / 2
        )
    
    @staticmethod
    def get_pixel_coordinates(
        file_path: str,
        x: float,
        y: float
    ) -> Tuple[int, int]:
        """
        Convert geographic coordinates to pixel coordinates.
        
        Args:
            file_path: Path to raster file
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Pixel coordinates (row, col)
        """
        with rasterio.open(file_path) as dataset:
            return dataset.index(x, y)
    
    @staticmethod
    def get_geographic_coordinates(
        file_path: str,
        row: int,
        col: int
    ) -> Tuple[float, float]:
        """
        Convert pixel coordinates to geographic coordinates.
        
        Args:
            file_path: Path to raster file
            row: Row index
            col: Column index
            
        Returns:
            Geographic coordinates (x, y)
        """
        with rasterio.open(file_path) as dataset:
            return dataset.xy(row, col)
    
    @staticmethod
    def copy_metadata(
        source_path: str,
        target_path: str
    ) -> None:
        """
        Copy metadata from source to target raster.
        
        Args:
            source_path: Source raster path
            target_path: Target raster path
        """
        with rasterio.open(source_path) as src:
            source_crs = src.crs
            source_transform = src.transform
        
        with rasterio.open(target_path, 'r+') as dst:
            dst.crs = source_crs
            dst.transform = source_transform


class MathHelper:
    """
    Mathematical utility functions.
    Provides helper methods for numerical operations.
    """
    
    @staticmethod
    def normalize(
        array: np.ndarray,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None
    ) -> np.ndarray:
        """
        Normalize array to [0, 1] range.
        
        Args:
            array: Input array
            min_val: Minimum value (optional)
            max_val: Maximum value (optional)
            
        Returns:
            Normalized array
        """
        array = array.astype(np.float32)
        
        if min_val is None:
            min_val = np.min(array)
        if max_val is None:
            max_val = np.max(array)
        
        if max_val - min_val == 0:
            return np.zeros_like(array)
        
        return (array - min_val) / (max_val - min_val)
    
    @staticmethod
    def rescale(
        array: np.ndarray,
        new_min: float,
        new_max: float
    ) -> np.ndarray:
        """
        Rescale array to new range.
        
        Args:
            array: Input array
            new_min: New minimum value
            new_max: New maximum value
            
        Returns:
            Rescaled array
        """
        normalized = MathHelper.normalize(array)
        return normalized * (new_max - new_min) + new_min
    
    @staticmethod
    def compute_gradient(
        array: np.ndarray,
        axis: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradient of array.
        
        Args:
            array: Input array
            axis: Axis for gradient (optional)
            
        Returns:
            Gradient tuple (dy, dx)
        """
        return np.gradient(array, axis=axis)
    
    @staticmethod
    def compute_distance(
        point1: Tuple[float, float],
        point2: Tuple[float, float]
    ) -> float:
        """
        Compute Euclidean distance between two points.
        
        Args:
            point1: First point (x, y)
            point2: Second point (x, y)
            
        Returns:
            Distance
        """
        return np.sqrt(
            (point1[0] - point2[0]) ** 2 +
            (point1[1] - point2[1]) ** 2
        )


class FileHelper:
    """
    File utility functions.
    Provides helper methods for file operations.
    """
    
    @staticmethod
    def ensure_directory(directory: str) -> None:
        """
        Ensure directory exists, create if necessary.
        
        Args:
            directory: Directory path
        """
        os.makedirs(directory, exist_ok=True)
    
    @staticmethod
    def get_file_extension(file_path: str) -> str:
        """
        Get file extension.
        
        Args:
            file_path: File path
            
        Returns:
            File extension
        """
        return os.path.splitext(file_path)[1]
    
    @staticmethod
    def change_extension(
        file_path: str,
        new_extension: str
    ) -> str:
        """
        Change file extension.
        
        Args:
            file_path: File path
            new_extension: New extension
            
        Returns:
            New file path
        """
        base = os.path.splitext(file_path)[0]
        return base + new_extension
    
    @staticmethod
    def read_json(file_path: str) -> Dict[str, Any]:
        """
        Read JSON file.
        
        Args:
            file_path: JSON file path
            
        Returns:
            Dictionary from JSON
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def write_json(
        data: Dict[str, Any],
        file_path: str
    ) -> None:
        """
        Write dictionary to JSON file.
        
        Args:
            data: Data dictionary
            file_path: Output file path
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def file_exists(file_path: str) -> bool:
        """
        Check if file exists.
        
        Args:
            file_path: File path
            
        Returns:
            True if exists
        """
        return os.path.exists(file_path)
    
    @staticmethod
    def delete_file(file_path: str) -> None:
        """
        Delete file if exists.
        
        Args:
            file_path: File path
        """
        if os.path.exists(file_path):
            os.remove(file_path)
