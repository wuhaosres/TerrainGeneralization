"""
DEM context manager implementation.
Provides context management for DEM processing operations.
"""

import os
import numpy as np
import rasterio
from typing import Optional, Dict, Any, Tuple
from contextlib import contextmanager
from dataclasses import dataclass

from utils.helpers import GeoHelper


@dataclass
class DEMMetadata:
    """
    Container for DEM metadata.
    """
    crs: Any
    transform: Any
    nodata: Optional[float]
    width: int
    height: int
    bounds: Tuple[float, float, float, float]
    resolution: Tuple[float, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'crs': str(self.crs),
            'transform': self.transform,
            'nodata': self.nodata,
            'width': self.width,
            'height': self.height,
            'bounds': self.bounds,
            'resolution': self.resolution
        }


class DEMContext:
    """
    Context object for DEM processing.
    
    Encapsulates DEM data and metadata for processing operations.
    """
    
    def __init__(
        self,
        file_path: str,
        data: Optional[np.ndarray] = None,
        metadata: Optional[DEMMetadata] = None
    ):
        """
        Initialize DEM context.
        
        Args:
            file_path: Path to DEM file
            data: DEM data array (optional)
            metadata: DEM metadata (optional)
        """
        self._file_path = file_path
        self._data = data
        self._metadata = metadata
        self._modified = False
    
    @property
    def data(self) -> np.ndarray:
        """Get DEM data."""
        if self._data is None:
            self._load_data()
        return self._data
    
    @data.setter
    def data(self, value: np.ndarray) -> None:
        """Set DEM data."""
        self._data = value
        self._modified = True
    
    @property
    def metadata(self) -> DEMMetadata:
        """Get DEM metadata."""
        if self._metadata is None:
            self._load_metadata()
        return self._metadata
    
    @property
    def file_path(self) -> str:
        """Get file path."""
        return self._file_path
    
    @property
    def is_modified(self) -> bool:
        """Check if data has been modified."""
        return self._modified
    
    def _load_data(self) -> None:
        """Load DEM data from file."""
        with rasterio.open(self._file_path) as dataset:
            self._data = dataset.read(1)
    
    def _load_metadata(self) -> None:
        """Load DEM metadata from file."""
        with rasterio.open(self._file_path) as dataset:
            self._metadata = DEMMetadata(
                crs=dataset.crs,
                transform=dataset.transform,
                nodata=dataset.nodata,
                width=dataset.width,
                height=dataset.height,
                bounds=(
                    dataset.bounds.left,
                    dataset.bounds.bottom,
                    dataset.bounds.right,
                    dataset.bounds.top
                ),
                resolution=(dataset.res[0], dataset.res[1])
            )
    
    def save(self, output_path: Optional[str] = None) -> None:
        """
        Save DEM data to file.
        
        Args:
            output_path: Output path (optional, defaults to original path)
        """
        output_path = output_path or self._file_path
        
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=self._data.shape[0],
            width=self._data.shape[1],
            count=1,
            dtype=self._data.dtype,
            crs=self.metadata.crs,
            transform=self.metadata.transform,
            nodata=self.metadata.nodata
        ) as dataset:
            dataset.write(self._data, 1)
        
        self._modified = False
    
    def get_value_at_point(self, x: float, y: float) -> float:
        """
        Get elevation value at geographic coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Elevation value
        """
        row, col = GeoHelper.get_pixel_coordinates(self._file_path, x, y)
        
        if 0 <= row < self.data.shape[0] and 0 <= col < self.data.shape[1]:
            return self.data[row, col]
        
        return np.nan


class DEMContextManager:
    """
    Context manager for DEM processing operations.
    
    Provides automatic resource management for DEM files,
    ensuring proper cleanup and error handling.
    """
    
    def __init__(
        self,
        file_path: str,
        mode: str = 'r',
        auto_save: bool = False
    ):
        """
        Initialize DEM context manager.
        
        Args:
            file_path: Path to DEM file
            mode: Access mode ('r' for read, 'w' for write)
            auto_save: Automatically save changes on exit
        """
        self._file_path = file_path
        self._mode = mode
        self._auto_save = auto_save
        self._context: Optional[DEMContext] = None
    
    def __enter__(self) -> DEMContext:
        """Enter context and load DEM."""
        if not os.path.exists(self._file_path):
            raise FileNotFoundError(f"DEM file not found: {self._file_path}")
        
        self._context = DEMContext(self._file_path)
        
        if self._mode == 'r':
            _ = self._context.data
        
        return self._context
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and cleanup."""
        if self._context is not None:
            if self._auto_save and self._context.is_modified:
                self._context.save()
            
            self._context = None
    
    @staticmethod
    @contextmanager
    def open(
        file_path: str,
        mode: str = 'r',
        auto_save: bool = False
    ):
        """
        Static context manager method.
        
        Args:
            file_path: Path to DEM file
            mode: Access mode
            auto_save: Auto-save on exit
            
        Yields:
            DEM context
        """
        manager = DEMContextManager(file_path, mode, auto_save)
        yield manager.__enter__()
        manager.__exit__(None, None, None)


class MultiDEMContextManager:
    """
    Context manager for multiple DEM files.
    
    Provides unified management for processing multiple
    DEM files simultaneously.
    """
    
    def __init__(self, file_paths: list):
        """
        Initialize multi-DEM context manager.
        
        Args:
            file_paths: List of DEM file paths
        """
        self._file_paths = file_paths
        self._contexts: Dict[str, DEMContext] = {}
    
    def __enter__(self) -> Dict[str, DEMContext]:
        """Enter context and load all DEMs."""
        for file_path in self._file_paths:
            if os.path.exists(file_path):
                self._contexts[file_path] = DEMContext(file_path)
        
        return self._contexts
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and cleanup all DEMs."""
        self._contexts.clear()
    
    def get_context(self, file_path: str) -> Optional[DEMContext]:
        """
        Get context for specific file.
        
        Args:
            file_path: File path
            
        Returns:
            DEM context if found
        """
        return self._contexts.get(file_path)
    
    def get_all_data(self) -> Dict[str, np.ndarray]:
        """
        Get data from all DEMs.
        
        Returns:
            Dictionary of file paths to data arrays
        """
        return {
            path: ctx.data
            for path, ctx in self._contexts.items()
        }
