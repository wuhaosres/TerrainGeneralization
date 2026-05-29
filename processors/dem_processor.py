"""
DEM (Digital Elevation Model) processor implementation.
Provides comprehensive DEM processing capabilities.
"""

import os
import numpy as np
import cv2
import rasterio
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

try:
    import arcpy
    from arcpy.sa import *
    from arcpy.ddd import *
    ARCPY_AVAILABLE = True
except ImportError:
    ARCPY_AVAILABLE = False

from core.base import TerrainProcessorBase
from core.interfaces import IProcessor


class DEMProcessingType(Enum):
    """Enumeration of DEM processing types."""
    HILLSHADE = "hillshade"
    SLOPE = "slope"
    ASPECT = "aspect"
    CURVATURE = "curvature"
    CONTOUR = "contour"


@dataclass
class DEMConfiguration:
    """
    Configuration container for DEM processing parameters.
    Encapsulates all tunable parameters for DEM operations.
    """
    z_factor: float = 1.0
    azimuth: float = 315.0
    altitude: float = 45.0
    contour_interval: float = 100.0
    base_contour: float = 0.0
    output_resolution: Tuple[int, int] = (800, 800)
    output_dpi: int = 96
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.z_factor <= 0:
            raise ValueError("z_factor must be positive")
        if not (0 <= self.azimuth <= 360):
            raise ValueError("azimuth must be between 0 and 360")
        if not (0 <= self.altitude <= 90):
            raise ValueError("altitude must be between 0 and 90")
        if self.contour_interval <= 0:
            raise ValueError("contour_interval must be positive")


class DEMDataContainer:
    """
    Container for DEM data with metadata.
    Implements IDataContainer interface.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        extent: Optional[Tuple[float, float, float, float]] = None
    ):
        """
        Initialize DEM data container.
        
        Args:
            data: DEM data array
            metadata: Metadata dictionary
            extent: Spatial extent (xmin, ymin, xmax, ymax)
        """
        self._data = data
        self._metadata = metadata or {}
        self._extent = extent or (0, 0, data.shape[1], data.shape[0])
    
    def get_data(self) -> np.ndarray:
        """Get the underlying data array."""
        return self._data
    
    def set_data(self, data: np.ndarray) -> None:
        """Set the data array."""
        self._data = data
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata associated with the data."""
        return self._metadata.copy()
    
    def get_extent(self) -> Tuple[float, float, float, float]:
        """Get the spatial extent of the data."""
        return self._extent
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Get data shape."""
        return self._data.shape
    
    @property
    def dtype(self):
        """Get data type."""
        return self._data.dtype


class DEMProcessor(TerrainProcessorBase, IProcessor):
    """
    DEM processor with comprehensive processing capabilities.
    
    This class implements multiple processing operations for Digital
    Elevation Models including hillshade generation, contour extraction,
    and multi-angle visualization.
    
    Attributes:
        processor_type: Type identifier for the processor
        version: Version number
    """
    
    processor_type = 'dem_processor'
    version = '2.0'
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DEM processor.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self._dem_config = self._create_dem_config()
        self._data_container: Optional[DEMDataContainer] = None
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        pass
    
    def _create_dem_config(self) -> DEMConfiguration:
        """Create DEM configuration from config dict."""
        return DEMConfiguration(
            z_factor=self._config.get('z_factor', 1.0),
            azimuth=self._config.get('azimuth', 315.0),
            altitude=self._config.get('altitude', 45.0),
            contour_interval=self._config.get('contour_interval', 100.0),
            base_contour=self._config.get('base_contour', 0.0),
            output_resolution=self._config.get('output_resolution', (800, 800)),
            output_dpi=self._config.get('output_dpi', 96)
        )
    
    def load_data(self, file_path: str) -> DEMDataContainer:
        """
        Load DEM data from file.
        
        Args:
            file_path: Path to DEM file
            
        Returns:
            DEM data container
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"DEM file not found: {file_path}")
        
        with rasterio.open(file_path) as dataset:
            data = dataset.read(1)
            metadata = {
                'crs': dataset.crs,
                'transform': dataset.transform,
                'nodata': dataset.nodata
            }
            bounds = dataset.bounds
            extent = (bounds.left, bounds.bottom, bounds.right, bounds.top)
        
        self._data_container = DEMDataContainer(
            data=data,
            metadata=metadata,
            extent=extent
        )
        
        return self._data_container
    
    def process(self, data: Any, **kwargs) -> Any:
        """
        Process DEM data.
        
        Args:
            data: Input DEM data
            **kwargs: Processing parameters
            
        Returns:
            Processed data
        """
        processing_type = kwargs.get('processing_type', DEMProcessingType.HILLSHADE)
        
        if processing_type == DEMProcessingType.HILLSHADE:
            return self.generate_hillshade(data, **kwargs)
        elif processing_type == DEMProcessingType.CONTOUR:
            return self.extract_contours(data, **kwargs)
        elif processing_type == DEMProcessingType.CURVATURE:
            return self.compute_curvature(data)
        else:
            raise ValueError(f"Unsupported processing type: {processing_type}")
    
    def save_data(self, data: Any, file_path: str) -> None:
        """
        Save processed data to file.
        
        Args:
            data: Data to save
            file_path: Output file path
        """
        if isinstance(data, np.ndarray):
            cv2.imwrite(file_path, data)
        else:
            raise TypeError("Data must be numpy array")
    
    def generate_hillshade(
        self,
        dem_data: np.ndarray,
        azimuth: Optional[float] = None,
        altitude: Optional[float] = None,
        z_factor: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate hillshade from DEM.
        
        Args:
            dem_data: DEM data array
            azimuth: Azimuth angle (optional)
            altitude: Altitude angle (optional)
            z_factor: Z factor (optional)
            
        Returns:
            Hillshade array
        """
        if not ARCPY_AVAILABLE:
            return self._generate_hillshade_numpy(
                dem_data,
                azimuth or self._dem_config.azimuth,
                altitude or self._dem_config.altitude,
                z_factor or self._dem_config.z_factor
            )
        
        return self._generate_hillshade_arcpy(dem_data, azimuth, altitude, z_factor)
    
    def _generate_hillshade_numpy(
        self,
        dem_data: np.ndarray,
        azimuth: float,
        altitude: float,
        z_factor: float
    ) -> np.ndarray:
        """
        Generate hillshade using numpy (fallback method).
        
        Args:
            dem_data: DEM data
            azimuth: Azimuth angle
            altitude: Altitude angle
            z_factor: Z factor
            
        Returns:
            Hillshade array
        """
        dem_data = dem_data.astype(np.float32)
        
        x, y = np.gradient(dem_data * z_factor)
        
        azimuth_rad = np.radians(azimuth)
        altitude_rad = np.radians(altitude)
        
        slope = np.pi / 2.0 - np.arctan(np.sqrt(x * x + y * y))
        aspect = np.arctan2(-x, y)
        
        hillshade = (
            np.cos(altitude_rad) * np.cos(slope) +
            np.sin(altitude_rad) * np.sin(slope) * np.cos(azimuth_rad - aspect)
        )
        
        hillshade = np.clip(hillshade * 255, 0, 255).astype(np.uint8)
        
        return hillshade
    
    def _generate_hillshade_arcpy(
        self,
        dem_data: np.ndarray,
        azimuth: Optional[float],
        altitude: Optional[float],
        z_factor: Optional[float]
    ) -> np.ndarray:
        """
        Generate hillshade using ArcPy.
        
        Args:
            dem_data: DEM data
            azimuth: Azimuth angle
            altitude: Altitude angle
            z_factor: Z factor
            
        Returns:
            Hillshade array
        """
        pass
    
    def generate_multi_angle_hillshade(
        self,
        dem_data: np.ndarray,
        angles: List[float] = [0, 120, 240],
        altitude: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate multi-angle hillshade for RGB visualization.
        
        Args:
            dem_data: DEM data array
            angles: List of azimuth angles
            altitude: Altitude angle
            
        Returns:
            RGB hillshade array
        """
        hillshades = []
        
        for angle in angles:
            hs = self.generate_hillshade(
                dem_data,
                azimuth=angle,
                altitude=altitude
            )
            hillshades.append(hs)
        
        return np.dstack(hillshades)
    
    def compute_curvature(self, dem_data: np.ndarray) -> np.ndarray:
        """
        Compute terrain curvature from DEM.
        
        Args:
            dem_data: DEM data array
            
        Returns:
            Curvature array
        """
        dy, dx = np.gradient(dem_data)
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
    
    def extract_contours(
        self,
        dem_data: np.ndarray,
        interval: Optional[float] = None,
        base: Optional[float] = None
    ) -> List[np.ndarray]:
        """
        Extract contour lines from DEM.
        
        Args:
            dem_data: DEM data array
            interval: Contour interval
            base: Base contour value
            
        Returns:
            List of contour arrays
        """
        interval = interval or self._dem_config.contour_interval
        base = base or self._dem_config.base_contour
        
        contours = []
        levels = np.arange(
            dem_data.min(),
            dem_data.max(),
            interval
        )
        
        for level in levels:
            level_contours = self._extract_contour_at_level(dem_data, level)
            contours.extend(level_contours)
        
        return contours
    
    def _extract_contour_at_level(
        self,
        dem_data: np.ndarray,
        level: float
    ) -> List[np.ndarray]:
        """
        Extract contour at specific level.
        
        Args:
            dem_data: DEM data
            level: Contour level
            
        Returns:
            List of contour arrays
        """
        binary = (dem_data >= level).astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        return contours
    
    def get_extent(self, file_path: str) -> Tuple[float, float, float, float]:
        """
        Get spatial extent of DEM file.
        
        Args:
            file_path: Path to DEM file
            
        Returns:
            Extent tuple (xmin, ymin, xmax, ymax)
        """
        with rasterio.open(file_path) as dataset:
            bounds = dataset.bounds
            return (bounds.left, bounds.bottom, bounds.right, bounds.top)
    
    def get_center(self, file_path: str) -> Tuple[float, float]:
        """
        Get center coordinates of DEM file.
        
        Args:
            file_path: Path to DEM file
            
        Returns:
            Center coordinates (x, y)
        """
        extent = self.get_extent(file_path)
        return (
            (extent[0] + extent[2]) / 2,
            (extent[1] + extent[3]) / 2
        )
    
    def compute_z_factor(self, latitude: float) -> float:
        """
        Compute Z factor based on latitude.
        
        Args:
            latitude: Latitude in degrees
            
        Returns:
            Z factor value
        """
        zf_list = [
            0.00000898, 0.00000912, 0.00000956,
            0.00001036, 0.00001171, 0.00001395,
            0.00001792, 0.00002619, 0.00005156
        ]
        
        try:
            lat_index = int(latitude / 10)
            if 0 <= lat_index < len(zf_list):
                return zf_list[lat_index]
        except:
            pass
        
        return 1.0
