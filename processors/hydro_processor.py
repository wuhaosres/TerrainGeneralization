"""
Hydrological analysis processor implementation.
Provides hydrological feature extraction and analysis.
"""

import os
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass

try:
    import arcpy
    from arcpy.sa import *
    ARCPY_AVAILABLE = True
except ImportError:
    ARCPY_AVAILABLE = False

from core.base import TerrainProcessorBase


@dataclass
class HydroConfiguration:
    """
    Configuration container for hydrological analysis parameters.
    """
    flow_accumulation_threshold: int = 1000
    fill_depressions: bool = True
    snap_pour_point: bool = True
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.flow_accumulation_threshold <= 0:
            raise ValueError("flow_accumulation_threshold must be positive")


class HydrologicalProcessor(TerrainProcessorBase):
    """
    Hydrological analysis processor.
    
    Provides comprehensive hydrological feature extraction including
    flow direction, flow accumulation, stream networks, and watersheds.
    
    Attributes:
        processor_type: Type identifier for the processor
        version: Version number
    """
    
    processor_type = 'hydro_processor'
    version = '2.0'
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize hydrological processor.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self._hydro_config = self._create_hydro_config()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        pass
    
    def _create_hydro_config(self) -> HydroConfiguration:
        """Create hydrological configuration from config dict."""
        return HydroConfiguration(
            flow_accumulation_threshold=self._config.get(
                'flow_accumulation_threshold', 1000
            ),
            fill_depressions=self._config.get('fill_depressions', True),
            snap_pour_point=self._config.get('snap_pour_point', True)
        )
    
    def process(self, *args, **kwargs) -> Any:
        """Main processing method."""
        dem_path = kwargs.get('dem_path')
        output_path = kwargs.get('output_path')
        
        if dem_path is None or output_path is None:
            raise ValueError("dem_path and output_path are required")
        
        return self.extract_stream_network(dem_path, output_path)
    
    def extract_stream_network(
        self,
        dem_path: str,
        output_path: str
    ) -> str:
        """
        Extract stream network from DEM.
        
        Args:
            dem_path: Path to input DEM
            output_path: Output path for stream network
            
        Returns:
            Path to output stream network
        """
        if not ARCPY_AVAILABLE:
            raise RuntimeError("ArcPy is required for hydrological analysis")
        
        arcpy.env.workspace = os.path.dirname(dem_path)
        arcpy.env.overwriteOutput = True
        
        filled_dem = self._fill_depressions(dem_path)
        
        flow_direction = self._calculate_flow_direction(filled_dem)
        
        flow_accumulation = self._calculate_flow_accumulation(flow_direction)
        
        stream_network = self._extract_streams(
            flow_accumulation,
            flow_direction,
            output_path
        )
        
        return stream_network
    
    def _fill_depressions(self, dem_path: str) -> Any:
        """
        Fill depressions in DEM.
        
        Args:
            dem_path: Path to DEM
            
        Returns:
            Filled DEM
        """
        if self._hydro_config.fill_depressions:
            return Fill(dem_path)
        return dem_path
    
    def _calculate_flow_direction(self, dem: Any) -> Any:
        """
        Calculate flow direction.
        
        Args:
            dem: Input DEM
            
        Returns:
            Flow direction raster
        """
        return FlowDirection(dem, "NORMAL")
    
    def _calculate_flow_accumulation(self, flow_direction: Any) -> Any:
        """
        Calculate flow accumulation.
        
        Args:
            flow_direction: Flow direction raster
            
        Returns:
            Flow accumulation raster
        """
        return FlowAccumulation(flow_direction)
    
    def _extract_streams(
        self,
        flow_accumulation: Any,
        flow_direction: Any,
        output_path: str
    ) -> str:
        """
        Extract stream network.
        
        Args:
            flow_accumulation: Flow accumulation raster
            flow_direction: Flow direction raster
            output_path: Output path
            
        Returns:
            Path to stream network
        """
        threshold = self._hydro_config.flow_accumulation_threshold
        stream = Con(flow_accumulation > threshold, 1)
        
        stream_link = StreamLink(stream, flow_direction)
        
        StreamToFeature(
            stream_link,
            flow_direction,
            output_path,
            "NO_SIMPLIFY"
        )
        
        return output_path
    
    def generate_contours(
        self,
        dem_path: str,
        output_path: str,
        interval: float = 100.0,
        base: float = 0.0,
        z_factor: float = 1.0
    ) -> str:
        """
        Generate contour lines from DEM.
        
        Args:
            dem_path: Path to input DEM
            output_path: Output path for contours
            interval: Contour interval
            base: Base contour value
            z_factor: Z factor for unit conversion
            
        Returns:
            Path to output contours
        """
        if not ARCPY_AVAILABLE:
            raise RuntimeError("ArcPy is required for contour generation")
        
        arcpy.env.workspace = os.path.dirname(dem_path)
        arcpy.env.overwriteOutput = True
        
        try:
            Contour(
                in_raster=dem_path,
                out_polyline_features=output_path,
                contour_interval=interval,
                base_contour=base,
                z_factor=z_factor
            )
        except Exception as e:
            print(f"Contour generation failed: {e}")
        
        return output_path
