"""
DEM visualization implementation.
Provides comprehensive visualization capabilities for terrain data.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import rasterio

try:
    from qgis.core import (
        QgsRasterLayer, QgsVectorLayer, QgsProject,
        QgsSingleBandPseudoColorRenderer, QgsColorRampShader,
        QgsRasterShader, QgsSimpleLineSymbolLayer, QgsLineSymbol,
        QgsSingleSymbolRenderer, QgsMapSettings, QgsMapRendererParallelJob,
        QgsStyle
    )
    from qgis.PyQt.QtGui import QColor, QImage, QPainter
    from qgis.PyQt.QtCore import QSize
    QGIS_AVAILABLE = True
except ImportError:
    QGIS_AVAILABLE = False

from core.base import TerrainProcessorBase
from core.interfaces import IVisualizer


@dataclass
class VisualizationConfig:
    """
    Configuration container for visualization parameters.
    """
    output_size: Tuple[int, int] = (800, 800)
    output_dpi: int = 96
    colormap: str = 'terrain'
    show_axes: bool = False
    show_colorbar: bool = True
    elevation_range: Optional[Tuple[float, float]] = None
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.output_size[0] <= 0 or self.output_size[1] <= 0:
            raise ValueError("output_size must be positive")
        if self.output_dpi <= 0:
            raise ValueError("output_dpi must be positive")


class DEMVisualizer(TerrainProcessorBase, IVisualizer):
    """
    DEM visualization processor.
    
    Provides comprehensive visualization capabilities including
    3D surface rendering, 2D color mapping, and professional
    cartographic output using QGIS.
    
    Attributes:
        processor_type: Type identifier for the processor
        version: Version number
    """
    
    processor_type = 'visualizer'
    version = '2.0'
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DEM visualizer.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self._vis_config = self._create_vis_config()
        self._current_visualization = None
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        pass
    
    def _create_vis_config(self) -> VisualizationConfig:
        """Create visualization configuration."""
        return VisualizationConfig(
            output_size=self._config.get('output_size', (800, 800)),
            output_dpi=self._config.get('output_dpi', 96),
            colormap=self._config.get('colormap', 'terrain'),
            show_axes=self._config.get('show_axes', False),
            show_colorbar=self._config.get('show_colorbar', True),
            elevation_range=self._config.get('elevation_range', None)
        )
    
    def process(self, *args, **kwargs) -> Any:
        """Main processing method."""
        return self.visualize(*args, **kwargs)
    
    def visualize(
        self,
        data: Any,
        visualization_type: str = '3d',
        **kwargs
    ) -> Any:
        """
        Visualize terrain data.
        
        Args:
            data: Terrain data to visualize
            visualization_type: Type of visualization ('3d', '2d', 'qgis')
            **kwargs: Visualization parameters
            
        Returns:
            Visualization object
        """
        if visualization_type == '3d':
            return self._visualize_3d(data, **kwargs)
        elif visualization_type == '2d':
            return self._visualize_2d(data, **kwargs)
        elif visualization_type == 'qgis':
            return self._visualize_qgis(data, **kwargs)
        else:
            raise ValueError(f"Unsupported visualization type: {visualization_type}")
    
    def save_visualization(self, output_path: str) -> None:
        """
        Save visualization to file.
        
        Args:
            output_path: Output file path
        """
        if self._current_visualization is None:
            raise RuntimeError("No visualization to save")
        
        if isinstance(self._current_visualization, plt.Figure):
            self._current_visualization.savefig(
                output_path,
                dpi=self._vis_config.output_dpi,
                bbox_inches='tight'
            )
        else:
            raise TypeError("Unsupported visualization type")
    
    def _visualize_3d(
        self,
        dem_data: np.ndarray,
        **kwargs
    ) -> plt.Figure:
        """
        Create 3D surface visualization.
        
        Args:
            dem_data: DEM data array
            **kwargs: Additional parameters
            
        Returns:
            Matplotlib figure
        """
        rows, cols = dem_data.shape
        
        x = np.arange(cols)
        y = np.arange(rows)
        x, y = np.meshgrid(x, y)
        
        z = dem_data
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(
            x, y, z,
            cmap=self._vis_config.colormap,
            linewidth=0,
            antialiased=False
        )
        
        if self._vis_config.show_colorbar:
            fig.colorbar(
                surf,
                shrink=0.5,
                aspect=5,
                label='Elevation (m)'
            )
        
        ax.set_title("DEM 3D Visualization")
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.set_zlabel("Elevation (m)")
        
        if self._vis_config.elevation_range:
            ax.set_zlim(self._vis_config.elevation_range)
        
        if not self._vis_config.show_axes:
            ax.set_axis_off()
        
        self._current_visualization = fig
        
        return fig
    
    def _visualize_2d(
        self,
        dem_data: np.ndarray,
        **kwargs
    ) -> plt.Figure:
        """
        Create 2D color visualization.
        
        Args:
            dem_data: DEM data array
            **kwargs: Additional parameters
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        im = ax.imshow(
            dem_data,
            cmap=self._vis_config.colormap
        )
        
        if self._vis_config.show_colorbar:
            fig.colorbar(im, ax=ax, label='Elevation (m)')
        
        ax.set_title("DEM 2D Visualization")
        
        if not self._vis_config.show_axes:
            ax.axis('off')
        
        self._current_visualization = fig
        
        return fig
    
    def _visualize_qgis(
        self,
        dem_path: str,
        contour_path: Optional[str] = None,
        stream_path: Optional[str] = None,
        output_path: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Create professional cartographic visualization using QGIS.
        
        Args:
            dem_path: Path to DEM file
            contour_path: Path to contour shapefile
            stream_path: Path to stream shapefile
            output_path: Output image path
            **kwargs: Additional parameters
            
        Returns:
            Output path if successful
        """
        if not QGIS_AVAILABLE:
            raise RuntimeError("QGIS is required for professional cartography")
        
        QgsProject.instance().removeAllMapLayers()
        
        dem_layer = QgsRasterLayer(dem_path, "DEM", "gdal")
        if not dem_layer.isValid():
            raise RuntimeError("Failed to load DEM layer")
        
        QgsProject.instance().addMapLayer(dem_layer)
        
        self._apply_pseudocolor_renderer(dem_layer)
        
        layers = [dem_layer]
        
        if contour_path and os.path.exists(contour_path):
            contour_layer = QgsVectorLayer(contour_path, "Contours", "ogr")
            if contour_layer.isValid():
                self._apply_line_style(contour_layer, QColor(0, 0, 0), 0.2)
                QgsProject.instance().addMapLayer(contour_layer)
                layers.append(contour_layer)
        
        if stream_path and os.path.exists(stream_path):
            stream_layer = QgsVectorLayer(stream_path, "Streams", "ogr")
            if stream_layer.isValid():
                self._apply_line_style(stream_layer, QColor(231, 76, 60), 0.66)
                QgsProject.instance().addMapLayer(stream_layer)
                layers.append(stream_layer)
        
        if output_path:
            self._render_map(layers, dem_layer.extent(), output_path)
            return output_path
        
        return None
    
    def _apply_pseudocolor_renderer(self, layer: Any) -> None:
        """Apply pseudocolor renderer to raster layer."""
        provider = layer.dataProvider()
        stats = provider.bandStatistics(1)
        min_val = stats.minimumValue
        max_val = stats.maximumValue
        
        style = QgsStyle.defaultStyle()
        spectral_ramp = style.colorRamp("Spectral")
        
        shader = QgsColorRampShader()
        shader.setSourceColorRamp(spectral_ramp)
        shader.setColorRampType(QgsColorRampShader.Interpolated)
        
        steps = 5
        color_items = []
        for i in range(steps + 1):
            val = min_val + (max_val - min_val) * i / steps
            color = spectral_ramp.color(1.0 - i / steps)
            color_items.append(QgsColorRampShader.ColorRampItem(val, color))
        
        shader.setColorRampItemList(color_items)
        
        raster_shader = QgsRasterShader()
        raster_shader.setRasterShaderFunction(shader)
        
        renderer = QgsSingleBandPseudoColorRenderer(provider, 1, raster_shader)
        layer.setRenderer(renderer)
        layer.triggerRepaint()
    
    def _apply_line_style(
        self,
        layer: Any,
        color: Any,
        width: float
    ) -> None:
        """Apply line style to vector layer."""
        line_symbol = QgsLineSymbol()
        line_symbol.deleteSymbolLayer(0)
        
        line_layer_style = QgsSimpleLineSymbolLayer()
        line_layer_style.setColor(color)
        line_layer_style.setWidth(width)
        line_symbol.appendSymbolLayer(line_layer_style)
        
        layer.setRenderer(QgsSingleSymbolRenderer(line_symbol))
        layer.triggerRepaint()
    
    def _render_map(
        self,
        layers: list,
        extent: Any,
        output_path: str
    ) -> None:
        """Render map to image."""
        settings = QgsMapSettings()
        settings.setLayers(layers)
        settings.setExtent(extent)
        settings.setOutputSize(QSize(*self._vis_config.output_size))
        settings.setOutputDpi(self._vis_config.output_dpi)
        
        render_job = QgsMapRendererParallelJob(settings)
        render_job.start()
        render_job.waitForFinished()
        
        img = render_job.renderedImage()
        img.save(output_path)
