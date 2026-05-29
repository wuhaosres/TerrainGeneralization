"""
Main entry point for terrain generalization framework.
Provides high-level API for terrain processing workflows.
"""

import os
from typing import Optional, Dict, Any, List
import numpy as np

from core.base import TerrainProcessorBase
from factories.processor_factory import ProcessorFactory, ProcessorBuilder
from evaluators.metrics import TerrainEvaluator
from visualizers.dem_visualizer import DEMVisualizer
from contexts.dem_context import DEMContextManager
from algorithms.slic import TerrainAwareSLIC
from algorithms.tps import ThinPlateSplineInterpolator
from processors.dem_processor import DEMProcessor
from processors.hydro_processor import HydrologicalProcessor


class TerrainGeneralizationPipeline:
    """
    High-level pipeline for terrain generalization.
    
    Orchestrates the complete workflow from DEM loading to
    generalization, evaluation, and visualization.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize terrain generalization pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self._config = config or {}
        self._processors: Dict[str, TerrainProcessorBase] = {}
        self._results: Dict[str, Any] = {}
        
        self._initialize_processors()
    
    def _initialize_processors(self) -> None:
        """Initialize all required processors."""
        self._processors['dem'] = DEMProcessor(self._config.get('dem', {}))
        self._processors['hydro'] = HydrologicalProcessor(self._config.get('hydro', {}))
        self._processors['slic'] = TerrainAwareSLIC(self._config.get('slic', {}))
        self._processors['tps'] = ThinPlateSplineInterpolator(self._config.get('tps', {}))
        self._processors['evaluator'] = TerrainEvaluator(self._config.get('evaluator', {}))
        self._processors['visualizer'] = DEMVisualizer(self._config.get('visualizer', {}))
    
    def execute(
        self,
        dem_path: str,
        output_dir: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute complete terrain generalization pipeline.
        
        Args:
            dem_path: Path to input DEM
            output_dir: Output directory
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        with DEMContextManager(dem_path) as dem_context:
            dem_data = dem_context.data
            
            self._results['original_dem'] = dem_data.copy()
            
            hillshade = self._processors['dem'].generate_hillshade(dem_data)
            self._results['hillshade'] = hillshade
            
            multi_hillshade = self._processors['dem'].generate_multi_angle_hillshade(
                dem_data
            )
            self._results['multi_hillshade'] = multi_hillshade
            
            curvature = self._processors['dem'].compute_curvature(dem_data)
            self._results['curvature'] = curvature
            
            if self._config.get('extract_streams', False):
                stream_path = os.path.join(output_dir, 'streams.shp')
                self._processors['hydro'].extract_stream_network(
                    dem_path,
                    stream_path
                )
                self._results['stream_path'] = stream_path
            
            if self._config.get('extract_contours', False):
                contour_path = os.path.join(output_dir, 'contours.shp')
                self._processors['hydro'].generate_contours(
                    dem_path,
                    contour_path
                )
                self._results['contour_path'] = contour_path
            
            if self._config.get('perform_segmentation', False):
                segments = self._processors['slic'].process(
                    multi_hillshade,
                    curvature
                )
                self._results['segments'] = segments
            
            if self._config.get('visualize', True):
                vis_path = os.path.join(output_dir, 'visualization.png')
                self._processors['visualizer'].visualize(
                    dem_data,
                    visualization_type='3d'
                )
                self._processors['visualizer'].save_visualization(vis_path)
                self._results['visualization_path'] = vis_path
        
        return self._results
    
    def get_processor(self, name: str) -> Optional[TerrainProcessorBase]:
        """
        Get processor by name.
        
        Args:
            name: Processor name
            
        Returns:
            Processor instance if found
        """
        return self._processors.get(name)
    
    def get_results(self) -> Dict[str, Any]:
        """Get all results."""
        return self._results.copy()


class TerrainAnalysisAPI:
    """
    Simplified API for terrain analysis operations.
    
    Provides convenient methods for common terrain processing tasks
    without requiring detailed knowledge of the framework architecture.
    """
    
    @staticmethod
    def load_dem(file_path: str) -> np.ndarray:
        """
        Load DEM from file.
        
        Args:
            file_path: Path to DEM file
            
        Returns:
            DEM data array
        """
        with DEMContextManager(file_path) as context:
            return context.data.copy()
    
    @staticmethod
    def generate_hillshade(
        dem_data: np.ndarray,
        azimuth: float = 315.0,
        altitude: float = 45.0
    ) -> np.ndarray:
        """
        Generate hillshade from DEM.
        
        Args:
            dem_data: DEM data array
            azimuth: Azimuth angle
            altitude: Altitude angle
            
        Returns:
            Hillshade array
        """
        processor = DEMProcessor({
            'azimuth': azimuth,
            'altitude': altitude
        })
        return processor.generate_hillshade(dem_data)
    
    @staticmethod
    def compute_curvature(dem_data: np.ndarray) -> np.ndarray:
        """
        Compute terrain curvature.
        
        Args:
            dem_data: DEM data array
            
        Returns:
            Curvature array
        """
        processor = DEMProcessor()
        return processor.compute_curvature(dem_data)
    
    @staticmethod
    def segment_terrain(
        image: np.ndarray,
        terrain_features: np.ndarray,
        num_segments: int = 200
    ) -> np.ndarray:
        """
        Segment terrain using SLIC algorithm.
        
        Args:
            image: Input image
            terrain_features: Terrain feature map
            num_segments: Number of segments
            
        Returns:
            Segmentation labels
        """
        processor = TerrainAwareSLIC({'num_segments': num_segments})
        return processor.process(image, terrain_features)
    
    @staticmethod
    def interpolate_surface(
        points: np.ndarray,
        values: np.ndarray,
        regularization: float = 1e-6
    ):
        """
        Interpolate surface using TPS.
        
        Args:
            points: Point coordinates
            values: Values at points
            regularization: Regularization parameter
            
        Returns:
            TPS model
        """
        processor = ThinPlateSplineInterpolator({
            'regularization': regularization
        })
        return processor.process(points, values)
    
    @staticmethod
    def evaluate_terrain(
        reference: np.ndarray,
        target: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate terrain using multiple metrics.
        
        Args:
            reference: Reference DEM
            target: Target DEM
            
        Returns:
            Dictionary of metric values
        """
        evaluator = TerrainEvaluator()
        results = evaluator.process(reference, target)
        
        return {
            name: result.value
            for name, result in results.items()
        }
    
    @staticmethod
    def visualize_3d(
        dem_data: np.ndarray,
        output_path: Optional[str] = None
    ):
        """
        Create 3D visualization of DEM.
        
        Args:
            dem_data: DEM data array
            output_path: Output file path (optional)
            
        Returns:
            Matplotlib figure
        """
        visualizer = DEMVisualizer()
        fig = visualizer.visualize(dem_data, visualization_type='3d')
        
        if output_path:
            visualizer.save_visualization(output_path)
        
        return fig


def create_pipeline(config: Optional[Dict[str, Any]] = None) -> TerrainGeneralizationPipeline:
    """
    Factory function to create terrain generalization pipeline.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        Pipeline instance
    """
    return TerrainGeneralizationPipeline(config)


def quick_process(
    dem_path: str,
    output_dir: str,
    operations: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Quick processing function for common operations.
    
    Args:
        dem_path: Path to input DEM
        output_dir: Output directory
        operations: List of operations to perform
        
    Returns:
        Dictionary of results
    """
    operations = operations or ['hillshade', 'curvature', 'visualize']
    
    config = {
        'extract_streams': 'streams' in operations,
        'extract_contours': 'contours' in operations,
        'perform_segmentation': 'segmentation' in operations,
        'visualize': 'visualize' in operations
    }
    
    pipeline = create_pipeline(config)
    return pipeline.execute(dem_path, output_dir)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python main.py <dem_path> <output_dir>")
        sys.exit(1)
    
    dem_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    results = quick_process(dem_path, output_dir)
    
    print("Processing completed successfully!")
    print(f"Results saved to: {output_dir}")
    for key, value in results.items():
        if not isinstance(value, np.ndarray):
            print(f"  {key}: {value}")
