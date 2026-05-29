"""
Terrain evaluation metrics implementation.
Provides comprehensive metrics for terrain generalization assessment.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from abc import abstractmethod
import geopandas as gpd
from shapely.ops import unary_union, polygonize
from shapely.geometry import LineString
from skimage.metrics import mean_squared_error, structural_similarity as ssim
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops

from core.base import TerrainProcessorBase
from core.interfaces import IEvaluator


@dataclass
class MetricResult:
    """
    Container for metric evaluation results.
    """
    name: str
    value: float
    unit: str = ""
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'description': self.description
        }


class BaseMetric(IEvaluator):
    """
    Abstract base class for terrain metrics.
    Implements the Strategy pattern for metric computation.
    """
    
    @abstractmethod
    def compute(
        self,
        reference: Any,
        target: Any,
        **kwargs
    ) -> MetricResult:
        """
        Compute metric between reference and target.
        
        Args:
            reference: Reference data
            target: Target data
            **kwargs: Additional parameters
            
        Returns:
            Metric result
        """
        pass
    
    def compute_metric(self, reference: Any, target: Any) -> float:
        """Implement IEvaluator interface."""
        result = self.compute(reference, target)
        return result.value
    
    @abstractmethod
    def get_metric_name(self) -> str:
        """Get metric name."""
        pass
    
    @abstractmethod
    def get_metric_range(self) -> Tuple[float, float]:
        """Get valid range for metric."""
        pass


class RMSEMetric(BaseMetric):
    """
    Root Mean Square Error metric for DEM comparison.
    """
    
    def compute(
        self,
        reference: np.ndarray,
        target: np.ndarray,
        **kwargs
    ) -> MetricResult:
        """
        Compute RMSE between two DEM arrays.
        
        Args:
            reference: Reference DEM
            target: Target DEM
            **kwargs: Additional parameters
            
        Returns:
            RMSE metric result
        """
        mse = mean_squared_error(reference, target)
        rmse = np.sqrt(mse)
        
        return MetricResult(
            name="RMSE",
            value=float(rmse),
            unit="meters",
            description="Root Mean Square Error"
        )
    
    def get_metric_name(self) -> str:
        return "RMSE"
    
    def get_metric_range(self) -> Tuple[float, float]:
        return (0.0, float('inf'))


class SMRMetric(BaseMetric):
    """
    Stream Match Ratio metric for hydrological feature comparison.
    """
    
    def compute(
        self,
        reference: Any,
        target: Any,
        buffer_radius: float = 40.0,
        **kwargs
    ) -> MetricResult:
        """
        Compute stream match ratio.
        
        Args:
            reference: Reference stream geometry
            target: Target stream geometry
            buffer_radius: Buffer radius for matching
            **kwargs: Additional parameters
            
        Returns:
            SMR metric result
        """
        target_buffer = target.buffer(buffer_radius)
        
        matched_part = reference.intersection(target_buffer)
        
        total_length = reference.length
        matched_length = matched_part.length
        
        smr = matched_length / total_length if total_length > 0 else 0.0
        
        return MetricResult(
            name="SMR",
            value=float(smr),
            unit="ratio",
            description="Stream Match Ratio"
        )
    
    def get_metric_name(self) -> str:
        return "SMR"
    
    def get_metric_range(self) -> Tuple[float, float]:
        return (0.0, 1.0)


class SSIMetric(BaseMetric):
    """
    Structural Similarity Index metric for DEM comparison.
    """
    
    def compute(
        self,
        reference: np.ndarray,
        target: np.ndarray,
        **kwargs
    ) -> MetricResult:
        """
        Compute SSIM between two DEM arrays.
        
        Args:
            reference: Reference DEM
            target: Target DEM
            **kwargs: Additional parameters
            
        Returns:
            SSIM metric result
        """
        ref_norm = self._normalize(reference)
        tgt_norm = self._normalize(target)
        
        score, _ = ssim(
            ref_norm,
            tgt_norm,
            full=True,
            data_range=1.0
        )
        
        return MetricResult(
            name="SSIM",
            value=float(score),
            unit="index",
            description="Structural Similarity Index"
        )
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range."""
        image = image.astype(np.float32)
        min_val = np.min(image)
        max_val = np.max(image)
        
        if max_val - min_val == 0:
            return np.zeros_like(image)
        
        return (image - min_val) / (max_val - min_val)
    
    def get_metric_name(self) -> str:
        return "SSIM"
    
    def get_metric_range(self) -> Tuple[float, float]:
        return (0.0, 1.0)


class RoughnessMetric(BaseMetric):
    """
    Terrain roughness metric based on gradient analysis.
    """
    
    def compute(
        self,
        reference: np.ndarray,
        target: Optional[np.ndarray] = None,
        **kwargs
    ) -> MetricResult:
        """
        Compute terrain roughness.
        
        Args:
            reference: DEM array
            target: Not used for this metric
            **kwargs: Additional parameters
            
        Returns:
            Roughness metric result
        """
        gx = ndimage.sobel(reference, axis=0, mode='constant')
        gy = ndimage.sobel(reference, axis=1, mode='constant')
        
        gradient_magnitude = np.sqrt(gx ** 2 + gy ** 2)
        
        roughness = np.nanmean(gradient_magnitude)
        
        return MetricResult(
            name="Roughness",
            value=float(roughness),
            unit="gradient",
            description="Terrain Roughness"
        )
    
    def get_metric_name(self) -> str:
        return "Roughness"
    
    def get_metric_range(self) -> Tuple[float, float]:
        return (0.0, float('inf'))


class TRIMetric(BaseMetric):
    """
    Terrain Ruggedness Index metric.
    """
    
    def compute(
        self,
        reference: np.ndarray,
        target: Optional[np.ndarray] = None,
        window_size: int = 3,
        **kwargs
    ) -> MetricResult:
        """
        Compute Terrain Ruggedness Index.
        
        Args:
            reference: DEM array
            target: Not used for this metric
            window_size: Window size for TRI calculation
            **kwargs: Additional parameters
            
        Returns:
            TRI metric result
        """
        dem = np.nan_to_num(reference, nan=0.0)
        
        def tri_kernel(window):
            center = window[4]
            diff_sum = np.sum((window - center) ** 2)
            return np.sqrt(diff_sum)
        
        tri_map = ndimage.generic_filter(
            dem,
            tri_kernel,
            size=window_size,
            mode='constant',
            cval=0
        )
        
        tri_mean = np.nanmean(tri_map)
        
        return MetricResult(
            name="TRI",
            value=float(tri_mean),
            unit="meters",
            description="Terrain Ruggedness Index"
        )
    
    def get_metric_name(self) -> str:
        return "TRI"
    
    def get_metric_range(self) -> Tuple[float, float]:
        return (0.0, float('inf'))


class MetricCollection:
    """
    Collection of metrics for comprehensive evaluation.
    Implements the Composite pattern for metric aggregation.
    """
    
    def __init__(self):
        """Initialize metric collection."""
        self._metrics: Dict[str, BaseMetric] = {}
    
    def add_metric(self, name: str, metric: BaseMetric) -> None:
        """
        Add a metric to the collection.
        
        Args:
            name: Metric name
            metric: Metric instance
        """
        self._metrics[name] = metric
    
    def remove_metric(self, name: str) -> None:
        """
        Remove a metric from the collection.
        
        Args:
            name: Metric name to remove
        """
        if name in self._metrics:
            del self._metrics[name]
    
    def compute_all(
        self,
        reference: Any,
        target: Any,
        **kwargs
    ) -> Dict[str, MetricResult]:
        """
        Compute all metrics in the collection.
        
        Args:
            reference: Reference data
            target: Target data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of metric results
        """
        results = {}
        
        for name, metric in self._metrics.items():
            try:
                result = metric.compute(reference, target, **kwargs)
                results[name] = result
            except Exception as e:
                results[name] = MetricResult(
                    name=name,
                    value=float('nan'),
                    description=f"Error: {str(e)}"
                )
        
        return results
    
    def get_metric_names(self) -> List[str]:
        """Get list of metric names."""
        return list(self._metrics.keys())


class TerrainEvaluator(TerrainProcessorBase):
    """
    Comprehensive terrain evaluator.
    
    Provides unified interface for computing multiple terrain metrics
    and generating evaluation reports.
    
    Attributes:
        processor_type: Type identifier for the processor
        version: Version number
    """
    
    processor_type = 'evaluator'
    version = '2.0'
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize terrain evaluator.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self._metric_collection = self._create_metric_collection()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        pass
    
    def _create_metric_collection(self) -> MetricCollection:
        """Create default metric collection."""
        collection = MetricCollection()
        
        collection.add_metric('rmse', RMSEMetric())
        collection.add_metric('smr', SMRMetric())
        collection.add_metric('ssim', SSIMetric())
        collection.add_metric('roughness', RoughnessMetric())
        collection.add_metric('tri', TRIMetric())
        
        return collection
    
    def process(
        self,
        reference: Any,
        target: Any,
        **kwargs
    ) -> Dict[str, MetricResult]:
        """
        Evaluate terrain data.
        
        Args:
            reference: Reference terrain data
            target: Target terrain data
            **kwargs: Evaluation parameters
            
        Returns:
            Dictionary of metric results
        """
        return self._metric_collection.compute_all(
            reference,
            target,
            **kwargs
        )
    
    def generate_report(
        self,
        results: Dict[str, MetricResult],
        output_path: str
    ) -> None:
        """
        Generate evaluation report.
        
        Args:
            results: Metric results
            output_path: Output file path
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("===== Terrain Evaluation Results =====\n")
            f.write(f"Output Path: {output_path}\n\n")
            
            for name, result in results.items():
                f.write(f"{result.name}: {result.value:.6f}\n")
                if result.unit:
                    f.write(f"  Unit: {result.unit}\n")
                if result.description:
                    f.write(f"  Description: {result.description}\n")
                f.write("\n")
            
            f.write("=" * 40 + "\n")
    
    def add_custom_metric(
        self,
        name: str,
        metric: BaseMetric
    ) -> None:
        """
        Add custom metric to evaluator.
        
        Args:
            name: Metric name
            metric: Metric instance
        """
        self._metric_collection.add_metric(name, metric)
    
    def remove_metric(self, name: str) -> None:
        """
        Remove metric from evaluator.
        
        Args:
            name: Metric name to remove
        """
        self._metric_collection.remove_metric(name)
