"""
Interface definitions for terrain processing components.
Defines contracts that all implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class IProcessor(ABC):
    """
    Interface for terrain data processors.
    Defines the contract for all processor implementations.
    """
    
    @abstractmethod
    def load_data(self, file_path: str) -> Any:
        """
        Load terrain data from file.
        
        Args:
            file_path: Path to input file
            
        Returns:
            Loaded data object
        """
        pass
    
    @abstractmethod
    def process(self, data: Any, **kwargs) -> Any:
        """
        Process terrain data.
        
        Args:
            data: Input terrain data
            **kwargs: Processing parameters
            
        Returns:
            Processed data
        """
        pass
    
    @abstractmethod
    def save_data(self, data: Any, file_path: str) -> None:
        """
        Save processed data to file.
        
        Args:
            data: Data to save
            file_path: Output file path
        """
        pass


class IEvaluator(ABC):
    """
    Interface for terrain evaluation metrics.
    Defines the contract for all evaluator implementations.
    """
    
    @abstractmethod
    def compute_metric(self, reference: Any, target: Any) -> float:
        """
        Compute evaluation metric between reference and target.
        
        Args:
            reference: Reference data (ground truth)
            target: Target data to evaluate
            
        Returns:
            Metric value
        """
        pass
    
    @abstractmethod
    def get_metric_name(self) -> str:
        """
        Get the name of the metric.
        
        Returns:
            Metric name string
        """
        pass
    
    @abstractmethod
    def get_metric_range(self) -> Tuple[float, float]:
        """
        Get the valid range for the metric.
        
        Returns:
            Tuple of (min_value, max_value)
        """
        pass


class IVisualizer(ABC):
    """
    Interface for terrain visualization components.
    Defines the contract for all visualizer implementations.
    """
    
    @abstractmethod
    def visualize(self, data: Any, **kwargs) -> Any:
        """
        Visualize terrain data.
        
        Args:
            data: Terrain data to visualize
            **kwargs: Visualization parameters
            
        Returns:
            Visualization object
        """
        pass
    
    @abstractmethod
    def save_visualization(self, output_path: str) -> None:
        """
        Save visualization to file.
        
        Args:
            output_path: Output file path
        """
        pass


class IAlgorithm(ABC):
    """
    Interface for terrain processing algorithms.
    Defines the contract for algorithm implementations.
    """
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """
        Execute the algorithm.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Algorithm result
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get algorithm parameters.
        
        Returns:
            Dictionary of parameters
        """
        pass
    
    @abstractmethod
    def set_parameters(self, **kwargs) -> None:
        """
        Set algorithm parameters.
        
        Args:
            **kwargs: Parameter key-value pairs
        """
        pass


class IStrategy(ABC):
    """
    Interface for strategy pattern implementations.
    Defines interchangeable algorithms for terrain processing.
    """
    
    @abstractmethod
    def apply(self, data: Any, **kwargs) -> Any:
        """
        Apply the strategy to data.
        
        Args:
            data: Input data
            **kwargs: Strategy parameters
            
        Returns:
            Processed data
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """
        Get the name of the strategy.
        
        Returns:
            Strategy name string
        """
        pass


class IInterpolationStrategy(IStrategy):
    """
    Interface for interpolation strategies.
    Specializes IStrategy for interpolation algorithms.
    """
    
    @abstractmethod
    def interpolate(
        self,
        points: np.ndarray,
        values: np.ndarray,
        query_points: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate values at query points.
        
        Args:
            points: Known point coordinates (N, 2)
            values: Known values at points (N,)
            query_points: Query point coordinates (M, 2)
            
        Returns:
            Interpolated values at query points (M,)
        """
        pass


class ISegmentationStrategy(IStrategy):
    """
    Interface for segmentation strategies.
    Specializes IStrategy for segmentation algorithms.
    """
    
    @abstractmethod
    def segment(
        self,
        image: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Segment the input image.
        
        Args:
            image: Input image array
            **kwargs: Segmentation parameters
            
        Returns:
            Segmentation labels array
        """
        pass


class IDataContainer(ABC):
    """
    Interface for data container objects.
    Provides a unified interface for terrain data access.
    """
    
    @abstractmethod
    def get_data(self) -> np.ndarray:
        """
        Get the underlying data array.
        
        Returns:
            Numpy array of data
        """
        pass
    
    @abstractmethod
    def set_data(self, data: np.ndarray) -> None:
        """
        Set the data array.
        
        Args:
            data: Numpy array to set
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata associated with the data.
        
        Returns:
            Metadata dictionary
        """
        pass
    
    @abstractmethod
    def get_extent(self) -> Tuple[float, float, float, float]:
        """
        Get the spatial extent of the data.
        
        Returns:
            Tuple of (xmin, ymin, xmax, ymax)
        """
        pass
