"""
Interpolation strategy implementations.
Implements the Strategy pattern for interchangeable interpolation algorithms.
"""

from typing import Dict, Any, Optional, Type
from enum import Enum
import numpy as np

from core.interfaces import IInterpolationStrategy
from algorithms.tps import ThinPlateSplineInterpolator


class InterpolationType(Enum):
    """Enumeration of interpolation types."""
    TPS = "thin_plate_spline"
    IDW = "inverse_distance_weighting"
    KRIGING = "kriging"
    SPLINE = "spline"
    NEAREST = "nearest_neighbor"


class InterpolationStrategyFactory:
    """
    Factory for creating interpolation strategies.
    
    Implements the Factory Method pattern to provide
    flexible strategy instantiation.
    """
    
    _strategies: Dict[str, Type[IInterpolationStrategy]] = {}
    
    @classmethod
    def register_strategy(
        cls,
        strategy_type: str,
        strategy_class: Type[IInterpolationStrategy]
    ) -> None:
        """
        Register an interpolation strategy.
        
        Args:
            strategy_type: Strategy type identifier
            strategy_class: Strategy class
        """
        cls._strategies[strategy_type] = strategy_class
    
    @classmethod
    def create_strategy(
        cls,
        strategy_type: str,
        config: Optional[Dict[str, Any]] = None
    ) -> IInterpolationStrategy:
        """
        Create an interpolation strategy.
        
        Args:
            strategy_type: Type of strategy to create
            config: Strategy configuration
            
        Returns:
            Strategy instance
        """
        if strategy_type not in cls._strategies:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        strategy_class = cls._strategies[strategy_type]
        return strategy_class(config=config)
    
    @classmethod
    def get_available_strategies(cls) -> list:
        """
        Get list of available strategy types.
        
        Returns:
            List of strategy type names
        """
        return list(cls._strategies.keys())


InterpolationStrategyFactory.register_strategy(
    InterpolationType.TPS.value,
    ThinPlateSplineInterpolator
)


class InterpolationContext:
    """
    Context for interpolation strategy execution.
    
    Implements the Context role in the Strategy pattern,
    managing strategy selection and execution.
    """
    
    def __init__(
        self,
        strategy: Optional[IInterpolationStrategy] = None
    ):
        """
        Initialize interpolation context.
        
        Args:
            strategy: Interpolation strategy to use
        """
        self._strategy = strategy
    
    def set_strategy(self, strategy: IInterpolationStrategy) -> None:
        """
        Set the interpolation strategy.
        
        Args:
            strategy: Strategy to use
        """
        self._strategy = strategy
    
    def execute_interpolation(
        self,
        points: np.ndarray,
        values: np.ndarray,
        query_points: np.ndarray
    ) -> np.ndarray:
        """
        Execute interpolation using current strategy.
        
        Args:
            points: Known point coordinates
            values: Known values
            query_points: Query point coordinates
            
        Returns:
            Interpolated values
        """
        if self._strategy is None:
            raise RuntimeError("No strategy set")
        
        return self._strategy.interpolate(points, values, query_points)
    
    def get_strategy_name(self) -> str:
        """
        Get name of current strategy.
        
        Returns:
            Strategy name
        """
        if self._strategy is None:
            return "None"
        return self._strategy.get_strategy_name()
