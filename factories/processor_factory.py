"""
Processor factory implementation.
Implements the Abstract Factory pattern for processor creation.
"""

from typing import Dict, Any, Optional, Type
from dataclasses import dataclass
from enum import Enum

from core.base import TerrainProcessorBase
from core.interfaces import IProcessor, IEvaluator, IVisualizer, IAlgorithm


class ProcessorType(Enum):
    """Enumeration of processor types."""
    DEM = "dem"
    HYDRO = "hydro"
    INTERPOLATION = "interpolation"
    SEGMENTATION = "segmentation"
    EVALUATOR = "evaluator"
    VISUALIZER = "visualizer"


@dataclass
class FactoryConfiguration:
    """
    Configuration for processor factory.
    """
    default_config: Dict[str, Any] = None
    cache_instances: bool = True
    
    def __post_init__(self):
        if self.default_config is None:
            self.default_config = {}


class ProcessorFactory:
    """
    Abstract factory for creating terrain processors.
    
    Implements the Factory Method and Abstract Factory patterns
    to provide flexible processor instantiation.
    """
    
    _registry: Dict[str, Type[TerrainProcessorBase]] = {}
    _instances: Dict[str, TerrainProcessorBase] = {}
    
    def __init__(self, config: Optional[FactoryConfiguration] = None):
        """
        Initialize processor factory.
        
        Args:
            config: Factory configuration
        """
        self._config = config or FactoryConfiguration()
    
    @classmethod
    def register_processor(
        cls,
        processor_type: str,
        processor_class: Type[TerrainProcessorBase]
    ) -> None:
        """
        Register a processor class.
        
        Args:
            processor_type: Type identifier
            processor_class: Processor class
        """
        cls._registry[processor_type] = processor_class
    
    @classmethod
    def create_processor(
        cls,
        processor_type: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> TerrainProcessorBase:
        """
        Create a processor instance.
        
        Args:
            processor_type: Type of processor to create
            config: Processor configuration
            **kwargs: Additional arguments
            
        Returns:
            Processor instance
        """
        if processor_type not in cls._registry:
            raise ValueError(f"Unknown processor type: {processor_type}")
        
        processor_class = cls._registry[processor_type]
        
        merged_config = {}
        if hasattr(cls, '_config') and cls._config.default_config:
            merged_config.update(cls._config.default_config)
        if config:
            merged_config.update(config)
        merged_config.update(kwargs)
        
        instance = processor_class(config=merged_config)
        
        return instance
    
    @classmethod
    def get_or_create(
        cls,
        processor_type: str,
        config: Optional[Dict[str, Any]] = None
    ) -> TerrainProcessorBase:
        """
        Get existing instance or create new one.
        
        Args:
            processor_type: Type of processor
            config: Processor configuration
            
        Returns:
            Processor instance
        """
        cache_key = f"{processor_type}_{hash(frozenset(config.items())) if config else 'default'}"
        
        if cache_key in cls._instances:
            return cls._instances[cache_key]
        
        instance = cls.create_processor(processor_type, config)
        cls._instances[cache_key] = instance
        
        return instance
    
    @classmethod
    def get_available_processors(cls) -> list:
        """
        Get list of available processor types.
        
        Returns:
            List of processor type names
        """
        return list(cls._registry.keys())
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear cached instances."""
        cls._instances.clear()


class ProcessorBuilder:
    """
    Builder pattern implementation for complex processor creation.
    
    Provides a fluent interface for configuring and creating
    processors with multiple parameters.
    """
    
    def __init__(self):
        """Initialize builder."""
        self._processor_type: Optional[str] = None
        self._config: Dict[str, Any] = {}
        self._custom_params: Dict[str, Any] = {}
    
    def set_type(self, processor_type: str) -> 'ProcessorBuilder':
        """
        Set processor type.
        
        Args:
            processor_type: Type of processor
            
        Returns:
            Builder instance
        """
        self._processor_type = processor_type
        return self
    
    def set_config(self, **kwargs) -> 'ProcessorBuilder':
        """
        Set configuration parameters.
        
        Args:
            **kwargs: Configuration parameters
            
        Returns:
            Builder instance
        """
        self._config.update(kwargs)
        return self
    
    def add_param(self, key: str, value: Any) -> 'ProcessorBuilder':
        """
        Add custom parameter.
        
        Args:
            key: Parameter key
            value: Parameter value
            
        Returns:
            Builder instance
        """
        self._custom_params[key] = value
        return self
    
    def build(self) -> TerrainProcessorBase:
        """
        Build processor instance.
        
        Returns:
            Configured processor instance
        """
        if self._processor_type is None:
            raise ValueError("Processor type must be set")
        
        merged_config = {**self._config, **self._custom_params}
        
        return ProcessorFactory.create_processor(
            self._processor_type,
            config=merged_config
        )


def register_processor(processor_type: str):
    """
    Decorator for automatic processor registration.
    
    Args:
        processor_type: Type identifier for processor
        
    Returns:
        Decorator function
    """
    def decorator(cls):
        ProcessorFactory.register_processor(processor_type, cls)
        return cls
    return decorator
