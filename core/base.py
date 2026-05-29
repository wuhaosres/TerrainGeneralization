"""
Base classes and metaclasses for terrain processing framework.
Implements core architectural patterns and abstractions.
"""

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional, Tuple, Type
import threading
import weakref


class SingletonMeta(type):
    """
    Thread-safe singleton metaclass implementation.
    Ensures only one instance of a class exists throughout the application.
    """
    
    _instances = {}
    _lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]


class ValidatedMeta(ABCMeta):
    """
    Metaclass that enforces validation of class attributes.
    Automatically validates required attributes upon class creation.
    """
    
    def __new__(mcs, name: str, bases: Tuple[Type, ...], namespace: Dict[str, Any]):
        cls = super().__new__(mcs, name, bases, namespace)
        
        if hasattr(cls, '_required_attributes'):
            for attr in cls._required_attributes:
                if not hasattr(cls, attr):
                    raise AttributeError(
                        f"Class {name} must define required attribute '{attr}'"
                    )
        
        return cls


class TerrainMeta(ValidatedMeta):
    """
    Custom metaclass for terrain processing components.
    Combines validation and automatic registration capabilities.
    """
    
    _registry = weakref.WeakValueDictionary()
    
    def __new__(mcs, name: str, bases: Tuple[Type, ...], namespace: Dict[str, Any]):
        cls = super().__new__(mcs, name, bases, namespace)
        
        if not name.startswith('Abstract'):
            mcs._registry[name] = cls
        
        return cls
    
    @classmethod
    def get_registry(mcs) -> Dict[str, Type]:
        """Retrieve all registered terrain processing classes."""
        return dict(mcs._registry)


class TerrainProcessorBase(metaclass=TerrainMeta):
    """
    Abstract base class for all terrain processors.
    Defines the core interface and common functionality.
    
    This class implements the Template Method pattern, providing
    a skeleton algorithm that subclasses can override.
    """
    
    _required_attributes = ['processor_type', 'version']
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the terrain processor.
        
        Args:
            config: Configuration dictionary for processor parameters
        """
        self._config = config or {}
        self._initialized = False
        self._validate_config()
    
    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """
        Main processing method to be implemented by subclasses.
        
        Returns:
            Processing result
        """
        pass
    
    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate configuration parameters.
        Must be implemented by subclasses.
        """
        pass
    
    def initialize(self) -> None:
        """
        Initialize the processor.
        Template method that can be overridden.
        """
        if not self._initialized:
            self._setup()
            self._initialized = True
    
    def _setup(self) -> None:
        """
        Setup method called during initialization.
        Can be overridden by subclasses.
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Retrieve current configuration."""
        return self._config.copy()
    
    def update_config(self, **kwargs) -> None:
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Configuration key-value pairs
        """
        self._config.update(kwargs)
        self._validate_config()
    
    @property
    def is_initialized(self) -> bool:
        """Check if processor is initialized."""
        return self._initialized
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._cleanup()
    
    def _cleanup(self) -> None:
        """
        Cleanup method called when exiting context.
        Can be overridden by subclasses.
        """
        pass


class ObservableMixin:
    """
    Mixin class that adds observer pattern functionality.
    Allows objects to notify observers of state changes.
    """
    
    def __init__(self):
        self._observers = []
    
    def add_observer(self, observer: Any) -> None:
        """
        Add an observer to the notification list.
        
        Args:
            observer: Observer object with update() method
        """
        if observer not in self._observers:
            self._observers.append(observer)
    
    def remove_observer(self, observer: Any) -> None:
        """
        Remove an observer from the notification list.
        
        Args:
            observer: Observer to remove
        """
        if observer in self._observers:
            self._observers.remove(observer)
    
    def notify_observers(self, *args, **kwargs) -> None:
        """
        Notify all observers of a state change.
        
        Args:
            *args: Positional arguments to pass to observers
            **kwargs: Keyword arguments to pass to observers
        """
        for observer in self._observers:
            if hasattr(observer, 'update'):
                observer.update(self, *args, **kwargs)
