"""
Metaclass implementations for advanced class behavior.
Provides automatic validation and registration mechanisms.
"""

from typing import Any, Callable, Dict, List, Optional, Type
from .base import ValidatedMeta
import inspect


class TerrainMeta(ValidatedMeta):
    """
    Enhanced metaclass for terrain processing components.
    Provides automatic registration and dependency injection.
    """
    
    _registry: Dict[str, Type] = {}
    _dependencies: Dict[str, List[str]] = {}
    
    def __new__(
        mcs,
        name: str,
        bases: tuple,
        namespace: dict,
        **kwargs
    ):
        """
        Create a new class with automatic registration.
        
        Args:
            name: Class name
            bases: Base classes
            namespace: Class namespace
            **kwargs: Additional keyword arguments
            
        Returns:
            Newly created class
        """
        cls = super().__new__(mcs, name, bases, namespace)
        
        if not name.startswith('Abstract') and not name.startswith('_'):
            mcs._registry[name] = cls
            
            if hasattr(cls, '_dependencies'):
                mcs._dependencies[name] = cls._dependencies
        
        return cls
    
    @classmethod
    def get_class(mcs, name: str) -> Optional[Type]:
        """
        Retrieve a class by name from the registry.
        
        Args:
            name: Class name to retrieve
            
        Returns:
            Class if found, None otherwise
        """
        return mcs._registry.get(name)
    
    @classmethod
    def get_all_classes(mcs) -> Dict[str, Type]:
        """
        Get all registered classes.
        
        Returns:
            Dictionary of registered classes
        """
        return mcs._registry.copy()
    
    @classmethod
    def get_dependencies(mcs, class_name: str) -> List[str]:
        """
        Get dependencies for a class.
        
        Args:
            class_name: Name of the class
            
        Returns:
            List of dependency names
        """
        return mcs._dependencies.get(class_name, [])


class PropertyMeta(type):
    """
    Metaclass that enforces property definitions.
    Automatically creates getters and setters for annotated attributes.
    """
    
    def __new__(mcs, name: str, bases: tuple, namespace: dict):
        """
        Create class with automatic property generation.
        
        Args:
            name: Class name
            bases: Base classes
            namespace: Class namespace
            
        Returns:
            Created class with properties
        """
        annotations = namespace.get('__annotations__', {})
        
        for attr_name, attr_type in annotations.items():
            if not attr_name.startswith('_'):
                private_name = f'_{attr_name}'
                
                def make_property(name, private):
                    def getter(self):
                        return getattr(self, private, None)
                    
                    def setter(self, value):
                        if not isinstance(value, attr_type):
                            raise TypeError(
                                f"{name} must be of type {attr_type}"
                            )
                        setattr(self, private, value)
                    
                    return property(getter, setter)
                
                namespace[attr_name] = make_property(attr_name, private_name)
        
        return super().__new__(mcs, name, bases, namespace)


class CachedMethodMeta(type):
    """
    Metaclass that provides automatic method result caching.
    Caches method results based on arguments.
    """
    
    def __new__(mcs, name: str, bases: tuple, namespace: dict):
        """
        Create class with method caching capabilities.
        
        Args:
            name: Class name
            bases: Base classes
            namespace: Class namespace
            
        Returns:
            Class with cached methods
        """
        cls = super().__new__(mcs, name, bases, namespace)
        
        for attr_name, attr_value in namespace.items():
            if callable(attr_value) and not attr_name.startswith('_'):
                if hasattr(attr_value, '_cache_enabled'):
                    setattr(cls, attr_name, mcs._create_cached_method(attr_value))
        
        return cls
    
    @staticmethod
    def _create_cached_method(method: Callable) -> Callable:
        """
        Create a cached version of a method.
        
        Args:
            method: Original method
            
        Returns:
            Cached method wrapper
        """
        cache = {}
        
        def cached_method(self, *args, **kwargs):
            cache_key = (args, tuple(sorted(kwargs.items())))
            
            if cache_key not in cache:
                cache[cache_key] = method(self, *args, **kwargs)
            
            return cache[cache_key]
        
        cached_method._cache = cache
        return cached_method


def cache_result(func: Callable) -> Callable:
    """
    Decorator to enable method result caching.
    
    Args:
        func: Function to cache
        
    Returns:
        Decorated function with caching
    """
    func._cache_enabled = True
    return func


class TrackedMeta(type):
    """
    Metaclass that tracks instance creation and deletion.
    Useful for debugging and memory management.
    """
    
    _instances: Dict[int, object] = {}
    _instance_count: int = 0
    
    def __call__(cls, *args, **kwargs):
        """
        Create and track a new instance.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            New instance
        """
        instance = super().__call__(*args, **kwargs)
        instance_id = id(instance)
        cls._instances[instance_id] = instance
        cls._instance_count += 1
        return instance
    
    @classmethod
    def get_instance_count(mcs) -> int:
        """
        Get total number of instances created.
        
        Returns:
            Instance count
        """
        return mcs._instance_count
    
    @classmethod
    def get_active_instances(mcs) -> List[object]:
        """
        Get list of active instances.
        
        Returns:
            List of active instances
        """
        return list(mcs._instances.values())
