"""
Metric decorators implementation.
Provides decorators for enhancing metric computation.
"""

import time
import functools
from typing import Callable, Any, Dict
import numpy as np


def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to measure execution time of functions.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function with timing
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        if isinstance(result, dict):
            result['execution_time'] = execution_time
        else:
            print(f"{func.__name__} executed in {execution_time:.4f} seconds")
        
        return result
    
    return wrapper


def cache_decorator(func: Callable) -> Callable:
    """
    Decorator to cache function results.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function with caching
    """
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cache_key = (args, tuple(sorted(kwargs.items())))
        
        if cache_key not in cache:
            cache[cache_key] = func(*args, **kwargs)
        
        return cache[cache_key]
    
    wrapper.cache = cache
    wrapper.clear_cache = lambda: cache.clear()
    
    return wrapper


def validation_decorator(
    *validators: Callable,
    **kw_validators: Callable
) -> Callable:
    """
    Decorator to validate function arguments.
    
    Args:
        *validators: Positional argument validators
        **kw_validators: Keyword argument validators
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i, validator in enumerate(validators):
                if i < len(args):
                    if not validator(args[i]):
                        raise ValueError(
                            f"Validation failed for argument {i}"
                        )
            
            for key, validator in kw_validators.items():
                if key in kwargs:
                    if not validator(kwargs[key]):
                        raise ValueError(
                            f"Validation failed for argument '{key}'"
                        )
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def logging_decorator(logger: Any = None) -> Callable:
    """
    Decorator to add logging to functions.
    
    Args:
        logger: Logger instance
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            
            if logger:
                logger.info(f"Executing {func_name}")
            else:
                print(f"[LOG] Executing {func_name}")
            
            try:
                result = func(*args, **kwargs)
                
                if logger:
                    logger.info(f"{func_name} completed successfully")
                else:
                    print(f"[LOG] {func_name} completed successfully")
                
                return result
            
            except Exception as e:
                if logger:
                    logger.error(f"{func_name} failed: {str(e)}")
                else:
                    print(f"[ERROR] {func_name} failed: {str(e)}")
                
                raise
        
        return wrapper
    
    return decorator


def retry_decorator(
    max_retries: int = 3,
    delay: float = 1.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorator to retry function on failure.
    
    Args:
        max_retries: Maximum number of retries
        delay: Delay between retries in seconds
        exceptions: Exceptions to catch
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        time.sleep(delay)
                        print(f"Retry {attempt + 1}/{max_retries} for {func.__name__}")
            
            raise last_exception
        
        return wrapper
    
    return decorator


def memoize_decorator(func: Callable) -> Callable:
    """
    Decorator for memoization with LRU cache behavior.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function with memoization
    """
    cache = {}
    access_order = []
    max_size = 128
    
    @functools.wraps(func)
    def wrapper(*args):
        cache_key = args
        
        if cache_key in cache:
            access_order.remove(cache_key)
            access_order.append(cache_key)
            return cache[cache_key]
        
        result = func(*args)
        
        if len(cache) >= max_size:
            oldest_key = access_order.pop(0)
            del cache[oldest_key]
        
        cache[cache_key] = result
        access_order.append(cache_key)
        
        return result
    
    wrapper.cache = cache
    wrapper.cache_info = lambda: {
        'size': len(cache),
        'max_size': max_size,
        'hits': len(access_order)
    }
    
    return wrapper
