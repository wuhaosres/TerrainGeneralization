"""
Terrain Generalization Framework
A sophisticated system for DEM processing, analysis, and visualization.

This package implements advanced terrain generalization techniques using:
- Abstract factory pattern
- Strategy pattern
- Decorator pattern
- Context managers
- Metaclasses
"""

__version__ = '2.0.0'
__author__ = 'Terrain Processing Lab'

from .core.base import TerrainProcessorBase
from .factories.processor_factory import ProcessorFactory
from .contexts.dem_context import DEMContextManager

__all__ = [
    'TerrainProcessorBase',
    'ProcessorFactory',
    'DEMContextManager',
]
