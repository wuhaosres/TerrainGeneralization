"""
Algorithm module initialization
"""

from .slic import TerrainAwareSLIC, SLICConfiguration
from .tps import ThinPlateSplineInterpolator, TPSConfiguration

__all__ = [
    'TerrainAwareSLIC',
    'SLICConfiguration',
    'ThinPlateSplineInterpolator',
    'TPSConfiguration',
]
