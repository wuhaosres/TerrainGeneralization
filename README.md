# Terrain Generalization Framework

A sophisticated Python framework for Digital Elevation Model (DEM) processing, terrain generalization, and analysis.

## Architecture Overview

This framework implements advanced software engineering patterns and practices:

### Design Patterns Used

1. **Abstract Factory Pattern** - `factories/processor_factory.py`
   - Flexible processor instantiation
   - Configuration-driven object creation

2. **Strategy Pattern** - `strategies/interpolation_strategy.py`
   - Interchangeable algorithms
   - Runtime algorithm selection

3. **Decorator Pattern** - `decorators/metric_decorators.py`
   - Function enhancement
   - Cross-cutting concerns

4. **Template Method Pattern** - `core/base.py`
   - Algorithm skeleton definition
   - Customizable processing steps

5. **Observer Pattern** - `core/base.py` (ObservableMixin)
   - Event notification system
   - Loose coupling between components

6. **Builder Pattern** - `factories/processor_factory.py`
   - Complex object construction
   - Fluent interface

7. **Context Manager Pattern** - `contexts/dem_context.py`
   - Resource management
   - Automatic cleanup

### Metaclasses

- **SingletonMeta** - Ensures single instance per class
- **ValidatedMeta** - Automatic attribute validation
- **TerrainMeta** - Component registration and dependency tracking
- **CachedMethodMeta** - Automatic method result caching
- **TrackedMeta** - Instance tracking for debugging

## Project Structure

```
op_code/
├── core/                   # Core abstractions and base classes
│   ├── base.py            # Base classes and metaclasses
│   ├── interfaces.py      # Interface definitions
│   └── meta.py            # Advanced metaclass implementations
│
├── algorithms/            # Algorithm implementations
│   ├── slic.py           # Terrain-aware SLIC segmentation
│   └── tps.py            # Thin Plate Spline interpolation
│
├── processors/            # Data processing components
│   ├── dem_processor.py  # DEM processing operations
│   └── hydro_processor.py # Hydrological analysis
│
├── evaluators/            # Evaluation metrics
│   └── metrics.py        # Terrain evaluation metrics
│
├── visualizers/           # Visualization components
│   └── dem_visualizer.py # DEM visualization tools
│
├── utils/                 # Utility functions
│   └── helpers.py        # Helper utilities
│
├── factories/             # Object factories
│   └── processor_factory.py # Processor creation
│
├── strategies/            # Strategy pattern implementations
│   └── interpolation_strategy.py # Interpolation strategies
│
├── decorators/            # Function decorators
│   └── metric_decorators.py # Metric enhancement decorators
│
├── contexts/              # Context managers
│   └── dem_context.py    # DEM resource management
│
└── main.py               # Main entry point
```

## Key Features

### 1. Terrain-Aware SLIC Segmentation
- Incorporates terrain features (curvature) into superpixel segmentation
- Structure line constraints for boundary adherence
- Configurable parameters: segments, compactness, terrain weight

### 2. Thin Plate Spline Interpolation
- Regularized TPS for smooth surface fitting
- JSON serialization for model persistence
- Efficient evaluation at arbitrary points

### 3. Comprehensive Evaluation Metrics
- **RMSE** - Root Mean Square Error
- **SMR** - Stream Match Ratio
- **SSIM** - Structural Similarity Index
- **Roughness** - Terrain roughness
- **TRI** - Terrain Ruggedness Index

### 4. Professional Visualization
- 3D surface rendering
- 2D color mapping
- QGIS-based cartographic output

### 5. Hydrological Analysis
- Stream network extraction
- Contour generation
- Flow direction and accumulation

## Usage Examples

### Basic Usage

```python
from op_code.main import TerrainAnalysisAPI

# Load DEM
dem_data = TerrainAnalysisAPI.load_dem('path/to/dem.tif')

# Generate hillshade
hillshade = TerrainAnalysisAPI.generate_hillshade(dem_data)

# Compute curvature
curvature = TerrainAnalysisAPI.compute_curvature(dem_data)

# Visualize
TerrainAnalysisAPI.visualize_3d(dem_data, 'output.png')
```

### Advanced Pipeline

```python
from op_code.main import TerrainGeneralizationPipeline

# Configure pipeline
config = {
    'extract_streams': True,
    'extract_contours': True,
    'perform_segmentation': True,
    'visualize': True,
    'slic': {
        'num_segments': 500,
        'compactness': 10.0
    }
}

# Execute pipeline
pipeline = TerrainGeneralizationPipeline(config)
results = pipeline.execute('input.tif', 'output_dir/')
```

### Using Context Managers

```python
from op_code.contexts.dem_context import DEMContextManager

with DEMContextManager('dem.tif', auto_save=True) as ctx:
    # Access data
    data = ctx.data
    
    # Modify data
    ctx.data = processed_data
    
    # Auto-saved on exit
```

### Custom Metrics

```python
from op_code.evaluators.metrics import TerrainEvaluator, BaseMetric

class CustomMetric(BaseMetric):
    def compute(self, reference, target, **kwargs):
        # Custom computation logic
        value = my_custom_calculation(reference, target)
        return MetricResult(
            name="CustomMetric",
            value=value,
            description="My custom metric"
        )

evaluator = TerrainEvaluator()
evaluator.add_custom_metric('custom', CustomMetric())
```

## Dependencies

```
numpy>=1.20.0
opencv-python>=4.5.0
rasterio>=1.2.0
matplotlib>=3.3.0
scikit-image>=0.18.0
scipy>=1.6.0
geopandas>=0.9.0
shapely>=1.7.0
```

Optional dependencies:
```
arcpy (ArcGIS Pro)
qgis (QGIS Python API)
```

## Configuration

Configuration can be provided at multiple levels:

1. **Global Configuration** - Passed to pipeline constructor
2. **Processor Configuration** - Specific to each processor
3. **Runtime Parameters** - Passed to individual methods

Example:
```python
config = {
    'dem': {
        'z_factor': 1.0,
        'azimuth': 315.0
    },
    'slic': {
        'num_segments': 200,
        'compactness': 8.0,
        'terrain_weight': 4.0
    },
    'tps': {
        'regularization': 1e-6
    }
}
```

## Extension Points

### Adding New Processors

```python
from op_code.core.base import TerrainProcessorBase
from op_code.factories.processor_factory import register_processor

@register_processor('custom_processor')
class CustomProcessor(TerrainProcessorBase):
    processor_type = 'custom'
    version = '1.0'
    
    def process(self, *args, **kwargs):
        # Implementation
        pass
```

### Adding New Strategies

```python
from op_code.core.interfaces import IInterpolationStrategy
from op_code.strategies.interpolation_strategy import InterpolationStrategyFactory

class CustomInterpolation(IInterpolationStrategy):
    def interpolate(self, points, values, query_points):
        # Implementation
        pass

InterpolationStrategyFactory.register_strategy('custom', CustomInterpolation)
```

## Performance Considerations

- Use context managers for automatic resource cleanup
- Leverage caching decorators for expensive computations
- Configure appropriate segment counts for SLIC
- Adjust regularization for TPS based on data density

## License

MIT License

## Author

Terrain Processing Lab

## Version

2.0.0
