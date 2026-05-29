"""
Example configuration for terrain generalization pipeline.
"""

PIPELINE_CONFIG = {
    'dem': {
        'z_factor': 1.0,
        'azimuth': 315.0,
        'altitude': 45.0,
        'contour_interval': 100.0,
        'output_resolution': (800, 800),
        'output_dpi': 96
    },
    
    'hydro': {
        'flow_accumulation_threshold': 1000,
        'fill_depressions': True,
        'snap_pour_point': True
    },
    
    'slic': {
        'num_segments': 200,
        'compactness': 8.0,
        'terrain_weight': 4.0,
        'max_iterations': 15,
        'min_region_size': 30
    },
    
    'tps': {
        'regularization': 1e-6,
        'max_points': 10000,
        'tolerance': 1e-10
    },
    
    'evaluator': {},
    
    'visualizer': {
        'output_size': (800, 800),
        'output_dpi': 96,
        'colormap': 'terrain',
        'show_axes': False,
        'show_colorbar': True
    },
    
    'extract_streams': True,
    'extract_contours': True,
    'perform_segmentation': True,
    'visualize': True
}


ADVANCED_CONFIG = {
    'dem': {
        'z_factor': 1.0,
        'azimuth': 315.0,
        'altitude': 45.0
    },
    
    'slic': {
        'num_segments': 500,
        'compactness': 10.0,
        'terrain_weight': 7.0,
        'max_iterations': 20
    },
    
    'tps': {
        'regularization': 1e-5
    },
    
    'extract_streams': True,
    'extract_contours': True,
    'perform_segmentation': True,
    'visualize': True
}


MINIMAL_CONFIG = {
    'extract_streams': False,
    'extract_contours': False,
    'perform_segmentation': False,
    'visualize': True
}
