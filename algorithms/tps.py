"""
Thin Plate Spline (TPS) interpolation algorithm.
Implements advanced surface interpolation with regularization.
"""

import numpy as np
import json
from typing import Optional, Tuple, Dict, Any, Callable
from dataclasses import dataclass
from core.interfaces import IInterpolationStrategy
from core.base import TerrainProcessorBase


@dataclass
class TPSConfiguration:
    """
    Configuration container for TPS interpolation parameters.
    Encapsulates all tunable parameters for interpolation.
    """
    regularization: float = 1e-6
    max_points: int = 10000
    tolerance: float = 1e-10
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.regularization < 0:
            raise ValueError("regularization must be non-negative")
        if self.max_points <= 0:
            raise ValueError("max_points must be positive")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")


class TPSKernel:
    """
    Thin Plate Spline kernel function implementation.
    Provides the radial basis function for TPS interpolation.
    """
    
    @staticmethod
    def compute(distance: np.ndarray) -> np.ndarray:
        """
        Compute TPS kernel: r^2 * log(r)
        
        Args:
            distance: Distance array
            
        Returns:
            Kernel values
        """
        result = np.zeros_like(distance)
        mask = distance > 0
        result[mask] = distance[mask] ** 2 * np.log(distance[mask])
        return result
    
    @staticmethod
    def compute_safe(distance: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
        """
        Compute TPS kernel with numerical stability.
        
        Args:
            distance: Distance array
            epsilon: Small value to avoid log(0)
            
        Returns:
            Kernel values
        """
        result = np.zeros_like(distance)
        mask = distance > 0
        result[mask] = distance[mask] ** 2 * np.log(distance[mask] + epsilon)
        return result


class TPSModel:
    """
    Thin Plate Spline model container.
    Stores interpolation coefficients and provides evaluation.
    """
    
    def __init__(
        self,
        control_points: np.ndarray,
        weights: np.ndarray,
        affine_coeffs: np.ndarray,
        regularization: float
    ):
        """
        Initialize TPS model.
        
        Args:
            control_points: Control point coordinates (N, 2)
            weights: Radial basis function weights (N,)
            affine_coeffs: Affine coefficients (3,)
            regularization: Regularization parameter
        """
        self.control_points = control_points
        self.weights = weights
        self.affine_coeffs = affine_coeffs
        self.regularization = regularization
    
    def evaluate(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Evaluate TPS surface at given coordinates.
        
        Args:
            x: X coordinates or (N, 2) array
            y: Y coordinates (optional)
            
        Returns:
            Interpolated values
        """
        if y is None:
            xy = np.asarray(x)
            if xy.ndim == 1 and xy.shape[0] == 2:
                xy = xy.reshape(1, 2)
            elif xy.ndim != 2 or xy.shape[1] != 2:
                raise ValueError("x must be (N, 2) or separate x, y arrays")
            x_vals = xy[:, 0]
            y_vals = xy[:, 1]
        else:
            x_vals = np.asarray(x).flatten()
            y_vals = np.asarray(y).flatten()
        
        query_points = np.column_stack([x_vals, y_vals])
        
        distances = self._compute_distances(query_points)
        
        kernel_values = TPSKernel.compute_safe(distances)
        
        affine = (
            self.affine_coeffs[0] +
            self.affine_coeffs[1] * x_vals +
            self.affine_coeffs[2] * y_vals
        )
        
        radial = kernel_values @ self.weights
        
        return affine + radial
    
    def _compute_distances(self, query_points: np.ndarray) -> np.ndarray:
        """
        Compute distances between query points and control points.
        
        Args:
            query_points: Query point coordinates (M, 2)
            
        Returns:
            Distance matrix (M, N)
        """
        diff = query_points[:, np.newaxis, :] - self.control_points[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert model to dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        return {
            'control_points': self.control_points.tolist(),
            'weights': self.weights.tolist(),
            'affine_coeffs': self.affine_coeffs.tolist(),
            'regularization': self.regularization
        }
    
    def to_json(self) -> str:
        """
        Convert model to JSON string.
        
        Returns:
            JSON string
        """
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TPSModel':
        """
        Create model from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            TPSModel instance
        """
        return cls(
            control_points=np.array(data['control_points']),
            weights=np.array(data['weights']),
            affine_coeffs=np.array(data['affine_coeffs']),
            regularization=data.get('regularization', 1e-6)
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'TPSModel':
        """
        Create model from JSON string.
        
        Args:
            json_str: JSON string
            
        Returns:
            TPSModel instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)


class ThinPlateSplineInterpolator(TerrainProcessorBase, IInterpolationStrategy):
    """
    Thin Plate Spline interpolator with regularization.
    
    This class implements the Strategy pattern for interpolation,
    providing smooth surface fitting through scattered data points.
    
    Attributes:
        processor_type: Type identifier for the processor
        version: Version number
    """
    
    processor_type = 'interpolation'
    version = '2.0'
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize TPS interpolator.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self._tps_config = self._create_tps_config()
        self._model: Optional[TPSModel] = None
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        pass
    
    def _create_tps_config(self) -> TPSConfiguration:
        """Create TPS configuration from config dict."""
        return TPSConfiguration(
            regularization=self._config.get('regularization', 1e-6),
            max_points=self._config.get('max_points', 10000),
            tolerance=self._config.get('tolerance', 1e-10)
        )
    
    def process(
        self,
        points: np.ndarray,
        values: np.ndarray
    ) -> TPSModel:
        """
        Fit TPS model to data points.
        
        Args:
            points: Point coordinates (N, 2)
            values: Values at points (N,)
            
        Returns:
            Fitted TPS model
        """
        self._tps_config.validate()
        
        points = np.asarray(points)
        values = np.asarray(values)
        
        if points.shape[0] != values.shape[0]:
            raise ValueError("Number of points and values must match")
        
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Points must be (N, 2) array")
        
        self._model = self._fit_model(points, values)
        
        return self._model
    
    def interpolate(
        self,
        points: np.ndarray,
        values: np.ndarray,
        query_points: np.ndarray
    ) -> np.ndarray:
        """
        Implement IInterpolationStrategy interface.
        
        Args:
            points: Known point coordinates (N, 2)
            values: Known values at points (N,)
            query_points: Query point coordinates (M, 2)
            
        Returns:
            Interpolated values at query points (M,)
        """
        model = self.process(points, values)
        return model.evaluate(query_points)
    
    def apply(self, data: Any, **kwargs) -> Any:
        """Implement IStrategy interface."""
        points = kwargs.get('points')
        values = kwargs.get('values')
        
        if points is None or values is None:
            raise ValueError("points and values are required")
        
        return self.process(points, values)
    
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return "ThinPlateSpline"
    
    def _fit_model(
        self,
        points: np.ndarray,
        values: np.ndarray
    ) -> TPSModel:
        """
        Fit TPS model to data.
        
        Args:
            points: Point coordinates
            values: Values at points
            
        Returns:
            Fitted TPS model
        """
        n = points.shape[0]
        
        distances = self._compute_pairwise_distances(points)
        
        kernel_matrix = TPSKernel.compute_safe(distances)
        
        affine_matrix = np.hstack([
            np.ones((n, 1)),
            points
        ])
        
        system_matrix = self._build_system_matrix(
            kernel_matrix,
            affine_matrix
        )
        
        rhs = np.hstack([values, np.zeros(3)])
        
        coefficients = self._solve_system(system_matrix, rhs)
        
        weights = coefficients[:n]
        affine_coeffs = coefficients[n:]
        
        return TPSModel(
            control_points=points,
            weights=weights,
            affine_coeffs=affine_coeffs,
            regularization=self._tps_config.regularization
        )
    
    def _compute_pairwise_distances(self, points: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distances between points.
        
        Args:
            points: Point coordinates (N, 2)
            
        Returns:
            Distance matrix (N, N)
        """
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))
    
    def _build_system_matrix(
        self,
        kernel_matrix: np.ndarray,
        affine_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Build the linear system matrix for TPS.
        
        Args:
            kernel_matrix: Kernel matrix (N, N)
            affine_matrix: Affine matrix (N, 3)
            
        Returns:
            System matrix (N+3, N+3)
        """
        n = kernel_matrix.shape[0]
        
        if self._tps_config.regularization > 0:
            kernel_matrix += self._tps_config.regularization * np.eye(n)
        
        zero_block = np.zeros((3, 3))
        
        top_block = np.hstack([kernel_matrix, affine_matrix])
        bottom_block = np.hstack([affine_matrix.T, zero_block])
        
        return np.vstack([top_block, bottom_block])
    
    def _solve_system(
        self,
        system_matrix: np.ndarray,
        rhs: np.ndarray
    ) -> np.ndarray:
        """
        Solve the linear system.
        
        Args:
            system_matrix: System matrix
            rhs: Right-hand side vector
            
        Returns:
            Solution vector
        """
        try:
            return np.linalg.solve(system_matrix, rhs)
        except np.linalg.LinAlgError:
            coefficients, *_ = np.linalg.lstsq(
                system_matrix,
                rhs,
                rcond=None
            )
            return coefficients
    
    def get_model(self) -> Optional[TPSModel]:
        """
        Get the fitted model.
        
        Returns:
            TPS model if fitted, None otherwise
        """
        return self._model
    
    def load_model(self, model: TPSModel) -> None:
        """
        Load a pre-fitted model.
        
        Args:
            model: TPS model to load
        """
        self._model = model
    
    def load_model_from_json(self, json_str: str) -> None:
        """
        Load model from JSON string.
        
        Args:
            json_str: JSON string representation
        """
        self._model = TPSModel.from_json(json_str)
    
    def evaluate(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Evaluate the fitted model at given coordinates.
        
        Args:
            x: X coordinates or (N, 2) array
            y: Y coordinates (optional)
            
        Returns:
            Interpolated values
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call process() first.")
        
        return self._model.evaluate(x, y)
    
    def generate_surface(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        resolution: int = 500
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a surface grid from the fitted model.
        
        Args:
            x_range: (x_min, x_max) tuple
            y_range: (y_min, y_max) tuple
            resolution: Grid resolution
            
        Returns:
            (X, Y, Z) grid arrays
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call process() first.")
        
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        Z = self.evaluate(X, Y)
        
        return X, Y, Z
