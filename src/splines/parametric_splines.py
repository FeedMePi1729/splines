
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Union, Tuple, Optional

class ParametricSplines:
    """
    Fits multi-dimensional data using parametric smoothing splines (Reinsch algorithm).
    
    Allows one variable to be the "independent" variable (e.g., CDS level) which is 
    bijectively mapped to a time parameter t. All variables (independent and dependent)
    are then modeled as smoothing splines x(t), y(t), z(t), etc.
    """
    
    def __init__(self, independent: np.ndarray, dependents: Dict[str, np.ndarray], 
                 smoothing: float = 0.0):
        """
        Initialize and fit the parametric splines.

        Args:
            independent: 1D array of the independent variable (e.g., cds_level).
            dependents: Dictionary mapping variable names to 1D arrays of dependent variables.
            smoothing: Smoothing parameter lambda >= 0.
                       0.0 = Interpolating spline (passes through all points).
                       >0.0 = Smoothing spline.
        """
        self.independent_name = "independent"
        self.smoothing = smoothing
        
        # 1. Prepare Data
        # Sort by independent variable to ensure monotonicity for the mapping
        indep_raw = np.array(independent, dtype=float)
        
        # Check standard input validity
        if len(indep_raw) < 2:
            raise ValueError("At least 2 data points are required.")
            
        sort_idx = np.argsort(indep_raw)
        x_sorted_all = indep_raw[sort_idx]
        
        # Identify unique independent values
        unique_x, unique_indices = np.unique(x_sorted_all, return_inverse=True)
        # unique_indices maps original sorted indices to unique x indices
        
        self.x_sorted = unique_x
        
        # Prepare dependents
        self.dependents_data = {}
        for name, data in dependents.items():
            if len(data) != len(indep_raw):
                raise ValueError(f"Dependent variable '{name}' length mismatch.")
            
            y_sorted_all = np.array(data, dtype=float)[sort_idx]
            
            # Average duplicates
            y_sums = np.bincount(unique_indices, weights=y_sorted_all)
            y_counts = np.bincount(unique_indices)
            
            self.dependents_data[name] = y_sums / y_counts

        # 2. Define Parametric Time t
        x_min, x_max = self.x_sorted[0], self.x_sorted[-1]
        self.t = (self.x_sorted - x_min) / (x_max - x_min)
        
        # 3. Fit Splines for each dimension
        self.splines: Dict[str, dict] = {}
        
        # Fit independent variable (identity mapping essentially, but good for consistency)
        self.splines[self.independent_name] = self._fit_reinsch(self.t, self.x_sorted, smoothing)
        
        # Fit dependent variables
        for name, y_data in self.dependents_data.items():
            self.splines[name] = self._fit_reinsch(self.t, y_data, smoothing)

    def _fit_reinsch(self, t: np.ndarray, y: np.ndarray, s: float) -> dict:
        """
        Implements Reinsch (1967) algorithm for cubic smoothing splines.
        Minimizes: sum((y_i - f(t_i))^2) + s * integral(f''(t)^2 dt)
        
        Returns dictionary with spline coefficients on interval [t_i, t_{i+1}]:
        S(x) = a_i + b_i(x-t_i) + c_i(x-t_i)^2 + d_i(x-t_i)^3
        """
        n = len(t)
        h = np.diff(t)
        
        # 1. Compute Q matrix (tridiagonal)
        Q = np.zeros((n, n-2))
        for i in range(n-2):
            Q[i, i] = 1.0 / h[i]
            Q[i+1, i] = -1.0 / h[i] - 1.0 / h[i+1]
            Q[i+2, i] = 1.0 / h[i+1]
            
        # 2. Compute T matrix (positive definite tridiagonal)
        T = np.zeros((n-2, n-2))
        for i in range(n-2):
            T[i, i] = (h[i] + h[i+1]) / 3.0
            if i < n-3:
                T[i, i+1] = h[i+1] / 6.0
                T[i+1, i] = h[i+1] / 6.0
                
        # 3. Solve for spline coefficients
        QT_y = Q.T @ y
        
        if s > 1e-12:
            # Smoothing Spline
            M_sys = T + s * (Q.T @ Q)
            M_vec = np.linalg.solve(M_sys, QT_y)
            
            # Recover a (fitted values at knots)
            a = y - s * (Q @ M_vec)
            c_inner = M_vec / 2.0
            
        else:
            # Exact Interpolation
            a = y 
            M = np.linalg.solve(T, QT_y)
            c_inner = M / 2.0
            
        c = np.concatenate(([0], c_inner, [0]))
        d = (c[1:] - c[:-1]) / (3.0 * h)
        b = (a[1:] - a[:-1]) / h - (h / 3.0) * (2.0 * c[:-1] + c[1:])
        
        return {
            'a': a[:-1],
            'b': b,
            'c': c[:-1],
            'd': d,
            'x': t[:-1]
        }
        
    def _evaluate_spline_vectorized(self, t_vals: np.ndarray, coeffs: dict) -> np.ndarray:
        """Evaluate spline at multiple t values."""
        x_knots = coeffs['x']
        indices = np.searchsorted(x_knots, t_vals, side='right') - 1
        indices = np.clip(indices, 0, len(x_knots) - 1)
        
        dt = t_vals - x_knots[indices]
        return (coeffs['a'][indices] + 
                coeffs['b'][indices] * dt + 
                coeffs['c'][indices] * dt**2 + 
                coeffs['d'][indices] * dt**3)

    def query(self, independent_value: Union[float, np.ndarray]) -> Dict[str, Union[float, np.ndarray]]:
        """
        Query the dependent variables for given independent variable value(s).
        """
        scalar_input = np.isscalar(independent_value)
        x_target = np.atleast_1d(independent_value)
                
        x_min, x_max = self.x_sorted[0], self.x_sorted[-1]
        
        # Simple linear map since we defined t this way and x is the reference
        t_query = (x_target - x_min) / (x_max - x_min)
        
        results = {}
        # Return the queried independent value itself (for verification)
        results[self.independent_name] = x_target
        
        for name, coeffs in self.splines.items():
            if name == self.independent_name:
                continue
            results[name] = self._evaluate_spline_vectorized(t_query, coeffs)
            
        if scalar_input:
            return {k: float(v[0]) for k, v in results.items()}
        return results

    def plot_data(self) -> go.Figure:
        """Visualize the raw data points."""
        fig = go.Figure()
        
        # We plot each dependent vs independent
        for name, y_data in self.dependents_data.items():
            fig.add_trace(go.Scatter(
                x=self.x_sorted,
                y=y_data,
                mode='markers',
                name=f'{name} (Data)'
            ))
            
        fig.update_layout(
            title="Raw Data",
            xaxis_title=self.independent_name,
            yaxis_title="Dependent Values"
        )
        return fig

    def plot_fitted(self, num_points: int = 100) -> go.Figure:
        """Visualize the fitted curves."""
        fig = go.Figure()
        
        x_grid = np.linspace(self.x_sorted[0], self.x_sorted[-1], num_points)
        t_grid = (x_grid - self.x_sorted[0]) / (self.x_sorted[-1] - self.x_sorted[0])
        
        for name, coeffs in self.splines.items():
            if name == self.independent_name:
                continue
            y_fit = self._evaluate_spline_vectorized(t_grid, coeffs)
            fig.add_trace(go.Scatter(
                x=x_grid,
                y=y_fit,
                mode='lines',
                name=f'{name} (Fit, s={self.smoothing})'
            ))
            
        fig.update_layout(
            title=f"Fitted Parametric Splines (s={self.smoothing})",
            xaxis_title=self.independent_name,
            yaxis_title="Dependent Values"
        )
        return fig

    def plot_combined(self, num_points: int = 100) -> go.Figure:
        """Visualize both raw data and fitted curves."""
        fig = self.plot_fitted(num_points)
        
        # Add data markers
        for name, y_data in self.dependents_data.items():
            fig.add_trace(go.Scatter(
                x=self.x_sorted,
                y=y_data,
                mode='markers',
                name=f'{name} (Data)',
                marker=dict(size=6, opacity=0.6)
            ))
            
        fig.update_layout(title=f"Combined Data and Fit (s={self.smoothing})")
        return fig
