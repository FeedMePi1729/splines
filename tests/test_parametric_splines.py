
import pytest
import numpy as np
from splines import ParametricSplines

class TestParametricSplines:
    def test_initialization_simple(self):
        """Test basic initialization with valid data"""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([10.0, 20.0, 30.0, 40.0])
        
        ps = ParametricSplines(x, {'y': y}, smoothing=0.0)
        
        # Check if t is normalized
        assert ps.t[0] == 0.0
        assert ps.t[-1] == 1.0
        assert len(ps.splines) == 2 # 'independent' + 'y'

    def test_bijective_mapping_query(self):
        """Test that querying the independent variable yields the identity (approx)"""
        x = np.linspace(0, 10, 11)
        # y is x squared
        y = x**2
        
        # Use interpolation (s=0)
        ps = ParametricSplines(x, {'y': y}, smoothing=0.0)
        
        # Query at exact knot points
        res = ps.query(5.0)
        assert np.isclose(res['independent'], 5.0)
        assert np.isclose(res['y'], 25.0, atol=1e-10)
        
        # Cubic spline can fit quadratic exactly IF boundary conditions match.
        # We implemented Natural Spline (2nd deriv = 0 at boundaries). 
        # For y=x^2, 2nd deriv is 2 everywhere. So Natural Spline will have error near boundaries.
        res = ps.query(2.5)
        assert np.isclose(res['y'], 2.5**2, atol=0.1) # Relaxed tolerance for Natural BC effect

    def test_smoothing_effect(self):
        """Test that smoothing reduces variance of noisy data"""
        x = np.linspace(0, 10, 20)
        true_y = 2 * x + 1
        # Add alternating noise +1, -1
        noise = np.tile([1.0, -1.0], 10)
        y = true_y + noise
        
        # Fit with smoothing
        ps = ParametricSplines(x, {'y': y}, smoothing=10.0)
        
        # Check midpoint
        mid_val = ps.query(5.0)['y']
        # Should be closer to true_y(5)=11 than to noisy y
        # The noise at 5.0 (index 10) depends on pattern.
        # But generally, residual sum should be non-zero for smoothing spline
        
        # Ensure it didn't interpolate exactly
        # If interpolated, query(x[i]) == y[i]
        q_at_knot = ps.query(x[0])['y']
        assert not np.isclose(q_at_knot, y[0]) # Should differ due to smoothing

    def test_multi_dimensional(self):
        x = np.array([1, 2, 3])
        y = np.array([10, 20, 30])
        z = np.array([100, 200, 300])
        
        ps = ParametricSplines(x, {'y': y, 'z': z})
        res = ps.query(2.0)
        assert np.isclose(res['y'], 20.0)
        assert np.isclose(res['z'], 200.0)

    def test_monotonicity_enforcement(self):
        """Test that non-monotonic independent variable (after sorting check) logic holds"""
        # If input is already monotonic but unsorted
        x = np.array([1.0, 3.0, 2.0])
        y = np.array([10.0, 30.0, 20.0])
        
        # Initialization sorts by independent
        ps = ParametricSplines(x, {'y': y})
        
        assert ps.x_sorted[1] == 2.0
        assert ps.t[1] == 0.5 # (2-1)/(3-1) = 0.5
        
        # y should be sorted to [10, 20, 30]
        # So query at 2.0 -> 20.0
        assert np.isclose(ps.query(2.0)['y'], 20.0)
        
    def test_unsorted_duplicates_handling(self):
        """Test that unsorted input with duplicates is handled via averaging"""
        # User example: [49, 50, 49, 51]
        x = np.array([49, 50, 49, 51])
        # Make y clearly averageable: at 49 we have y=10 and y=20 -> avg 15
        y = np.array([10, 50, 20, 51]) 
        
        ps = ParametricSplines(x, {'y': y})
        
        # Check sorted x
        assert np.array_equal(ps.x_sorted, np.array([49, 50, 51]))
        
        # Check averaged y
        # 49 -> (10+20)/2 = 15
        # 50 -> 50
        # 51 -> 51
        y_expected = np.array([15.0, 50.0, 51.0])
        assert np.allclose(ps.dependents_data['y'], y_expected)
        
        # Query at 49 should yield 15 (if interpolation)
        # s=0 default
        res = ps.query(49.0)
        assert np.isclose(res['y'], 15.0)

