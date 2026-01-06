import numpy as np
import plotly.io as pio
from splines import ParametricSplines

# Generate synthetic data
def generate_data():
    np.random.seed(42)
    # Independent variable: CDS Level
    cds_level = np.sort(np.random.uniform(50, 500, 20))
    
    # Dependent variables with some noise
    # Beta decreases with level
    true_beta = 1000 / cds_level
    beta = true_beta + np.random.normal(0, 0.5, len(cds_level))
    
    # Intercept increases with level
    true_intercept = 500 + 2 * cds_level
    intercept = true_intercept + np.random.normal(0, 20, len(cds_level))
    
    return cds_level, beta, intercept

def main():
    cds_level, beta, intercept = generate_data()
    
    dependents = {
        'beta': beta,
        'intercept': intercept
    }
    
    print("Fitting Interpolating Spline (s=0)...")
    ps_interp = ParametricSplines(cds_level, dependents, smoothing=0.0)
    
    print("Fitting Smoothing Spline (s=5.0)...")
    ps_smooth = ParametricSplines(cds_level, dependents, smoothing=5.0) # Penalty is usually large for unnormalized sum-sq
    
    # IMPORTANT: The Reinsch algorithm s parameter scale depends on data scale and N.
    # Typically S is roughly N * sigma^2.
    # Our data variance is around 0.5^2 = 0.25 for beta, 20^2=400 for intercept. 
    # With direct sum of squares, we might need different S per variable? 
    # Current implementation uses one S.
    # Let's try separate classes or note that S might need tuning.
    
    # Query specific value
    target_level = 150.0
    query_res = ps_smooth.query(target_level)
    print(f"\nQuerying Level {target_level}:")
    print(f"Beta: {query_res['beta']:.4f}")
    print(f"Intercept: {query_res['intercept']:.4f}")
    
    # Visualization
    print("\nGenerating Plots...")
    fig_combined = ps_smooth.plot_combined()
    
    # fig_combined.show() # Uncomment to show locally if supported
    
    # For now, just print success
    print("Successfully generated plots.")

if __name__ == "__main__":
    main()
