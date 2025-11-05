import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('xy_data.csv')
x_data = data['x'].values
y_data = data['y'].values

# Number of data points
n_points = len(x_data)

# Define the parametric equations
def parametric_curve(t, theta, M, X):
    """
    x = (t * cos(θ) - e^(M|t|) * sin(0.3t) * sin(θ) + X)
    y = (42 + t * sin(θ) + e^(M|t|) * sin(0.3t) * cos(θ))
    """
    theta_rad = np.deg2rad(theta)
    
    exp_term = np.exp(M * np.abs(t))
    sin_term = np.sin(0.3 * t)
    
    x = t * np.cos(theta_rad) - exp_term * sin_term * np.sin(theta_rad) + X
    y = 42 + t * np.sin(theta_rad) + exp_term * sin_term * np.cos(theta_rad)
    
    return x, y

# Objective function: minimize the L1 distance
def objective_function(params):
    """
    Compute the L1 distance between the data points and the parametric curve
    """
    theta, M, X = params
    
    # Generate t values corresponding to the data points
    # We need to estimate t for each data point
    # Use a simple approach: assume t is uniformly distributed
    t_values = np.linspace(6, 60, n_points)
    
    x_pred, y_pred = parametric_curve(t_values, theta, M, X)
    
    # Compute L1 distance
    l1_distance = np.sum(np.abs(x_data - x_pred) + np.abs(y_data - y_pred))
    
    return l1_distance

# Alternative objective: find t values that minimize distance for each point
def objective_function_optimized(params):
    """
    For each data point, find the best t value, then compute L1 distance
    """
    theta, M, X = params
    
    total_distance = 0
    t_range = np.linspace(6, 60, 1000)
    
    for i in range(n_points):
        x_target, y_target = x_data[i], y_data[i]
        
        # For each t, compute distance to target point
        x_curve, y_curve = parametric_curve(t_range, theta, M, X)
        distances = np.abs(x_curve - x_target) + np.abs(y_curve - y_target)
        
        # Find minimum distance
        min_dist = np.min(distances)
        total_distance += min_dist
    
    return total_distance

# Define bounds for the parameters
bounds = [
    (0, 50),      # theta in degrees
    (-0.05, 0.05), # M
    (0, 100)       # X
]

print("Starting optimization...")
print("This may take a few minutes...\n")

# Use differential evolution for global optimization
result = differential_evolution(
    objective_function_optimized,
    bounds,
    seed=42,
    maxiter=200,
    popsize=30,
    tol=0.01,
    atol=0.01,
    workers=-1,
    updating='deferred',
    disp=True
)

theta_opt, M_opt, X_opt = result.x

print("\n" + "="*60)
print("OPTIMIZATION RESULTS")
print("="*60)
print(f"Optimal θ (theta): {theta_opt:.6f} degrees")
print(f"Optimal M:         {M_opt:.6f}")
print(f"Optimal X:         {X_opt:.6f}")
print(f"L1 Distance:       {result.fun:.6f}")
print("="*60)

# Convert theta to radians for the final equation
theta_rad_opt = np.deg2rad(theta_opt)
print(f"\nθ in radians:      {theta_rad_opt:.6f}")

# Generate the curve for visualization
t_plot = np.linspace(6, 60, 1000)
x_plot, y_plot = parametric_curve(t_plot, theta_opt, M_opt, X_opt)

# Plot the results
plt.figure(figsize=(12, 8))
plt.scatter(x_data, y_data, c='blue', s=10, alpha=0.5, label='Data Points')
plt.plot(x_plot, y_plot, 'r-', linewidth=2, label='Fitted Curve')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Parametric Curve Fitting')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.savefig('curve_fit.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nPlot saved as 'curve_fit.png'")

# Save results to file
with open('optimization_results.txt', 'w') as f:
    f.write("PARAMETRIC CURVE FITTING RESULTS\n")
    f.write("="*60 + "\n\n")
    f.write("Original Equations:\n")
    f.write("x = (t * cos(θ) - e^(M|t|) * sin(0.3t) * sin(θ) + X)\n")
    f.write("y = (42 + t * sin(θ) + e^(M|t|) * sin(0.3t) * cos(θ))\n\n")
    f.write("Unknowns: θ, M, X\n")
    f.write("Parameter t range: 6 < t < 60\n\n")
    f.write("OPTIMAL VALUES:\n")
    f.write("-"*60 + "\n")
    f.write(f"θ (theta) = {theta_opt:.6f} degrees\n")
    f.write(f"θ (theta) = {theta_rad_opt:.6f} radians\n")
    f.write(f"M         = {M_opt:.6f}\n")
    f.write(f"X         = {X_opt:.6f}\n")
    f.write(f"L1 Distance = {result.fun:.6f}\n")
    f.write("-"*60 + "\n\n")
    f.write("Desmos Format:\n")
    f.write(f"\\left(t*\\cos({theta_rad_opt:.6f})-e^{{{M_opt:.6f}\\left|t\\right|}}\\cdot\\sin(0.3t)\\sin({theta_rad_opt:.6f})\\ +{X_opt:.6f},42+\\ t*\\sin({theta_rad_opt:.6f})+e^{{{M_opt:.6f}\\left|t\\right|}}\\cdot\\sin(0.3t)\\cos({theta_rad_opt:.6f})\\right)\n")

print("\nResults saved to 'optimization_results.txt'")
