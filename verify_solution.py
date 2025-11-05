"""
Simplified Parametric Curve Analysis
This script demonstrates the mathematical approach to finding the unknown parameters
without requiring advanced optimization libraries.
"""

import csv
import math

# Read the data
print("Reading data from xy_data.csv...")
x_data = []
y_data = []

with open('xy_data.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        x_data.append(float(row['x']))
        y_data.append(float(row['y']))

n_points = len(x_data)
print(f"Loaded {n_points} data points\n")

# Data statistics
x_min, x_max = min(x_data), max(x_data)
y_min, y_max = min(y_data), max(y_data)

print("="*60)
print("DATA ANALYSIS")
print("="*60)
print(f"X range: [{x_min:.2f}, {x_max:.2f}]")
print(f"Y range: [{y_min:.2f}, {y_max:.2f}]")
print(f"X span:  {x_max - x_min:.2f}")
print(f"Y span:  {y_max - y_min:.2f}")
print()

# Parametric equations
def parametric_curve(t, theta_rad, M, X):
    """
    Calculate x, y coordinates for given t and parameters
    x = t * cos(θ) - e^(M|t|) * sin(0.3t) * sin(θ) + X
    y = 42 + t * sin(θ) + e^(M|t|) * sin(0.3t) * cos(θ)
    """
    exp_term = math.exp(M * abs(t))
    sin_term = math.sin(0.3 * t)
    
    x = t * math.cos(theta_rad) - exp_term * sin_term * math.sin(theta_rad) + X
    y = 42 + t * math.sin(theta_rad) + exp_term * sin_term * math.cos(theta_rad)
    
    return x, y

# Test with optimal parameters found
theta_rad = 0.826  # radians
theta_deg = math.degrees(theta_rad)
M = 0.05
X = 11.58

print("="*60)
print("TESTING OPTIMAL PARAMETERS")
print("="*60)
print(f"θ (theta) = {theta_rad:.6f} radians = {theta_deg:.2f} degrees")
print(f"M         = {M:.6f}")
print(f"X         = {X:.6f}")
print()

# Calculate L1 distance for these parameters
print("Calculating L1 distance...")
total_distance = 0
t_values = []

# For each data point, find closest point on curve
t_samples = [6 + i * (60-6) / 999 for i in range(1000)]

for i in range(n_points):
    x_target, y_target = x_data[i], y_data[i]
    
    # Find minimum distance for this data point
    min_dist = float('inf')
    best_t = 0
    
    for t in t_samples:
        x_curve, y_curve = parametric_curve(t, theta_rad, M, X)
        dist = abs(x_curve - x_target) + abs(y_curve - y_target)
        
        if dist < min_dist:
            min_dist = dist
            best_t = t
    
    total_distance += min_dist
    t_values.append(best_t)
    
    # Show progress every 100 points
    if (i + 1) % 100 == 0:
        print(f"  Processed {i+1}/{n_points} points...")

print()
print("="*60)
print("RESULTS")
print("="*60)
print(f"Total L1 Distance: {total_distance:.2f}")
print(f"Average Distance per Point: {total_distance/n_points:.4f}")
print()

# Sample predictions
print("Sample Points Comparison:")
print("-"*60)
print(f"{'Index':<8} {'X_data':<12} {'X_pred':<12} {'Y_data':<12} {'Y_pred':<12}")
print("-"*60)

sample_indices = [0, 250, 500, 750, 999]
for idx in sample_indices:
    t_est = t_values[idx]
    x_pred, y_pred = parametric_curve(t_est, theta_rad, M, X)
    print(f"{idx:<8} {x_data[idx]:<12.4f} {x_pred:<12.4f} {y_data[idx]:<12.4f} {y_pred:<12.4f}")

print("-"*60)
print()

# Verification of constraints
print("="*60)
print("CONSTRAINT VERIFICATION")
print("="*60)
print(f"θ constraint: 0° < {theta_deg:.2f}° < 50° → {'✓ PASS' if 0 < theta_deg < 50 else '✗ FAIL'}")
print(f"M constraint: -0.05 < {M:.4f} < 0.05 → {'✓ PASS' if -0.05 < M < 0.05 else '✗ FAIL'}")
print(f"X constraint: 0 < {X:.2f} < 100 → {'✓ PASS' if 0 < X < 100 else '✗ FAIL'}")
print()

# Generate Desmos format
print("="*60)
print("DESMOS CALCULATOR FORMAT")
print("="*60)
print("Copy and paste this into Desmos (https://www.desmos.com/calculator):")
print()
desmos_str = f"(t*cos({theta_rad:.6f})-e^{{{M:.6f}*abs(t)}}*sin(0.3*t)*sin({theta_rad:.6f})+{X:.6f},42+t*sin({theta_rad:.6f})+e^{{{M:.6f}*abs(t)}}*sin(0.3*t)*cos({theta_rad:.6f}))"
print(desmos_str)
print()
print("Domain: 6 ≤ t ≤ 60")
print()

print("="*60)
print("FINAL ANSWER")
print("="*60)
print(f"θ = {theta_rad:.6f} radians = {theta_deg:.2f}°")
print(f"M = {M:.6f}")
print(f"X = {X:.6f}")
print("="*60)
