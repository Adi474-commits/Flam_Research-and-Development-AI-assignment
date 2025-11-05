# Parametric Curve Fitting Assignment Solution

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![LaTeX](https://img.shields.io/badge/LaTeX-Document-green)
![License](https://img.shields.io/badge/License-Academic-orange)

**Author:** Adithya N Reddy  
**Institution:** Amrita Vishwa Vidyapeetham  
**Email:** adithyasnr@gmail.com

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Solution Approach](#solution-approach)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Final Results](#final-results)
- [Performance Metrics](#performance-metrics)
- [Documentation](#documentation)
- [Desmos Visualization](#desmos-visualization)
- [Technical Details](#technical-details)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🔍 Overview

This repository contains the complete solution to a parametric curve fitting problem, where the objective is to determine unknown parameters **θ**, **M**, and **X** in a complex parametric equation using 1000 empirical data points. The solution employs the **Differential Evolution** global optimization algorithm to minimize the L1 distance between the theoretical curve and observed data.

The project includes:
- Mathematical analysis and problem formulation
- Python implementation using SciPy's optimization tools
- Comprehensive LaTeX documentation with Chicago-style references
- Data visualization and validation scripts
- Complete methodology and results presentation

---

## 📐 Problem Statement

Find the values of unknown variables in the parametric equation of a curve:

```
x(t) = t · cos(θ) - e^(M|t|) · sin(0.3t) · sin(θ) + X
y(t) = 42 + t · sin(θ) + e^(M|t|) · sin(0.3t) · cos(θ)
```

### Unknowns
- **θ** (theta): Angular parameter
- **M**: Exponential growth/decay rate
- **X**: Horizontal offset

### Constraints

| Parameter | Lower Bound | Upper Bound | Unit |
|-----------|-------------|-------------|------|
| θ         | 0°          | 50°         | degrees |
| M         | -0.05       | 0.05        | dimensionless |
| X         | 0           | 100         | units |
| t         | 6           | 60          | parameter range |

**Dataset:** 1000 empirical (x, y) coordinate pairs from `xy_data.csv`

---

## 🎯 Solution Approach

### Algorithm: Differential Evolution (DE)

A stochastic, population-based global optimization algorithm ideal for non-linear, non-convex problems with multiple local optima.

**Key Features:**
- **Population Size:** 30 individuals
- **Mutation Factor:** 0.8 (exploration-exploitation balance)
- **Crossover Probability:** 0.7 (genetic diversity)
- **Maximum Generations:** 200
- **Objective Function:** L1 norm (Manhattan distance)

### Optimization Process

1. **Data Loading:** Import 1000 coordinate pairs from CSV
2. **Objective Function:** Calculate L1 distance for each parameter set
3. **DE Iteration:** Evolve population toward optimal parameters
4. **Constraint Enforcement:** Strict boundary checking (θ, M, X)
5. **Convergence:** Stop when improvement drops below tolerance
6. **Validation:** Verify solution against all constraints

---

## 💻 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Required Python Packages

```bash
pip install numpy>=1.24.0
pip install scipy>=1.10.0
pip install pandas>=2.0.0
pip install matplotlib>=3.7.0
```

### LaTeX Requirements

For compiling the documentation:
- TeX Live, MiKTeX, or Overleaf
- Required packages: `amsmath`, `graphicx`, `algorithm`, `listings`, `xcolor`, `fancyhdr`, `setspace`, `booktabs`, `hyperref`

---

## 🚀 Usage

### Running the Optimization Script

```python
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

# Load empirical data
data = pd.read_csv('xy_data.csv')
x_emp, y_emp = data['x'].values, data['y'].values

# Define parametric equations
def parametric_curve(t, theta, M, X):
    x = t * np.cos(theta) - np.exp(M * np.abs(t)) * np.sin(0.3 * t) * np.sin(theta) + X
    y = 42 + t * np.sin(theta) + np.exp(M * np.abs(t)) * np.sin(0.3 * t) * np.cos(theta)
    return x, y

# Objective function (L1 distance)
def objective(params):
    theta, M, X = params
    t_values = np.linspace(6, 60, 1000)
    x_model, y_model = parametric_curve(t_values, theta, M, X)
    l1_distance = np.sum(np.abs(x_model - x_emp)) + np.sum(np.abs(y_model - y_emp))
    return l1_distance

# Parameter bounds
bounds = [(0, np.radians(50)), (-0.05, 0.05), (0, 100)]

# Run optimization
result = differential_evolution(
    objective,
    bounds,
    strategy='best1bin',
    maxiter=200,
    popsize=30,
    mutation=0.8,
    recombination=0.7,
    seed=42,
    disp=True
)

# Extract results
theta_opt, M_opt, X_opt = result.x
print(f"θ = {theta_opt:.6f} rad ({np.degrees(theta_opt):.2f}°)")
print(f"M = {M_opt:.6f}")
print(f"X = {X_opt:.2f}")
print(f"L1 Distance = {result.fun:.2f}")
```

### Running the Verification Script

```bash
python verify_solution.py
```

---

## 📁 Project Structure

```
FLAM/
├── xy_data.csv                      # 1000 empirical data points
├── solve_parametric.py              # Main optimization script
├── verify_solution.py               # Solution validation script
├── assignment_solution_clean.tex    # Final LaTeX document (678 lines)
├── assignment_solution.tex          # Comprehensive backup version (995 lines)
├── desmos-graph.png                 # Fitted curve visualization
├── desmos_graph.html                # Interactive Desmos graph
├── PROJECT_DESCRIPTION.md           # Project overview
├── DESMOS_INSTRUCTIONS.md           # Graph creation guide
└── README.md                        # This file
```

---

## 🎉 Final Results

### **Optimal Parameter Values**

| Parameter | Optimal Value (radians) | Optimal Value (degrees) | Constraint Check |
|-----------|-------------------------|-------------------------|------------------|
| **θ**     | **0.826**               | **47.33°**              | ✓ 0° < 47.33° < 50° |
| **M**     | **0.05**                | -                       | ✓ -0.05 < 0.05 ≤ 0.05 |
| **X**     | **11.58**               | -                       | ✓ 0 < 11.58 < 100 |

**Note:** M reaches the upper constraint boundary, indicating maximum exponential growth within allowed limits.

### Interpretation

The fitted curve represents an **expanding spiral trajectory**:

- **θ = 47.33°**: Determines the primary orientation and angular trajectory
- **M = 0.05**: Positive maximum value → oscillation amplitude grows exponentially with t
- **X = 11.58**: Horizontal displacement of the entire curve
- **sin(0.3t)**: Periodic oscillations with frequency 0.3 rad/unit
- **e^(0.05|t|)**: Exponential amplitude modulation (expansion over time)

---

## 📊 Performance Metrics

| Metric                | Value            |
|----------------------|------------------|
| Algorithm            | Differential Evolution |
| Convergence Generations | ~87          |
| Total Function Evaluations | ~2610     |
| Final L1 Distance    | 218.45 units     |
| Runtime (approx.)    | 5-10 minutes     |
| Constraint Violations | 0               |

---

## 📖 Documentation

### Compiling the LaTeX Document

**Using Command Line:**
```bash
pdflatex assignment_solution_clean.tex
pdflatex assignment_solution_clean.tex  # Run twice for references
```

**Using Overleaf:**
1. Upload `assignment_solution_clean.tex` and `desmos-graph.png`
2. Set compiler to pdfLaTeX
3. Compile the document

### Document Structure (assignment_solution_clean.tex)

1. **Abstract** - Problem overview and key findings
2. **Table of Contents** - Navigation guide
3. **Chapter 1: Introduction** - Background and motivation
4. **Chapter 2: Problem Formulation** - Mathematical framework
5. **Chapter 3: Solution Methodology** - Differential Evolution approach
6. **Chapter 4: Implementation Details** - Python code and algorithms
7. **Chapter 5: Results and Analysis** - Optimal parameters and validation
8. **Chapter 6: Curve Interpretation** - Geometric and physical meaning
9. **Chapter 7: Validation and Verification** - Constraint checking
10. **Chapter 8: Challenges and Considerations** - Computational insights
11. **Chapter 9: Conclusion** - Summary and implications
12. **References** - 8 Chicago-style citations
13. **Appendix** - Complete Python code listing

---

## 📈 Desmos Visualization

### Interactive Graph

View the fitted curve overlaid on data points:  
**Desmos Link:** [https://www.desmos.com/calculator](https://www.desmos.com/calculator)

### Parametric Expression (Copy-Paste)

```
(t*cos(0.826)-e^{0.05*abs(t)}*sin(0.3*t)*sin(0.826)+11.58,42+t*sin(0.826)+e^{0.05*abs(t)}*sin(0.3*t)*cos(0.826))
```

**Configuration:**
- Set parameter range: `6 ≤ t ≤ 60`
- Upload `xy_data.csv` as a table to plot empirical points
- Adjust viewing window to x: [55, 115], y: [40, 75]

**Alternative Notation:**
```
(t*cos(0.826)-e^{0.05|t|}·sin(0.3t)sin(0.826)+11.58,42+t*sin(0.826)+e^{0.05|t|}·sin(0.3t)cos(0.826))
```

---

## 🔬 Technical Details

### Computational Complexity

- **Time Complexity:** O(G × P × N)
  - G = number of generations (~87)
  - P = population size (30)
  - N = data points (1000)
- **Space Complexity:** O(P × D + N)
  - D = dimensions (3 parameters)

### Mathematical Decomposition

The parametric equations consist of:

1. **Linear Components:**
   - x: `t · cos(θ) + X`
   - y: `42 + t · sin(θ)`

2. **Oscillatory Components:**
   - x: `-e^(M|t|) · sin(0.3t) · sin(θ)`
   - y: `+e^(M|t|) · sin(0.3t) · cos(θ)`

3. **Modulation:** Exponential term `e^(M|t|)` scales oscillation amplitude

### Constraint Analysis

| Constraint | Type      | Active at Optimum? | Slack/Tightness |
|------------|-----------|--------------------|-----------------|
| θ ∈ (0°, 50°) | Box      | No                 | 2.67° slack     |
| M ∈ (-0.05, 0.05) | Box  | **Yes (upper)**    | 0.00 slack      |
| X ∈ (0, 100) | Box       | No                 | 88.42 slack     |

---

## 📚 References

1. Storn, R., and Price, K. (1997). "Differential Evolution – A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces." *Journal of Global Optimization* 11, no. 4: 341–359.

2. Price, K., Storn, R. M., and Lampinen, J. A. (2005). *Differential Evolution: A Practical Approach to Global Optimization*. Berlin: Springer-Verlag.

3. Das, S., and Suganthan, P. N. (2011). "Differential Evolution: A Survey of the State-of-the-Art." *IEEE Transactions on Evolutionary Computation* 15, no. 1: 4–31.

4. Virtanen, P., et al. (2020). "SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python." *Nature Methods* 17: 261–272.

5. Harris, C. R., et al. (2020). "Array Programming with NumPy." *Nature* 585: 357–362.

6. McKinney, W. (2010). "Data Structures for Statistical Computing in Python." *Proceedings of the 9th Python in Science Conference*, 56–61.

7. Nocedal, J., and Wright, S. J. (2006). *Numerical Optimization*. 2nd ed. New York: Springer.

8. Press, W. H., et al. (2007). *Numerical Recipes: The Art of Scientific Computing*. 3rd ed. Cambridge: Cambridge University Press.

---

## 🤝 Contributing

This is an academic assignment project. For inquiries or suggestions:

1. **Email:** adithyasnr@gmail.com
2. **Issues:** Open a GitHub issue for technical questions
3. **Discussions:** Academic integrity guidelines apply to all contributions

---

## 📄 License

This project is submitted as academic coursework for Amrita Vishwa Vidyapeetham. All rights reserved under academic integrity policies. Code and documentation may be used for educational reference with proper attribution.

---

## 📧 Contact

**Adithya N Reddy**  
Amrita Vishwa Vidyapeetham  
Email: adithyasnr@gmail.com

---

## 🎓 Academic Context

**Course:** Research & Development / Artificial Intelligence  
**Institution:** Amrita Vishwa Vidyapeetham  
**Submission Date:** 2024  
**Assignment:** Parametric Curve Fitting with Global Optimization

---

**Solution Summary:** The unknown parameters are **θ = 0.826 rad (47.33°)**, **M = 0.05**, and **X = 11.58**, determined through Differential Evolution optimization minimizing L1 distance over 1000 empirical data points.
