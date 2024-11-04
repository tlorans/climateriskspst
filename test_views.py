import sympy as sp
import numpy as np 
from scipy.optimize import minimize

# Define the inputs
# Prior factor risk premia (implied from the market)
lambda_ = sp.Matrix([0.0381, -0.0012]) 

# Covariance matrix of prior factor risk premia (Γ_ψ)
Omega = sp.Matrix([[0.20**2, 0], [0, 0.20**2]]) 

# Manager's views on the factors (v_ψ)
lambda_views = sp.Matrix([0.05, 0.])  # Manager's views on the factors

# Uncertainty of the manager's views (Φ_ψ)
Phi_lambda_ = sp.Matrix([[0.10**2, 0], [0, 0.05**2]])  # View uncertainty (confidence level)

# Bayesian update formula to compute the posterior factor risk premia
Omega_posterior = (Omega.inv() + Phi_lambda_.inv()).inv()  # Updated covariance matrix
lambda_posterior = Omega_posterior * (Omega.inv() * lambda_ + Phi_lambda_.inv() * lambda_views)


# Display the posterior factor risk premia
print("Posterior factor risk premia (\lambda):")
print((lambda_posterior))

# Display the posterior covariance matrix of factor risk premia
print("Posterior covariance matrix of factor risk premia (omega):")
print(Omega_posterior)

# Setting up the mean-variance optimization for factor allocations using the posterior risk premia and covariance matrix

# Convert sympy matrices to numpy arrays for further calculations
lambda_posterior_np = np.array(lambda_posterior).astype(float).flatten()
Omega_posterior_np = np.array(Omega_posterior).astype(float)

# Define the risk tolerance coefficient gamma for the mean-variance optimization
gamma = 0.5  # example value

# Define the objective function for the mean-variance optimization problem
def objective_factors(y):
    # y represents the factor allocations
    return 0.5 * y.T @ Omega_posterior_np @ y - gamma * y.T @ lambda_posterior_np

# Constraints: weights sum to 1 (fully invested in one of the factors), weights are non-negative (long-only constraint)
constraints_factors = [
    {'type': 'eq', 'fun': lambda y: np.sum(y) - 1},  # sum of weights == 1 (fully invested in one factor)
    {'type': 'ineq', 'fun': lambda y: y}  # y >= 0 (long-only constraint)
]


# Initial guess for the factor weights (equal allocation across factors)
initial_weights_factors = np.ones(len(lambda_posterior_np)) / len(lambda_posterior_np)

# Perform optimization to find the optimal factor allocation
result_factors = minimize(objective_factors, initial_weights_factors, constraints=constraints_factors, method='SLSQP')

# Output the optimized factor allocations
print(result_factors.x)

