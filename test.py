import numpy as np

# Inputs
K = 2  # Number of rewarded factors

# Covariance matrix of the factors (Omega)
Sigma_f = np.diag([0.04, 0.01])  # Factor covariance matrix (diagonal for uncorrelated factors)

# Factor risk premia (lambda)
lambda_f = np.array([0.03, 0.02])

# Calculate the inverse of the covariance matrix
Sigma_f_inv = np.linalg.inv(Sigma_f)

# Compute gamma = (1_n^\top Sigma_f_inv * mu_f)^{-1}
ones_vector = np.ones(K)
gamma = 1 / (ones_vector.T @ Sigma_f_inv @ lambda_f)

# Compute optimal weights for the factors: w^* = gamma * Sigma_f_inv * lambda_f
optimal_weights = gamma * Sigma_f_inv @ lambda_f

# Expected portfolio return
optimal_return = optimal_weights.T @ lambda_f

# Portfolio variance and standard deviation
optimal_variance = optimal_weights.T @ Sigma_f @ optimal_weights
optimal_std_dev = np.sqrt(optimal_variance)

# Sharpe ratio
optimal_sharpe_ratio = optimal_return / optimal_std_dev

# Output
print("Optimal Weights on Factors:", optimal_weights)
print("Expected Portfolio Return:", optimal_return)
print("Portfolio Volatility:", optimal_std_dev)
print("Maximized Sharpe Ratio:", optimal_sharpe_ratio)
