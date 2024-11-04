import numpy as np
import pandas as pd

# Define parameters
delta = 2.5  # Risk aversion parameter
tau = 0.05   # Scale factor for prior uncertainty

# Assume we have three factors: Value, Growth, and BMG
factors = ["Value", "Growth", "BMG"]

# Benchmark portfolio weights (e.g., equal weight)
w_bmk = np.array([1/3, 1/3, 1/3])

# Covariance matrix (assumed for simplicity)
cov_matrix = np.array([
    [0.02, 0.0018, 0.0015],
    [0.0018, 0.015, 0.0012],
    [0.0015, 0.0012, 0.025]
])

# Regime-based expected returns for each factor (bull or bear)
# Assume we have inferred these expected returns from the regime model
expected_returns_regimes = {
    "bull": np.array([0.06, 0.05, 0.04]),
    "bear": np.array([-0.02, 0.01, -0.03])
}

# Define views based on a detected "bull" regime
# Views are the expected returns of each factor in the bull regime
views = expected_returns_regimes["bear"]

# View matrix (each row represents a factor's deviation from market)
P = np.eye(len(factors))  # Identity matrix, as each view is specific to one factor

# Omega - confidence level in each view (diagonal matrix with small variances for high confidence)
omega = np.diag([0.02, 0.02, 0.02])  # Adjust values based on confidence level

# Prior (implied returns from benchmark, e.g., assume zero if neutral)
pi = np.zeros(len(factors))

# Black-Litterman calculations
# Calculate posterior expected returns for each factor
middle_term = np.linalg.inv(np.dot(np.dot(P, tau * cov_matrix), P.T) + omega)
adjustment = np.dot(np.dot(np.dot(tau * cov_matrix, P.T), middle_term), (views - np.dot(P, pi)))
posterior_returns = pi + adjustment

# Calculate the Black-Litterman weights
# w_BL = w_bmk + (1 / delta) * cov_matrix^-1 * posterior_returns
cov_inv = np.linalg.inv(cov_matrix)
w_bl_raw = w_bmk + (1 / delta) * np.dot(cov_inv, posterior_returns)

# Normalize weights to ensure they sum to one
w_bl = w_bl_raw / np.sum(w_bl_raw)

# Display results
factor_allocation = pd.DataFrame({
    "Factors": factors,
    "Benchmark Weights": w_bmk,
    "Posterior Expected Returns": posterior_returns,
    "Black-Litterman Weights": w_bl
})

print("Black-Litterman Portfolio Weights (Normalized):")
print(factor_allocation)
