import sympy as sp

# Define the inputs
# Prior factor risk premia (implied from the market)
psi_prior = sp.Matrix([3.81, -0.12])  # Implied risk premia for the two factors

# Covariance matrix of prior factor risk premia (Γ_ψ)
Gamma_psi_prior = sp.Matrix([[0.04, 0], [0, 0.04]])  # Prior uncertainties (sigma_ψ²)

# Manager's views on the factors (v_ψ)
views = sp.Matrix([5, 0])  # Manager's views on the factors

# Uncertainty of the manager's views (Φ_ψ)
Phi_psi = sp.Matrix([[0.10**2, 0], [0, 0.05**2]])  # View uncertainty (confidence level)

# Bayesian update formula to compute the posterior factor risk premia
Gamma_posterior_inv = (Gamma_psi_prior.inv() + Phi_psi.inv()).inv()  # Updated covariance matrix
psi_posterior = Gamma_posterior_inv * (Gamma_psi_prior.inv() * psi_prior + Phi_psi.inv() * views)

# Posterior covariance matrix for factor risk premia
Gamma_psi_posterior = Gamma_posterior_inv

# Display the posterior factor risk premia
print("Posterior factor risk premia (ψ̄):")
print((sp.latex(psi_posterior)))

# Display the posterior covariance matrix of factor risk premia
print("Posterior covariance matrix of factor risk premia (Γ̄_ψ):")
print(sp.latex(Gamma_psi_posterior))

# Now, update the asset risk premia (π) using the posterior factor risk premia
# B is the asset-factor loading matrix
B = sp.Matrix([[0.5, -0.3], [1.2, 0.6], [1.4, -0.3], [0.7, 0.3], [0.8, -0.9]])

# Posterior asset risk premia (π̄)
pi_posterior = B * psi_posterior

# Display the posterior asset risk premia
print("Posterior asset risk premia (π̄):")
print(sp.latex(pi_posterior))

# Example posterior covariance of the assets (Sigma)
D = sp.diag(0.12**2, 0.10**2, 0.08**2, 0.10**2, 0.12**2)  # Idiosyncratic risk matrix (D)
Omega = sp.diag(0.20**2, 0.05**2)  # Factor covariance matrix (Omega)

# Updated covariance matrix of asset returns (Σ̄)
Sigma_posterior = B * Omega * B.T + D

# Display the updated covariance matrix
print("Posterior covariance matrix of asset returns (Σ̄):")
print(sp.latex(Sigma_posterior))

# Optional: You can also compute the optimal portfolio weights based on the updated asset risk premia
gamma = 1  # Example risk tolerance
w_star_posterior = gamma * Sigma_posterior.inv() * pi_posterior

# Display optimal portfolio weights using posterior risk premia
print("Optimal portfolio weights (w* using posterior risk premia):")
print(sp.latex(w_star_posterior))
