import sympy as sp

# Define the inputs
# Prior factor risk premia (implied from the market)
lambda_ = sp.Matrix([0.0381, -0.0012])  # Implied risk premia for the two factors

# Covariance matrix of prior factor risk premia (Γ_ψ)
Omega = sp.Matrix([[0.20**2, 0], [0, 0.20**2]])  # Prior uncertainties (sigma_ψ²)

# Manager's views on the factors (v_ψ)
lambda_views = sp.Matrix([0.05, 0])  # Manager's views on the factors

# Uncertainty of the manager's views (Φ_ψ)
Phi_lambda_ = sp.Matrix([[0.10**2, 0], [0, 0.05**2]])  # View uncertainty (confidence level)

# Bayesian update formula to compute the posterior factor risk premia
Omega_posterior = (Omega.inv() + Phi_lambda_.inv()).inv()  # Updated covariance matrix
lambda_posterior = Omega_posterior * (Omega.inv() * lambda_ + Phi_lambda_.inv() * lambda_views)


# Display the posterior factor risk premia
print("Posterior factor risk premia (\lambda):")
print((sp.latex(lambda_posterior)))

# Display the posterior covariance matrix of factor risk premia
print("Posterior covariance matrix of factor risk premia (omega):")
print(sp.latex(Omega_posterior))
