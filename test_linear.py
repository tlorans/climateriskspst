import sympy as sp

# Variances of the factors (diagonal of the original Omega)
sigma_factors = sp.Matrix([0.1, 0.15, 0.2, 0.12, 0.18])  # Standard deviations of factors
sigma_g = 0.05  # Standard deviation of the unrewarded risk factor (BMG)

# Sensitivity coefficients (beta) for each factor
beta_f = sp.Matrix([0.2, 0.3, 0.25, 0.1, 0.4])

# Original covariance matrix Omega (diagonal matrix with factor variances)
Omega = sp.diag(*[sigma_factors[i]**2 for i in range(5)])

# Outer product of beta_f vector with itself, scaled by sigma_g^2
beta_outer = beta_f * beta_f.T * sigma_g**2

# New covariance matrix Omega'
Omega_prime = Omega + beta_outer

# Display the matrices
print(r'\Omega = ' + sp.latex(Omega))
print(r'\beta \beta^T \sigma_g^2 = ' + sp.latex(beta_outer))
print(r'\Omega^\prime = \Omega + \beta \beta^T \sigma_g^2 = ' + sp.latex(Omega_prime))
