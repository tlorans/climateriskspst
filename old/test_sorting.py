
import sympy as sp
import numpy as np


default_gamma = [1, 1, -1, 1, -1, -1]
N = 6  # Number of assets
print(default_gamma)
# Predefined values for beta and gamma
default_beta = [1, 1, 1, -1, -1, -1]
print(f"beta:{default_beta}")

# Convert beta and gamma inputs to Sympy matrices
gamma = sp.Matrix(default_gamma)
# Convert beta inputs to Sympy matrices
beta = sp.Matrix(default_beta)
# Convert beta and gamma inputs to NumPy arrays
beta_np = np.array(default_beta)
gamma_np = np.array(default_gamma)

# Get the indices of the sorted beta values
sorted_indices = np.argsort(beta_np)

print(f"Sorted indices by beta: {sorted_indices}")

# Get the top 3 (high beta) and bottom 3 (low beta) indices
high_beta_indices = sorted_indices[-3:]  # Indices for high beta
low_beta_indices = sorted_indices[:3]    # Indices for low beta
print(f"High beta indices: {high_beta_indices}")
print(f"Low beta indices: {low_beta_indices}")

# Sort the low beta assets by gamma
low_beta_high_gamma_sorted = low_beta_indices[np.argsort(gamma_np[low_beta_indices])][-1:]
print(f"Low beta high gamma: {low_beta_high_gamma_sorted}, gamma: {gamma_np[low_beta_high_gamma_sorted]}, beta: {beta_np[low_beta_high_gamma_sorted]}")

low_beta_low_gamma_sorted = low_beta_indices[np.argsort(gamma_np[low_beta_indices])][:1]
print(f"low beta low gamma: {low_beta_low_gamma_sorted}, gamma: {gamma_np[low_beta_low_gamma_sorted]}, beta: {beta_np[low_beta_low_gamma_sorted]}")

high_beta_high_gamma_sorted = high_beta_indices[np.argsort(gamma_np[high_beta_indices])][-1:]
print(f"High beta high gamma: {high_beta_high_gamma_sorted}, gamma: {gamma_np[high_beta_high_gamma_sorted]}, beta: {beta_np[high_beta_high_gamma_sorted]}")

high_beta_low_gamma_sorted = high_beta_indices[np.argsort(gamma_np[high_beta_indices])][:1]
print(f"High beta low gamma: {high_beta_low_gamma_sorted}, gamma: {gamma_np[high_beta_low_gamma_sorted]}, beta: {beta_np[high_beta_low_gamma_sorted]}")

# Combine the long and short positions
long = np.concatenate([low_beta_high_gamma_sorted, high_beta_high_gamma_sorted])
short = np.concatenate([low_beta_low_gamma_sorted, high_beta_low_gamma_sorted])

print(f"Long: {long}")
print(f"Short: {short}")

# Use SymPy's Rational to keep weights as fractions
w_long = sp.Matrix([0] * N)
w_short = sp.Matrix([0] * N)

# Assign long positions (1/3) to the selected assets
for idx in long:
    w_long[idx] = sp.Rational(1, 2)

# Assign short positions (-1/3) to the selected assets
for idx in short:
    w_short[idx] = sp.Rational(-1, 2)

# Combine long and short positions to form the final weight vector
w = w_long + w_short

print(w)

# print(gamma_np[np.argsort(gamma_np[sorted_indices][-3:])[-1:]])
# print(beta_np[np.argsort(gamma_np[sorted_indices][-3:])[-1:]])

# # print(np.argsort(gamma_np[sorted_indices][-3:])[:1])

# print(gamma_np[np.argsort(gamma_np[sorted_indices][-3:])[:1]])
# print(beta_np[np.argsort(gamma_np[sorted_indices][-3:])[:1]])

# print(gamma_np[np.argsort(gamma_np[sorted_indices][3:])[:1]])
# print(beta_np[np.argsort(gamma_np[sorted_indices][3:])[:1]])


# # Use SymPy's Rational to keep weights as fractions
# w_long = sp.Matrix([0, 0, 0, 0, 0, 0])
# w_short = sp.Matrix([0, 0, 0, 0, 0, 0])

# # Assign long positions (1/3) to the top 3 assets
# for idx in sorted_indices[-3:]:
#     w_long[idx] =sp.Rational(1, 3)

# # Assign short positions (-1/3) to the bottom 3 assets
# for idx in sorted_indices[:3]:
#     w_short[idx] = sp.Rational(-1, 3)

# # Combine long and short positions to form the final weight vector
# w = w_long + w_short

# print(w)