import streamlit as st
import numpy as np
import plotly.graph_objs as go
import numpy as np
import sympy as sp
import sympy.stats as stats

st.title('Factor Efficient Portfolio')

st.write(r"""
         In the previous section, we have built an investment tool - a hedging portfolio - that helps to reduce the exposure to unrewarded risks,
         while keeping the expected return of the portfolio unchanged. 
         In this section, we will show how to combine the hedging portfolio with the initial portfolio to recover mean-variance efficiency.
         """)

st.subheader('Enhancing the Risk Return Profile')

st.write(r"""
         Our initial portfolio $c$ with default values for the loadings on the rewarded and unrewarded risks has the variance:
            """)

st.latex(r"""
\begin{equation}
         \sigma^2_c = \frac{2}{3}\sigma^2_{\epsilon} + 4 \sigma^2_f + \frac{2}{3}\sigma^2_g
\end{equation}
         """)

st.write(r"""
         and the variance of the hedging portfolio $h$ is:
            """)

st.latex(r"""
\begin{equation}
            \sigma^2_h = \sigma^2_{\epsilon} + 4 \sigma^2_g
\end{equation}
            """)

st.write(r"""
         Therefore, we can form a portfolio $p$ as a combination of the initial portfolio $c$ and the hedging portfolio $h$:
            """)

st.latex(r"""
\begin{equation}
         w_p = w_c - \frac{1}{3} w_h
\end{equation}
            """)


default_gamma = [1, 1, -1, 1, -1, -1]
N = 6  # Number of assets

# Predefined values for beta and gamma
default_beta = [1, 1, 1, -1, -1, -1]

# Convert beta and gamma inputs to Sympy matrices
gamma = sp.Matrix(default_gamma)
# Convert beta inputs to Sympy matrices
beta = sp.Matrix(default_beta)

# Portfolio weights based on sorted betas (long the highest, short the lowest)
beta_np = np.array(default_beta)
gamma_np = np.array(default_gamma)

# Get the indices of the sorted beta values
sorted_indices = np.argsort(beta_np)

# Use SymPy's Rational to keep weights as fractions
w_long = sp.Matrix([0, 0, 0, 0, 0, 0])
w_short = sp.Matrix([0, 0, 0, 0, 0, 0])

# Assign long positions (1/3) to the top 3 assets
for idx in sorted_indices[-3:]:
    w_long[idx] =sp.Rational(1, 3)

# Assign short positions (-1/3) to the bottom 3 assets
for idx in sorted_indices[:3]:
    w_short[idx] = sp.Rational(-1, 3)

# Combine long and short positions to form the final weight vector
w_c = w_long + w_short


# Get the top 3 (high beta) and bottom 3 (low beta) indices
high_beta_indices = sorted_indices[-3:]  # Indices for high beta
low_beta_indices = sorted_indices[:3]    # Indices for low beta

low_beta_high_gamma_sorted = low_beta_indices[np.argsort(gamma_np[low_beta_indices])][-1:]

low_beta_low_gamma_sorted = low_beta_indices[np.argsort(gamma_np[low_beta_indices])][:1]

high_beta_high_gamma_sorted = high_beta_indices[np.argsort(gamma_np[high_beta_indices])][-1:]

high_beta_low_gamma_sorted = high_beta_indices[np.argsort(gamma_np[high_beta_indices])][:1]

# Combine the long and short positions
long = np.concatenate([low_beta_high_gamma_sorted, high_beta_high_gamma_sorted])
short = np.concatenate([low_beta_low_gamma_sorted, high_beta_low_gamma_sorted])

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
w_h = w_long + w_short

w = w_c - sp.Rational(1, 3) * w_h

st.write(r"""
         The weights of the portfolio are:
         """)

# Prepare weights in LaTeX format as a row vector
weights_latex = r"\begin{bmatrix} "
for i in range(6):
    weights_latex += f"{w[i]} & "
weights_latex = weights_latex[:-2] + r" \end{bmatrix}"  # Remove the last "&" and close the matrix

st.latex(r"""
w_p^T = """ + weights_latex)


# Define priced factor as normal random variable with variance properties
f = stats.Normal('f', 0, sp.symbols('sigma_f'))  # Priced factor f with E[f] = 0 and var(f) = sigma_f^2
# Characteristic premium
lambda_ = sp.symbols('lambda')
# Define idiosyncratic errors epsilon_i as random variables with zero mean and variance sigma_epsilon^2
epsilon = sp.Matrix([stats.Normal(f'epsilon', 0, sp.symbols('sigma_epsilon')) for i in range(N)])
# Define priced and unpriced factors as normal random variables with variance properties
g = stats.Normal('g', 0, sp.symbols('sigma_g'))  # Unpriced factor g with E[g] = 0 and var(g) = sigma_g^2

# Define symbols for variances of the factors and idiosyncratic error
sigma_g = sp.symbols('sigma_g')
# Define symbols for variances of the factors and idiosyncratic error
sigma_f, sigma_epsilon = sp.symbols('sigma_f sigma_epsilon')
# Step 1: Define the portfolio return formula symbolically
portfolio_return_with_g = w.dot(beta * (f + lambda_) + gamma * g + epsilon)

st.write(r"""
         Because the hedging portfolio $h$ is designed to not alter the expected return of the portfolio, the expected return of the portfolio $p$ is the same as the expected return of the initial portfolio $c$:
                """)
# Step 2: Take the expectation using sympy.stats
expected_portfolio_return_with_g = stats.E(portfolio_return_with_g)
st.latex(f"E[r_p] = {sp.latex(expected_portfolio_return_with_g)}")

# Contribution from the unpriced factor g:
# LaTeX: Var_g = (w^\top \gamma)^2 \sigma_g^2
variance_g = (w.dot(gamma))**2 * sigma_g**2  # Contribution from unpriced factor g
# Contribution from the priced factor f:
# LaTeX: Var_f = (w^\top \beta)^2 \sigma_f^2
variance_f = (w.dot(beta))**2 * sigma_f**2  # Contribution from priced factor f
# Contribution from the idiosyncratic errors:
# LaTeX: Var_\epsilon = w^\top w \times \sigma_\epsilon^2
variance_epsilon = w.dot(w) * sigma_epsilon**2  # Contribution from idiosyncratic errors
# Total variance of the portfolio:
total_portfolio_variance_with_g = variance_f + variance_g + variance_epsilon

st.write(r"""
         The hedging portfolio $h$ is designed to reduce the variance of the portfolio by removing the exposure to the unrewarded risk. The variance of the portfolio $p$ is:
            """)

st.latex(f"\\sigma^2_p = {sp.latex(total_portfolio_variance_with_g)}")


st.write(r"""
         Which help to improve the Sharpe ratio of the portfolio:
            """)

# Calculate the Sharpe ratio
sharpe_ratio_with_g = expected_portfolio_return_with_g / sp.sqrt(total_portfolio_variance_with_g)
st.latex(f"\\text{{Sharpe Ratio}} = {sp.latex(sharpe_ratio_with_g)}")


st.subheader('Optimal Hedge Ratio')

st.write(r"""
Daniel $\textit{et al.}$ (2020) showed that we can do better by combining the hedge portfolio $h$ with the characteristic portfolio $c$ in 
order to maximize the Sharpe ratio of the portfolio $p$. Given that the hedge portfolio has zero expected excess return, this is equivalent to finding the 
combination of the characteristic portfolio and the hedge portfolio that minimizes the variance of the resultant portfolio, that is:
         """)


st.latex(r"""
\begin{equation}
            \min_{\delta} \text{var}(r_c - \delta r_h) \implies \delta^* = \rho_{c,h}\frac{\sigma_c}{\sigma_h}
\end{equation}
            """)

st.subheader('Conclusion')