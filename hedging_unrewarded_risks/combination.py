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

st.subheader('Portfolio Combination')

N = 12  # Number of assets

# Predefined fixed values for beta
fixed_beta = [1, 1, 1, 1, 1,1, -1, -1, -1,-1,-1,-1]

st.sidebar.header("Input Desired Correlation Between Beta and Gamma")

# Ask user for the desired correlation coefficient
correlation = st.sidebar.selectbox(
    "Select the correlation between Beta and Gamma", 
    ("0", "1/3", "2/3")
)

# Predefined sets of gamma based on the correlation choices
gamma_sets = {
    "0": [1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1],
    "1/3": [1, 1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1],
    "2/3": [1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1],
}


# Select the gamma set based on the chosen correlation coefficient
selected_gamma = gamma_sets[correlation]

# Convert beta and gamma inputs to Sympy matrices
gamma = sp.Matrix(selected_gamma)
# Convert beta inputs to Sympy matrices
beta = sp.Matrix(fixed_beta)

# Step 1: Compute the means of beta and gamma
beta_mean = sp.Rational(sum(fixed_beta), N)
gamma_mean = sp.Rational(sum(selected_gamma), N)

# Step 2: Compute the covariance between beta and gamma
cov_beta_gamma = sp.Rational(0, 1)
for i in range(N):
    cov_beta_gamma += (beta[i] - beta_mean) * (gamma[i] - gamma_mean)
cov_beta_gamma /= N

# Step 3: Compute the standard deviations of beta and gamma
std_beta = sp.sqrt(sum((beta[i] - beta_mean)**2 for i in range(N)) / N)
std_gamma = sp.sqrt(sum((gamma[i] - gamma_mean)**2 for i in range(N)) / N)

# Step 4: Compute the correlation
correlation = cov_beta_gamma / (std_beta * std_gamma)

# Display the correlation formula
st.write(r"""
The correlation between $\beta$ and $\gamma$ is:
""")
st.latex(r"\rho(\beta, \gamma) = " + sp.latex(correlation.simplify()))



# Portfolio weights based on sorted betas (long the highest, short the lowest)
beta_np = np.array(fixed_beta)
gamma_np = np.array(selected_gamma)

# Get the indices of the sorted beta values
sorted_indices = np.argsort(beta_np)

# Use SymPy's Rational to keep weights as fractions
w_long = sp.Matrix([0] * N)
w_short = sp.Matrix([0] * N)

# Assign long positions (1/3) to the top 3 assets
for idx in sorted_indices[-6:]:
    w_long[idx] = sp.Rational(1, 6)

# Assign short positions (-1/3) to the bottom 3 assets
for idx in sorted_indices[:6]:
    w_short[idx] = sp.Rational(-1, 6)

# Combine long and short positions to form the final weight vector
w_c = w_long + w_short

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
portfolio_return_with_g = w_c.dot(beta * (f + lambda_) + gamma * g + epsilon)


# Step 2: Take the expectation using sympy.stats
expected_portfolio_return_with_g = stats.E(portfolio_return_with_g)

# Contribution from the unpriced factor g:
# LaTeX: Var_g = (w^\top \gamma)^2 \sigma_g^2
variance_g_c = (w_c.dot(gamma))**2 * sigma_g**2  # Contribution from unpriced factor g
# Contribution from the priced factor f:
# LaTeX: Var_f = (w^\top \beta)^2 \sigma_f^2
variance_f_c = (w_c.dot(beta))**2 * sigma_f**2  # Contribution from priced factor f
# Contribution from the idiosyncratic errors:
# LaTeX: Var_\epsilon = w^\top w \times \sigma_\epsilon^2
variance_epsilon = w_c.dot(w_c) * sigma_epsilon**2  # Contribution from idiosyncratic errors
# Total variance of the portfolio:
variance_c = variance_f_c + variance_g_c + variance_epsilon

st.write(r"""
         Our initial portfolio $c$ has the variance:
            """)

st.latex(f"\\sigma^2_c = {sp.latex(variance_c)}")



# Get the top 3 (high beta) and bottom 3 (low beta) indices
high_beta_indices = sorted_indices[-6:]  # Indices for high beta
low_beta_indices = sorted_indices[:6]    # Indices for low beta

low_beta_high_gamma_sorted = low_beta_indices[np.argsort(gamma_np[low_beta_indices])][-3:]

low_beta_low_gamma_sorted = low_beta_indices[np.argsort(gamma_np[low_beta_indices])][:3]

high_beta_high_gamma_sorted = high_beta_indices[np.argsort(gamma_np[high_beta_indices])][-3:]

high_beta_low_gamma_sorted = high_beta_indices[np.argsort(gamma_np[high_beta_indices])][:3]

# Combine the long and short positions
long = np.concatenate([low_beta_high_gamma_sorted, high_beta_high_gamma_sorted])
short = np.concatenate([low_beta_low_gamma_sorted, high_beta_low_gamma_sorted])

# Use SymPy's Rational to keep weights as fractions
w_long = sp.Matrix([0] * N)
w_short = sp.Matrix([0] * N)

# Assign long positions (1/3) to the selected assets
for idx in long:
    w_long[idx] = sp.Rational(1, 6)

# Assign short positions (-1/3) to the selected assets
for idx in short:
    w_short[idx] = sp.Rational(-1, 6)

# Combine long and short positions to form the final weight vector
w_h = w_long + w_short

# Contribution from the unpriced factor g:
# LaTeX: Var_g = (w^\top \gamma)^2 \sigma_g^2
variance_g_h = (w_h.dot(gamma))**2 * sigma_g**2  # Contribution from unpriced factor g
# Contribution from the priced factor f:
# LaTeX: Var_f = (w^\top \beta)^2 \sigma_f^2
variance_f_h = (w_h.dot(beta))**2 * sigma_f**2  # Contribution from priced factor f
# Contribution from the idiosyncratic errors:
# LaTeX: Var_\epsilon = w^\top w \times \sigma_\epsilon^2
variance_epsilon = w_h.dot(w_h) * sigma_epsilon**2  # Contribution from idiosyncratic errors
# Total variance of the portfolio:
variance_h = variance_f_h + variance_g_h + variance_epsilon


st.write(r"""
         and the variance of the hedging portfolio $h$ is:
            """)

st.latex(f"\\sigma^2_h = {sp.latex(variance_h)}")

st.write(r"""
         Therefore, we can form a portfolio $p$ as a combination of the initial portfolio $c$ and the hedging portfolio $h$:
            """)

# Sidebar: Adjust the weight of the hedging portfolio
hedge_weight = st.sidebar.slider(
    "Adjust the weight of the hedging portfolio",
    min_value=0.0,
    max_value=2.0,
    step=0.25
)

# Combine the initial portfolio with the hedging portfolio
w_p = w_c - sp.Rational(hedge_weight) * w_h

st.latex(f"w_p = w_c - {sp.Rational(hedge_weight)} w_h")


# Contribution from the unpriced factor g:
# LaTeX: Var_g = (w^\top \gamma)^2 \sigma_g^2
variance_g_p = (w_p.dot(gamma))**2 * sigma_g**2  # Contribution from unpriced factor g
# Contribution from the priced factor f:
# LaTeX: Var_f = (w^\top \beta)^2 \sigma_f^2
variance_f_p = (w_p.dot(beta))**2 * sigma_f**2  # Contribution from priced factor f
# Contribution from the idiosyncratic errors:
# LaTeX: Var_\epsilon = w^\top w \times \sigma_\epsilon^2
variance_epsilon = w_p.dot(w_p) * sigma_epsilon**2  # Contribution from idiosyncratic errors
# Total variance of the portfolio:
variance_p = variance_f_p + variance_g_p + variance_epsilon

st.write(r"""
         Which has the variance:
         """)

st.latex(f"\\sigma^2_p = {sp.latex(variance_p)}")

st.write(r"""
Play around with the slider to find a combination of the initial portfolio $c$ and the hedging portfolio $h$ that minimizes the variance of the portfolio $p$.
         """)


st.subheader('Optimal Hedge Ratio')

st.write(r"""
Daniel $\textit{et al.}$ (2020) showed that we can do better by combining the hedge portfolio $h$ with the characteristic portfolio $c$ in 
order to maximize the Sharpe ratio of the portfolio $p$. Given that the hedge portfolio has zero expected excess return, this is equivalent to finding the 
combination of the characteristic portfolio and the hedge portfolio that minimizes the variance of the resultant portfolio, that is:
         """)



#use correlation coefficient as the optimal hedge ratio
optimal_hedge_ratio = variance_g_c / variance_g_h

st.latex(r"""
\begin{equation}
\min_{\delta} \text{var}(r_c - \delta r_h) \implies \delta^* = \rho_{c,h}\frac{\sigma_c}{\sigma_h}
\end{equation}
""")
# st.write(fr"""The calculated value of the optimal hedge ratio is: $\delta^* = {sp.latex(optimal_hedge_ratio.simplify())}$.""")

# # Variance of the combined portfolio
# variance_p_optim = variance_c - optimal_hedge_ratio * variance_h

# st.write(r"""
#          The variance of the combined portfolio $p$ is:
#          """)
# st.latex(f"\\sigma^2_p = {sp.latex(variance_p_optim)}")

# w_p_opt = w_c - sp.Rational(optimal_hedge_ratio) * w_h


# st.latex(w_c.dot(gamma))
# st.latex(w_h.dot(gamma))
# st.write(fr"""$\gamma_p^* = {sp.latex(w_p_opt.dot(gamma))}$""")

# st.subheader('Conclusion')