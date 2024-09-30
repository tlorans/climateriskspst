import streamlit as st
import numpy as np
import plotly.graph_objs as go
import numpy as np
import sympy as sp
import sympy.stats as stats
import plotly.graph_objs as go


st.title('Rewarded and Unrewarded Risk in Finance')


st.write(r"""
Asset pricing theory suggest that one of the main challenge 
         in finance is the efficient diversification of unrewarded risks, 
         where "diversification" means "reduction" or "cancellation" (as in "diversify away")
            and "unrewarded" means "not compensated by a risk premium".
         Indeed, unrewarded risks are by definition not attractive for investors 
         who are inherently risk-averse and therefore only willing to take 
         risks if there is an associated reward to be expected in exchange for such 
         risk-taking, as shown by Markowitz (1952). (Amenc $\textit {et al.}$, 2014)
         """)

st.subheader('Rewarded Risk')

st.write(r"""
Following Daniel $\textit {et al.}$ (2020), 
we consider a one-period economy with $N$ assets. 
Realized excess returns are given by:
""")

st.latex(r"""
        \begin{equation}
r_i = \beta_i (f + \lambda) + \epsilon_i
\end{equation}
         """)

st.write(r"""
with $r_i$ the excess return of asset $i$, 
$\beta_i$ the factor loading on the rewarded factor $f$,
$\lambda$ the risk premium on $f$ and $\epsilon_i$ the idiosyncratic return. 
We have 
$\mathbb{E}[\epsilon_i] = \mathbb{E}[f] = 0$.

We have $r$ the $N \times 1$ vector of excess returns.
Taking expectations of our first equation, we have:
         """)

st.latex(r"""
         \begin{equation}
    \begin{aligned}
    \mu = \mathbb{E}[r] = \mathbb{E}[\beta (f + \lambda) + \epsilon] \\
    = \beta \mathbb{E}[f + \lambda] + \mathbb{E}[\epsilon] \\
    = \beta \lambda
    \end{aligned}
\end{equation}
         """)

st.write(r"""
         with $\beta$ the $N \times 1$ vector of factor loadings on $f$.
So, expected returns are driven by the exposure to the rewarded factor $f$ 
and the risk premium $\lambda$ on $f$.
Following Daniel $\textit {et al.}$ (2020), we now assume that expected returns 
are described by a single characteristic $x$:
                """)

st.latex(r"""
         \begin{equation}
    \mu = x \lambda_c
\end{equation}
                """)

st.write(r"""
         
where $\lambda_c$ is the risk premium on the characteristic $x$.

To have the two last equations to hold, we must have:
""")

st.latex(r"""
         \begin{equation}
    \begin{aligned}
    \beta \lambda = x \lambda_c \\
    \beta = x \frac{\lambda_c}{\lambda}
    \end{aligned}
\end{equation}
                """)

st.write(r"""
         This equation shows that the characteristic is a perfect 
proxy for the exposure to the rewarded factor. 

We now consider only 6 assets with equal market capitalization.
The assets have characteristics and loadings on the rewarded factor as:
""")

N = 6  # Number of assets

# Predefined values for beta and gamma
default_beta = [1, 1, 1, -1, -1, -1]

st.sidebar.header("Input Loadings for Each Asset")


# Collect beta inputs in the sidebar
beta_input = []
for i in range(N):
    beta_val = st.sidebar.number_input(f'Beta for Asset {i+1}', min_value=-1, max_value=1, value=default_beta[i], step=2, key=f'beta_{i}')
    beta_input.append(beta_val)


# Create a LaTeX table for the beta inputs
beta_latex = r"\begin{array}{|c|c|} \hline Asset & \beta \\ \hline "
for i in range(N):
    beta_latex += f"{i+1} & {beta_input[i]} \\\\ \\hline "
beta_latex += r"\end{array}"
st.latex(beta_latex)



# Convert beta inputs to Sympy matrices
beta = sp.Matrix(beta_input)

# Portfolio weights based on sorted betas (long the highest, short the lowest)
beta_np = np.array(beta_input)

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
w = w_long + w_short

# Display the resulting portfolio in LaTeX format
st.write(r"""
        We can construct a portfolio $c$ on the basis of characteristic 
$x$ (or $\beta$, as we have supposed that $x$ is a perfect proxy for $\beta$).
The weights are given by:
         """)
# Prepare weights in LaTeX format as a row vector
weights_latex = r"\begin{bmatrix} "
for i in range(6):
    weights_latex += f"{sp.latex(w[i])} & "
weights_latex = weights_latex[:-2] + r" \end{bmatrix}"  # Remove the last "&" and close the matrix


st.latex(r"""
w_c^T = """ + weights_latex)



# Define priced factor as normal random variable with variance properties
f = stats.Normal('f', 0, sp.symbols('sigma_f'))  # Priced factor f with E[f] = 0 and var(f) = sigma_f^2

# Define idiosyncratic errors epsilon_i as random variables with zero mean and variance sigma_epsilon^2
epsilon = sp.Matrix([stats.Normal(f'epsilon', 0, sp.symbols('sigma_epsilon')) for i in range(N)])
# Define symbols for variances of the factors and idiosyncratic error
sigma_f, sigma_epsilon = sp.symbols('sigma_f sigma_epsilon')

# Characteristic premium
lambda_ = sp.symbols('lambda')


# Step 1: Define the portfolio return formula symbolically
portfolio_return = w.dot(beta * (f + lambda_) + epsilon)

st.write(r"""
         We can now compute the return of the portfolio $c$:
                """)

st.latex(f"""= {sp.latex(portfolio_return)}""")

# Step 2: Take the expectation using sympy.stats
expected_portfolio_return = stats.E(portfolio_return)

# Contribution from the priced factor f:
# LaTeX: Var_f = (w^\top \beta)^2 \sigma_f^2
variance_f = (w.dot(beta))**2 * sigma_f**2  # Contribution from priced factor f


# Contribution from the idiosyncratic errors:
# LaTeX: Var_\epsilon = w^\top w \times \sigma_\epsilon^2
variance_epsilon = w.dot(w) * sigma_epsilon**2  # Contribution from idiosyncratic errors

# Total variance of the portfolio:
total_portfolio_variance = variance_f + variance_epsilon

# Calculate the Sharpe ratio
sharpe_ratio = expected_portfolio_return / sp.sqrt(total_portfolio_variance)

beta_p = beta.dot(w)

st.write(r"""
         
The portfolio returns capture the expected returns 
because it loads on the rewarded factor $f$.
The expected return of the portfolio is:
""")

st.latex(f"E[r_c] = {sp.latex(expected_portfolio_return)}")


st.write(r"""
         The variance of the portfolio is:
                """)

st.latex(f"\\sigma^2_c = {sp.latex(total_portfolio_variance)}")

st.write(r"""
         which give us the Sharpe ratio of the portfolio:
                """)

st.latex(f"\\text{{Sharpe Ratio}} = {sp.latex(sharpe_ratio)}")

st.write(r"""
         The portfolio $c$ is efficient because it loads on the rewarded factor $f$,
         while the idiosyncratic risk is diversified away."""
         )


st.subheader('Unrewarded Risk')

st.write(r"""
We now consider the unrewarded risk. We add a second factor $g$ to the model:
""")

st.latex(r"""
        \begin{equation}
r_i = \beta_i (f + \lambda) + \gamma_i g + \epsilon_i
\end{equation}
         """)

st.write(r"""
with $\gamma_i$ the factor loading on the unrewarded factor $g$,
We have $\mathbb{E}[\epsilon_i] = \mathbb{E}[f] = \mathbb{E}[g] = 0$.
Note that because it is unrewarded, there is no risk premium $\lambda_g$ on $g$.

Taking expectations of our first equation, we now have:
         """)

st.latex(r"""
         \begin{equation}
    \begin{aligned}
    \mu = \mathbb{E}[r] = \mathbb{E}[\beta (f + \lambda) + \gamma g + \epsilon] \\
    = \beta \mathbb{E}[f + \lambda] + \gamma \mathbb{E}[g] + \mathbb{E}[\epsilon] \\
    = \beta \lambda
    \end{aligned}
\end{equation}
         """)

st.write(r"""
Because $g$ is an unrewarded factor (ie. $\lambda_g = 0$),
the expected returns are still driven by the exposure to the rewarded factor $f$ only, 
and the risk premium $\lambda$ on $f$.
                """)

st.write(r"""
         We go back to our 6 assets with equal market capitalization example.
         In addition to exposure to the rewarded factor $f$,
            the assets have loadings $\gamma$ on the unrewarded factors $g$:
""")

default_gamma = [1, 1, -1, 1, -1, -1]

# Collect beta inputs in the sidebar
gamma_input = []
for i in range(N):
    gamma_val = st.sidebar.number_input(f'Gamma for Asset {i+1}', min_value=-1, max_value=1, value=default_gamma[i], step=2, key=f'gamma_{i}')
    gamma_input.append(gamma_val)



# Create a LaTeX table for the beta and gamma inputs
table_latex = r"\begin{array}{|c|c|c|} \hline Asset & \beta & \gamma \\ \hline "
for i in range(N):
    table_latex += f"{i+1} & {beta_input[i]} & {gamma_input[i]} \\\\ \\hline "
table_latex += r"\end{array}"
st.latex(table_latex)



# Convert beta and gamma inputs to Sympy matrices
gamma = sp.Matrix(gamma_input)



# Define priced and unpriced factors as normal random variables with variance properties
g = stats.Normal('g', 0, sp.symbols('sigma_g'))  # Unpriced factor g with E[g] = 0 and var(g) = sigma_g^2

# Define symbols for variances of the factors and idiosyncratic error
sigma_g = sp.symbols('sigma_g')

# Step 1: Define the portfolio return formula symbolically
portfolio_return_with_g = w.dot(beta * (f + lambda_) + gamma * g + epsilon)

st.write(r"""
         We can now compute the return of the portfolio $c$:
                """)

st.latex(f"""r_c = {sp.latex(portfolio_return_with_g)}""")

# Step 2: Take the expectation using sympy.stats
expected_portfolio_return_with_g = stats.E(portfolio_return_with_g)

# Contribution from the unpriced factor g:
# LaTeX: Var_g = (w^\top \gamma)^2 \sigma_g^2
variance_g = (w.dot(gamma))**2 * sigma_g**2  # Contribution from unpriced factor g

# Total variance of the portfolio:
total_portfolio_variance_with_g = variance_f + variance_g + variance_epsilon

# Calculate the Sharpe ratio
sharpe_ratio_with_g = expected_portfolio_return_with_g / sp.sqrt(total_portfolio_variance_with_g)

gamma_p = gamma.dot(w)

st.write(r"""
         
The portfolio returns captures the expected returns 
because it loads on the rewarded factor $f$, but it also loads 
on the unrewarded factor $g$.
Indeed, in our example there exists a cross-sectional correlation
between the characteristics and the loadings on the unrewarded factor.
Most assets with positive loadings on the rewarded factor have positive loadings on the unrewarded factor.
The expected return of the portfolio is:
""")

st.latex(f"E[r_c] = {sp.latex(expected_portfolio_return_with_g)}")


st.write(r"""
         The variance of the portfolio is:
                """)

st.latex(f"\\sigma^2_c = {sp.latex(total_portfolio_variance_with_g)}")

st.write(r"""
         which give us the Sharpe ratio of the portfolio:
                """)

st.latex(f"\\text{{Sharpe Ratio}} = {sp.latex(sharpe_ratio_with_g)}")

st.write(r"""
         The portfolio $c$ is not efficient because it loads on the unrewarded factor $g$.
Loading on the unrewarded factor $g$ is a source of risk (additional variance in the denominator of the Sharpe ratio)
that is not rewarded by the market (no risk premium $\lambda_g$ on $g$, and therefore no supplementary expected return in the numerator of the Sharpe ratio).
This simple example give us the broad idea about unrewarded risk exposure,
and how they undermine the efficiency of a portfolio."""
         )

# Step 1: Compute the means of beta and gamma
beta_mean = sp.Rational(sum(beta_input), N)
gamma_mean = sp.Rational(sum(gamma_input), N)

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
The symbolic correlation between $\beta$ and $\gamma$ is:
""")
st.latex(r"\rho(\beta, \gamma) = " + sp.latex(correlation.simplify()))
