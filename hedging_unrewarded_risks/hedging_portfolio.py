import streamlit as st
import numpy as np
import plotly.graph_objs as go
import numpy as np
import sympy as sp
import sympy.stats as stats
import plotly.graph_objs as go

st.title('Hedging from Unrewarded Risks')

st.write(r"""
We have seen that, as of today, no risk premium seems to be associated with climate risks. It is therefore to be treated as unrewarded risk. 
We have seen that if exposure to the rewarded risk is correlated with exposure to unrewarded risk, a portfolio rightly exposed to rewarded risk is mean-variance inefficient because
         it loads in the unrewarded risk. We can improve upon this. We want a method that (1) let untouched the rewarded risk exposure and (2) eliminate the unrewarded risk exposure.
         """)

st.subheader('Characteristic-Balanced Hedge Portfolio')

st.write(r"""
         We follow Daniel $\textit{et al.}$ (2020) and show 
how we can improve the portfolio $c$ from the first section.
We can form a hedge portfolio $h$ that is long in assets with high exposure to the unrewarded factor $g$ and short in assets with low exposure to $g$. 
But we need to ensure that the hedge portfolio, that will be used to enhance the initial characteristic portfolio $c$, will not alter the exposure to the rewarded factor $f$.
         
To do this, we can long assets with high exposure to the unrewarded factor $g$ and short assets with low exposure to $g$, controlled for the exposure to the rewarded factor $f$.
Let's consider 6 assets with the following loadings on the rewarded and unrewarded factors:
""")


default_gamma = [1, 1, -1, 1, -1, -1]
N = 6  # Number of assets

# Predefined values for beta and gamma
default_beta = [1, 1, 1, -1, -1, -1]

st.sidebar.header("Input Loadings for Each Asset")


# Collect beta inputs in the sidebar
beta_input = []
for i in range(N):
    beta_val = st.sidebar.number_input(f'Beta for Asset {i+1}', min_value=-1, max_value=1, value=default_beta[i], step=2, key=f'beta_{i}')
    beta_input.append(beta_val)


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
# Convert beta inputs to Sympy matrices
beta = sp.Matrix(beta_input)
# Convert beta and gamma inputs to NumPy arrays
beta_np = np.array(beta_input)
gamma_np = np.array(gamma_input)
# Get the indices of the sorted beta values
sorted_beta_indices = np.argsort(beta_np)

# Long the highest beta assets and short the lowest beta assets
long_beta_indices = sorted_beta_indices[-3:]
short_beta_indices = sorted_beta_indices[:3]

# Sort by gamma within the long beta sleeve and short beta sleeve separately
long_gamma_sorted_indices = long_beta_indices[np.argsort(gamma_np[long_beta_indices])]
short_gamma_sorted_indices = short_beta_indices[np.argsort(gamma_np[short_beta_indices])]

# Select high gamma for long and low gamma for short within each sleeve
high_gamma_long = long_gamma_sorted_indices[-2:]  # Highest gamma in long sleeve
low_gamma_short = short_gamma_sorted_indices[:2]  # Lowest gamma in short sleeve

# Assign rational equal weights to the high gamma and low gamma assets
w = sp.Matrix([0] * N)
for idx in high_gamma_long:
    w[idx] = sp.Rational(1, 2)  # Equal weight for the long stocks with high gamma

for idx in low_gamma_short:
    w[idx] = sp.Rational(-1, 2)  # Equal weight for the short stocks with low gamma


st.write(r"""
         The weights of the portfolio are:
         """)

# Prepare weights in LaTeX format as a row vector
weights_latex = r"\begin{bmatrix} "
for i in range(6):
    weights_latex += f"{w[i]} & "
weights_latex = weights_latex[:-2] + r" \end{bmatrix}"  # Remove the last "&" and close the matrix

st.latex(r"""
w_h^T = """ + weights_latex)

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
         We can now compute the return of the portfolio $c$:
                """)

st.latex(f"""r_h = {sp.latex(portfolio_return_with_g)}""")

# Step 2: Take the expectation using sympy.stats
expected_portfolio_return_with_g = stats.E(portfolio_return_with_g)


st.write(r"""
This portfolio goes long assets with high exposure to the 
unrewarded factor $g$ and short assets with low exposure to $g$.
The return of the portfolio $h$ is given by:
         """)




st.write(r"""

The loading of the portfolio $h$ on the rewarded factor $f$ is zero,
and the loading on the unrewarded factor $g$ is 2. Therefore, 
expected return of the portfolio $h$ is :
         """)


# Step 2: Take the expectation using sympy.stats
expected_portfolio_return_with_g = stats.E(portfolio_return_with_g) 

st.latex(f"E[r_h] = {sp.latex(expected_portfolio_return_with_g)}")

st.subheader('Hedge Portfolio')