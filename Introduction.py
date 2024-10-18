import streamlit as st
import sympy as sp

st.set_page_config(page_title="Climate Risks")

st.title('Climate Risks in Portfolio Construction')

st.subheader('Mean-Variance Efficient Portfolio with Multi-factor Model')

st.write(r"""We follow Rjaily $\textit{et al.}$ (2024). In an investment universe with $N$ assets, the investor assumes 
         that asset returns follow a multi-factor risk model:""")

st.latex(r"""
\begin{equation}
         R = B F + \epsilon
\end{equation}
         """)

st.write(r"""where $R$ is the $N \times 1$ vector of asset excess returns,
            $F$ is the $K \times 1$ vector of factor excess returns,
            $B$ is the $N \times K$ matrix of factor loadings, and
            $\epsilon$ is the $N \times 1$ vector of idiosyncratic returns.""")

st.write(r"""
We assume that $E(F) = \lambda$, $\text{Cov}(F) = \Omega$, $E(\epsilon) = 0$, $\text{Cov}(\epsilon) = D$,
and $\text{Cov}(F, \epsilon) = 0$.
         """)

st.write(r"""The vector of expected returns is given by:""")

st.latex(r"""
\begin{equation}
         \mu = B \lambda
\end{equation}
         """)

st.write(r"""While the covariance matrix of asset returns is given by:""")

st.latex(r"""
\begin{equation}
         \Sigma = B \Omega B^\top + D
\end{equation}
         """)


st.write(r"""Given a portfolio $x$ with a vector of weight $w$, it's expected returns is given by:""")

st.latex(r"""
\begin{equation}
         \mu_x = w^\top \mu = w^\top B \lambda
\end{equation}
         """)

st.write(r"""The variance of the portfolio is given by:""")

st.latex(r"""
\begin{equation}
         \sigma_x^2 = w^\top \Sigma w = w^\top(B \Omega B^\top w + D) w
\end{equation}
         """)

st.write(r"""The vector of beta coefficients is given by:""")

st.latex(r"""
\begin{equation}
         \beta_x = B^\top w
\end{equation}
         """)

st.write(r"""The Sharpe ratio of the portfolio is given by:""")

st.latex(r"""
\begin{equation}
         \text{SR}_x = \frac{\mu_x}{\sqrt{\sigma_x^2}}
\end{equation}
         """)

st.write(r"""We can assess the importance of common factors by calculating the propotyion 
         of the portfolio variance that is explained by the factors.""")

st.latex(r"""
         \begin{equation}
         \text{R}_c^{2} = \frac{w^\top B \Omega B^\top w}{w^\top(B \Omega B^\top w + D) w}
         \end{equation}
         """)

st.write(r"""We can uncover the individual contribution of each factor to the portfolio variance:""")

st.latex(r"""
            \begin{equation}
         \text{R}_j^{2} = \frac{w^\top B_j B_j^\top \sigma_j^2 w}{w^\top(B \Omega B^\top w + D) w}
            \end{equation}
            """)

st.write(r"""where $B_j$ is the $N \times 1$ vector of factor loadings on factor $j$,
            and $\sigma_j^2$ is the variance of factor $j$.""")

st.write(r"""
         The investor seeks to maximize the Sharpe ratio of the portfolio by choosing the optimal weights $w$.
         """)

st.write(r"""We consider the following optimization problem:""")

st.latex(r'''
         \begin{equation}
         \begin{aligned}
w^*(\gamma) = \underset{w}{\text{arg min}} \left( \frac{1}{2} w^\top \Sigma w - \gamma w^\top \mu \right) \\
\text{ subject to } \mathbf{1}_n^\top w = 1
\end{aligned}
\end{equation}
''')

st.write("where $\gamma$ is the risk tolerance parameter. We have the following solution:")

st.latex(r'''
w^* = \gamma \Sigma^{-1} \mu
''')

st.write(r"""where $\gamma = (1_n^\top \Sigma^{-1}\mu)^{-1}$.""")


# Define symbols for SymPy
n = 5  # Number of assets
k = 2  # Number of factors

B = sp.Matrix([[0.5, -0.3], [1.2, 0.6], [1.4, -0.3], [0.7, 0.3], [0.8, -0.9]])
Omega = sp.diag(0.20**2, 0.05**2)
D = sp.diag(0.12**2, 0.10**2, 0.08**2, 0.10**2, 0.12**2)
lambda_factors = sp.Matrix([0.04, 0.01])
r_f = 0.00  # Risk-free rate

# Compute expected returns of assets
mu = B * lambda_factors

# Compute covariance matrix of assets
Sigma = B * Omega * B.T + D

# Compute the optimal portfolio weights (tangency portfolio)
inv_Sigma = Sigma.inv()
gamma = sp.Symbol('gamma')  # Risk tolerance parameter
w_star = gamma * inv_Sigma * mu

# Display numerical example
st.subheader("Climate Risks as a New Risk Factor")

st.write(r"""
         We consider a simple example with two uncorrelated factors $F_1$ and $F_2$, with the following expected returns:
         """)

default_F_1_risk_premia = 0.04
default_F_2_risk_premia = 0.01

F1_risk_premia_val = st.sidebar.number_input('$E(F_1)$',min_value=0., max_value=0.1, value=default_F_1_risk_premia, step=0.01)
F2_risk_premia_val = st.sidebar.number_input('$E(F_2)$',min_value=0., max_value=0.1, value=default_F_2_risk_premia, step=0.01)


selected_risk_premia = [F1_risk_premia_val, F2_risk_premia_val]
lambda_factors = sp.Matrix(selected_risk_premia)

# Convert the SymPy matrix to a LaTeX string
lambda_latex = sp.latex(lambda_factors)

# Display the equation for risk premia
st.latex(rf"\lambda = \begin{{bmatrix}} {F1_risk_premia_val:.2f} \\ {F2_risk_premia_val:.2f} \end{{bmatrix}}")

st.write("And the following covariance matrix for the factors:")

Omega = sp.diag(0.20**2, 0.05**2)

# Convert the SymPy matrix to a LaTeX string
Omega_latex = sp.latex(Omega)

# Display the equation for the covariance matrix of factors
st.latex(rf"\Omega = {Omega_latex}")

st.write("We have 5 assets with the following factor loadings:")

B = sp.Matrix([[0.5, -0.3], [1.2, 0.6], [1.4, -0.3], [0.7, 0.3], [0.8, -0.9]])

# Convert the SymPy matrix to a LaTeX string
B_latex = sp.latex(B)

# Display the equation for factor loadings
st.latex(rf"""B = {B_latex}""")

st.write("And the following diagnoal matrix $D$ for the idiosyncratic risks:")

D = sp.diag(0.12**2, 0.10**2, 0.08**2, 0.10**2, 0.12**2)

# Convert the SymPy matrix to a LaTeX string
D_latex = sp.latex(D)

# Display the equation for the idiosyncratic risks
st.latex(rf"""D = {D_latex}""")

st.write("Therefore the vector of expected returns is given by:")

mu = B * lambda_factors

# Convert the SymPy matrix to a LaTeX string
mu_latex = sp.latex(mu)

# Display the equation for expected returns
st.latex(rf"\mu = {B_latex} {lambda_latex} = {mu_latex}")

st.write("And the covariance matrix of asset returns is given by:")

Sigma = B * Omega * B.T + D

# Convert the SymPy matrix to a LaTeX string
Sigma_latex = sp.latex(Sigma)

# Display the equation for the covariance matrix of asset returns
st.latex(rf"\Sigma = {B_latex} {Omega_latex} {B_latex}^\top + {D_latex}")
st.latex(rf"\Sigma = {Sigma_latex}")

st.write("We have therefore all what we need to get the tangent portfolio weights:")

gamma_val = 1

numerator = Sigma.inv()*mu
denominator = sp.Matrix([1]*n).T * Sigma.inv() * mu

w_star = numerator / denominator[0]

# Convert the SymPy matrix to a LaTeX string
w_star_latex = sp.latex(w_star)

# Display the equation for the optimal portfolio weights
st.latex(rf"""w^* = \gamma \Sigma^{{-1}} \mu = {w_star_latex}""")

portfolio_expected_return = w_star.T * mu   # Portfolio expected return
portfolio_variance = w_star.T * Sigma * w_star  # Portfolio variance

# Extract scalar values from the 1x1 matrices
portfolio_expected_return_scalar = round(portfolio_expected_return[0] * 100,2)
portfolio_variance_scalar = round(portfolio_variance[0] * 100,2)

# Compute Sharpe ratio
portfolio_sharpe_ratio = portfolio_expected_return_scalar / sp.sqrt(portfolio_variance_scalar)
# Compute portfolio betas and R-squared as before
portfolio_betas = B.T * w_star

# Compute the overall R-squared of the portfolio
numerator_r_squared_c = (w_star.T * B * Omega * B.T * w_star)[0]  # Extract scalar
denominator_r_squared_c = (w_star.T * Sigma * w_star)[0]  # Extract scalar

# Compute R-squared (overall)
r_squared_c = numerator_r_squared_c / denominator_r_squared_c * 100

# Compute R-squared for each factor
r_squared_factors = []
for j in range(B.shape[1]):  # Loop through each factor
    numerator_r_squared_j = (w_star.T * B[:, j] * B[:, j].T * Omega[j, j] * w_star)[0]  # Extract scalar
    denominator_r_squared_j = denominator_r_squared_c  # The denominator is the same
    r_squared_j = numerator_r_squared_j / denominator_r_squared_j
    r_squared_factors.append(r_squared_j)

# Convert values to latex-friendly strings
r_squared_c_latex = sp.latex(round(r_squared_c,2))
r_squared_factors_latex = [sp.latex(round(r2 * 100,2)) for r2 in r_squared_factors]

# Display the results in a LaTeX table

table_latex = r"\begin{array}{|c|c|} \hline \text{Metric} & \text{Value} \\ \hline"
table_latex += rf"\text{{Expected Return}} & {sp.latex(portfolio_expected_return_scalar)} \\"
table_latex += rf"\text{{Volatility}} & {sp.latex(sp.sqrt(portfolio_variance_scalar))} \\"
table_latex += rf"\text{{Sharpe Ratio}} & {sp.latex(portfolio_sharpe_ratio)} \\"
table_latex += rf"\beta_1 & {sp.latex(round(portfolio_betas[0], 2))} \\"
table_latex += rf"\beta_2 & {sp.latex(round(portfolio_betas[1], 2))} \\"
table_latex += rf"\text{{R}}^2_c & {r_squared_c_latex} \\"
table_latex += rf"\text{{R}}^2_1 & {r_squared_factors_latex[0]} \\"
table_latex += rf"\text{{R}}^2_2 & {r_squared_factors_latex[1]} \\"
table_latex += r"\hline"
table_latex += r"\end{array}"

st.latex(table_latex)

st.subheader("Climate Risks as an Unrewarded Risk Factor")


st.subheader("The Problem")