import streamlit as st
import sympy as sp
import numpy as np

st.set_page_config(page_title="Climate Risks")

st.title('Multi-factor model and Portfolio Implied Risk Premia')

st.subheader('Multi-factor Model')

st.write(r"""In an investment universe with $N$ assets, we assume the assets
         realized excess returns to follow a multi-factor risk model with $K$ factors:""")

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
We assume that $E(F) = \psi$, $\text{Cov}(F) = \Omega$, $E(\epsilon) = 0$, $\text{Cov}(\epsilon) = D$,
and $\text{Cov}(F, \epsilon) = 0$, $E(\epsilon) = 0$.
         """)

st.write(r"""The vector of expected returns is given by:""")

st.latex(r"""
\begin{equation}
         \pi = B \lambda
\end{equation}
         """)


st.write(r"""The covariance matrix of asset returns is given by:""")

st.latex(r"""
\begin{equation}
         \Sigma = B \Omega B^\top + D
\end{equation}
         """)

st.write(r"""Given a portfolio $x$ with a vector of weight $w$, it's risk premium is given by:""")

st.latex(r"""
\begin{equation}
         \pi_x = w^\top \pi = w^\top B \psi
\end{equation}
         """)

st.write(r"""The volatility of the portfolio is given by:""")

st.latex(r"""
\begin{equation}
         \sigma_x = \sqrt{w^\top \Sigma w} = \sqrt{w^\top(B \Omega B^\top w + D) w}
\end{equation}
         """)

st.write(r"""The vector of beta coefficients is given by:""")

st.latex(r"""
\begin{equation}
         \beta_x = B^\top w
\end{equation}
         """)

st.write(r"""We can assess the importance of common factors by calculating the proportion 
         of the portfolio variance that is explained by the factors:""")

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
         We have $\mu$ and $\Sigma$ any vector of expected excess returns and covariance matrix. We consider the following optimization problem:
         """)

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



st.write(r"""Let's consider an investment universe with two factors. The loading matrix $B$ is the identity matrix:""")

# Define symbols for SymPy
n = 2  # Number of assets
k = 2  # Number of factors

# Define the loading matrix B as the identity matrix
B = sp.eye(n)

# Convert the SymPy matrix to a LaTeX string
B_latex = sp.latex(B)

# Display the equation for factor loadings
st.latex(rf"""B = {B_latex}""")


st.write(r"""The two factor are uncorrelated and their volatilities are $20\%$ and $5\%$ respectively. The covariance matrix of the factors is given by:""")

Omega = sp.diag(0.20**2, 0.05**2)

# Convert the SymPy matrix to a LaTeX string
Omega_latex = sp.latex(Omega)

# Display the equation for the covariance matrix of factors
st.latex(rf"\Omega = {Omega_latex}")


st.write(r"""The expected returns of the factors are $4\%$ and $1\%$ respectively:""")

psi = sp.Matrix([0.04, 0.01])

# Convert the SymPy matrix to a LaTeX string
psi_latex = sp.latex(psi)

# Display the equation for the expected returns of the factors
st.latex(rf"\psi = {psi_latex}")

st.write(r"""The vector of expected returns of the assets is given by:""")

pi = B * psi

# Convert the SymPy matrix to a LaTeX string
pi_latex = sp.latex(pi)

# Display the equation for expected returns
st.latex(rf"\pi = {B_latex} {psi_latex} = {pi_latex}")

st.write(r"""The covariance matrix of asset returns is given by:""")

# Compute covariance matrix of assets
Sigma = B * Omega * B.T 

# Convert the SymPy matrix to a LaTeX string
Sigma_latex = sp.latex(Sigma)

# Display the equation for the covariance matrix of asset returns
st.latex(rf"\Sigma = {B_latex} {Omega_latex} {B_latex}^\top")
st.latex(rf"\Sigma = {Sigma_latex}")

st.write(r"""We have therefore all what we need to get the tangent portfolio weights:""")

numerator = Sigma.inv()*pi
denominator = sp.Matrix([1]*n).T * Sigma.inv() * pi

w_star = numerator / denominator[0]


# Convert the SymPy matrix to a LaTeX string
w_star_latex = sp.latex(w_star)

# Display the equation for the optimal portfolio weights
st.latex(rf"""w^* = \gamma \Sigma^{{-1}} \mu = {w_star_latex}""")



st.write(r"""The portfolio risk premia is given by:""")
portfolio_expected_return = w_star.T * pi   # Portfolio expected return

# convert the SymPy matrix to a LaTeX string
portfolio_expected_return_latex = sp.latex(portfolio_expected_return[0])

# Display the equation for the expected return of the portfolio
st.latex(rf"\pi_x = w^\top \pi = {w_star_latex} {pi_latex} = {portfolio_expected_return_latex}")

portfolio_variance = w_star.T * Sigma * w_star  # Portfolio variance

# Extract scalar values from the 1x1 matrices
portfolio_expected_return_scalar = round(portfolio_expected_return[0] * 100,2)
portfolio_variance_scalar = round(portfolio_variance[0] * 100,2)

st.write(r"""The volatility of the portfolio is given by:""")

# Convert the SymPy matrix to a LaTeX string
portfolio_vol_latex = sp.latex(sp.sqrt(portfolio_variance[0]))

# Display the equation for the variance of the portfolio
st.latex(rf"\sigma_x = \sqrt{{w^\top \Sigma w}} = \sqrt{{w^\top(B \Omega B^\top w) w}} = {portfolio_vol_latex}")

st.write(r"""and the Sharpe ratio is given by:""")

# Compute Sharpe ratio
portfolio_sharpe_ratio = portfolio_expected_return_scalar / sp.sqrt(portfolio_variance_scalar)

# Convert the SymPy matrix to a LaTeX string
portfolio_sharpe_ratio_latex = sp.latex(portfolio_sharpe_ratio)

# Display the equation for the Sharpe ratio
st.latex(rf"\text{{Sharpe Ratio}} = \frac{{\pi_x}}{{\sigma_x}} = {portfolio_sharpe_ratio_latex}")


# Compute portfolio betas and R-squared as before
portfolio_betas = B.T * w_star

st.write(r"""The vector of beta coefficients is given by:""")

# Convert the SymPy matrix to a LaTeX string
portfolio_betas_latex = sp.latex(portfolio_betas)

# Display the equation for the beta coefficients
st.latex(rf"\beta_x = B^\top w = {portfolio_betas_latex}")



# # Compute the overall R-squared of the portfolio
numerator_r_squared_c = (w_star.T * B * Omega * B.T * w_star)[0]  # Extract scalar
denominator_r_squared_c = (w_star.T * Sigma * w_star)[0]  # Extract scalar
# Compute R-squared (overall)
r_squared_c = numerator_r_squared_c / denominator_r_squared_c * 100

# Compute R-squared for each factor
r_squared_factors = []
for j in range(B.shape[1]):  # Loop through each factor
    B_j = sp.zeros(*B.shape)
    
    # Keep only the j-th column of B (set other columns to zero)
    B_j[:, j] = B[:, j]

    # Use the modified B_j matrix to calculate the numerator for the specific factor's R^2
    numerator_r_squared_j = (w_star.T * (B_j * Omega * B_j.T) * w_star)[0]  # Factor's own contribution + covariances
    denominator_r_squared_j = denominator_r_squared_c  # The denominator is the same
    r_squared_j = numerator_r_squared_j / denominator_r_squared_j
    r_squared_factors.append(r_squared_j)

# Convert values to latex-friendly strings
r_squared_c_latex = sp.latex(round(r_squared_c,2))
r_squared_factors_latex = [sp.latex(round(r2 * 100,2)) for r2 in r_squared_factors]

# Display the results in a LaTeX table

table_latex = r"\begin{array}{|c|c|} \hline \text{Metric} & \text{Value} \\ \hline"
table_latex += rf"\pi_x & {sp.latex(portfolio_expected_return_scalar)} \\"
table_latex += rf"\sigma_x & {sp.latex(sp.sqrt(portfolio_variance_scalar))} \\"
table_latex += rf"SR_x & {sp.latex(portfolio_sharpe_ratio)} \\"
table_latex += rf"\beta_1 & {sp.latex(round(portfolio_betas[0], 2))} \\"
table_latex += rf"\beta_2 & {sp.latex(round(portfolio_betas[1], 2))} \\"
table_latex += rf"\text{{R}}^c & {round(r_squared_c, 2)} \\"
table_latex += rf"\text{{R}}^2_1 & {r_squared_factors_latex[0]} \\"
table_latex += rf"\text{{R}}^2_2 & {r_squared_factors_latex[1]} \\"
table_latex += r"\hline"
table_latex += r"\end{array}"

st.write("The tangency portfolio displays the following statistics:")
st.latex(table_latex)

st.write("Note: Expected return, variance, Sharpe ratio, and R-squared values are expressed in percentages.")



st.subheader('Portfolio Implied Risk Premia')


st.write(r"""
Given an initial allocation $x$, we deduce that this portfolio is optimal if the vector of implied risk premia is equal to:
         """)

st.latex(r'''
\begin{equation}
         \tilde{\pi} = \tilde{\mu} \frac{1}{\gamma} \Sigma x
\end{equation}
''')

st.write(r"""
         Assuming we know the Sharpe ratio of the initial allocation, we can dedure that:
         """)

st.latex(r'''
\begin{equation}
         \tilde{\pi} = SR_x \frac{\Sigma x}{ \sqrt{x^\top \Sigma x}}
\end{equation}
''')

x = w_star

# Define SR (Sharpe ratio) as a SymPy Matrix if it isn't already
SR = portfolio_sharpe_ratio

# Calculate implied risk premia using SymPy's sqrt instead of NumPy's sqrt
pi_tilde = SR * (Sigma @ x) / sp.sqrt((x.T @ Sigma @ x)[0])

# Convert the SymPy matrix to a LaTeX string
pi_tilde_latex = sp.latex(pi_tilde)

# Display the equation for the implied risk premia
st.latex(rf"\tilde{{\pi}} = SR_x \frac{{\Sigma x}}{{ \sqrt{{x^\top \Sigma x}}}} = {pi_tilde_latex}")

st.write(r"""
         This last equation gives the risk premia required, or priced in, by the investor to hold portfolio $x$.
We see that the implied risk premia $\tilde{\pi}$ are exactly the same to the theorethical risk premia $\pi$ in the case of the tangent portfolio.
         """)



st.write(r"""We can decompose the portfolio's asset exposures $x$ by the portfolio's risk factor exposures $y$ as follows:""")

st.latex(r'''
\begin{equation}
         x = B_y y + \breve{B}_y \breve{y}
\end{equation}
''')

st.write(r"""where $B_y = (B^\top)^{+}$ is the Moore-Penrose inverse of $B^\top$, and $\breve{B}_y$ is any 
         $n \times (n - m)$ matrix spanning the null space of $B_y$. $\breve{y}$ corresponds to $n-m$ residual (or additional) factors 
         that have no economic interpretation. It follows that:
         """)

st.latex(r'''
         \begin{equation}
         \begin{aligned}
         y = B_x x \\
         \end{aligned}
         \end{equation}
         ''')




B_x = B.T
y = B_x @ x

# display the equation for the factor exposures
y_latex = sp.latex(y)

st.write("With the tangent portfolio, we deduce the following factor exposures:")

st.latex(rf"y = B_x x = {sp.latex(B_x)} {sp.latex(x)} = {y_latex}")

st.write(r"""Again, we see that $y$ is the same as the factor exposures $\beta_x$ of the tangent portfolio.""")


# # Compute B^+ (Moore-Penrose inverse of B)
B_pseudo_inv = B.pinv()


st.write(r"""
         In order to calculate the vector of factor risk premia, we use the relationship between the factor risk premia and the asset risk premia and deduce that:
         """)

st.latex(r'''
\begin{equation}
         \psi = B^{+} \pi
\end{equation}
''')

st.write(r""" and:""")

st.latex(r'''
\begin{equation}
         \tilde{\psi} = B^{+}\tilde{\pi} = SR_x \frac{B^+ (B \Omega B^\top + D)x}{\sqrt{x^\top (B \Omega B^\top + D)x}}
\end{equation}
         ''')


psi_tilde = B_pseudo_inv * pi_tilde

# Convert the SymPy matrix to a LaTeX string
psi_tilde_latex = sp.latex(psi_tilde)

# Display the equation for the factor risk premia
st.latex(rf"\tilde{{\psi}} = {psi_tilde_latex}")

st.write(r"""
         where $B^+$ is the Moore-Penrose inverse of $B$. Again, we see that the implied factor risk premia $\tilde{\psi}$ are exactly the same to the theorethical factor risk premia $\psi$ in the case of the tangent portfolio.""")


st.write(r"""
         We can show that:
         """)

st.latex(r'''
         \begin{equation}
         \tilde{\pi}_x = x^\top \tilde{\pi} \neq y^\top \tilde{\psi}
         \end{equation}
         ''')


st.write(r"""Indeed, we have:""")


st.latex(r'''
         \begin{equation}
         \frac{\partial \sigma_x}{\partial x} = B_y \frac{\partial \sigma_x}{\partial y} + \breve{B}_y \frac{\partial \sigma_x}{\partial \breve{y}}
         \end{equation}
         ''')

st.write(r"""
We deduce that the marginal risk of the $j$-th risk factor is given by:
         """)

st.latex(r'''
         \begin{equation}
         \frac{\partial \sigma_x}{\partial y_j} = (B_{\sigma} \frac{\partial \sigma_x}{\partial x})_j
         \end{equation}
         ''')

st.write(r"""
         where $B_{\sigma} = B^{+}$ For the residual factors, we have:
         """)

st.latex(r'''
         \begin{equation}
         \frac{\partial \sigma_x}{\partial \breve{y}} = (\breve{B}_{\sigma} \frac{\partial \sigma_x}{\partial x})_j
         \end{equation}
         ''')

st.write(r"""
         where $\breve{B}_{\sigma} = \text{null}(B^{+})^\top = \breve{B}_x$. We deduce that the risk premium of the portfolio is given by:
         """)

st.latex(r'''
         \begin{equation}
         \tilde{\pi}_x = y^\top \tilde{\psi} + \breve{y}^\top \breve{\upsilon}
         \end{equation}
         ''')

st.write(r"""where $\breve{\upsilon}$ is the vector of risk premia associated with the residual factors, and:""")

st.latex(r'''
         \begin{equation}
         \begin{aligned}
         y = B_x x \\
         \tilde{\psi} = B_{\sigma} \tilde{\pi} \\
         \end{aligned}
         \end{equation}
         ''')

st.write(r"""
         The last two equations are the core relationships for decomposing the risk premium of a portfolio with respect to the factor risk premia.""")


st.write(r"""The common risk premium is given by:""") 

common_risk_premium = y.T * psi_tilde

st.latex(rf"\tilde{{\psi}}_x = y^\top \tilde{{\psi}} = {sp.latex(common_risk_premium)}")

st.write(r"""The total risk premium is given by:""")

total_risk_premium = common_risk_premium 

st.latex(rf"\tilde{{\pi}}_x = \tilde{{\psi}}_x + \breve{{\upsilon}}_x =  y^\top \tilde{{\psi}} + \breve{{y}}^\top \breve{{\upsilon}} = {sp.latex(total_risk_premium[0])}")

st.write(r"""Again, we see that the total risk premium $\tilde{\pi}_x$ is exactly the same to the theorethical risk premium $\pi_x$ in the case of the tangent portfolio.""")
