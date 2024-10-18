import streamlit as st
import sympy as sp
import numpy as np

st.set_page_config(page_title="Climate Risks")

st.title('Climate Risks in Portfolio Construction')

st.subheader('Mean-Variance Efficient Portfolio with Multi-factor Model')

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
We assume that $E(F) = \lambda$, $\text{Cov}(F) = \Omega$, $E(\epsilon) = 0$, $\text{Cov}(\epsilon) = D$,
and $\text{Cov}(F, \epsilon) = 0$.
         """)

st.write(r"""The vector of expected returns is given by:""")

st.latex(r"""
\begin{equation}
         \mu = B \lambda
\end{equation}
         """)


st.write(r"""The covariance matrix of asset returns is given by:""")

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
st.subheader("Climate Risks as a Rewarded Risk")

st.write(r"""
         Now that we have set up the stage, let's first consider what it would mean 
         for portfolio construction if climate risks is to be considered as a new risk factor. 
         We consider a simple example with a factor $F_1$ and the new climate risks factor $F_2$. Both 
         are assumed to be uncorrelated. To be considered as a new rewarded risk factor, it must be that the 
            expected return of the new factor is positive $E(F_2) > 0$. 
         We have the following expected returns for the factors:
         """)

default_F_1_risk_premia = 0.04
default_F_2_risk_premia = 0.01

st.sidebar.header("Climate Risks as a Rewarded Risk")

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
table_latex += rf"\text{{R}}^2_1 & {r_squared_factors_latex[0]} \\"
table_latex += rf"\text{{R}}^2_2 & {r_squared_factors_latex[1]} \\"
table_latex += r"\hline"
table_latex += r"\end{array}"

st.write("The tangency portfolio displays the following statistics:")
st.latex(table_latex)

st.write("Note: Expected return, variance, Sharpe ratio, and R-squared values are expressed in percentages.")

st.write(r"""
As you can see, the tangency portfolio loads both on the factor $F_1$ and the new climate risks factor $F_2$.
If climate risks is a new rewarded risk factor, then the mean-variance efficient portfolio will load on it. There is therefore no 
reason a priori to exclude climate risks from the investment universe, as investors are compensated for bearing the risk.
         """)

st.write(r"""
         In that case, if investors choose to exclude climate risks from their investment universe, they would be missing out on the
         expected returns associated with the factor. This would lead to suboptimal portfolios. It can be 
         a choice to exclude climate risks from the investment universe, but it should be a conscious choice.
         This choice would be based on the investor's preferences and beliefs, and not on the basis of
         risk and return considerations.
         """)

st.subheader("Climate Risks as an Unrewarded Risk")


st.sidebar.header("Climate Risks as an Unrewarded Risk")
default_F_1_risk_premia_2 = 0.04
default_F_2_risk_premia_2 = 0.00
F1_risk_premia_val_2 = st.sidebar.number_input('Rewarded Factor: $E(F_1)$',min_value=0., max_value=0.1, value=default_F_1_risk_premia_2, step=0.01)
F2_risk_premia_val_2 = st.sidebar.number_input('Unrewarded Factor: $E(F_2)$',min_value=0., max_value=0.0, value=default_F_2_risk_premia_2, step=0.01)

st.write(r"""
         Now, let's consider the case where climate risks is not rewarded. This would be the case if the expected return of the
            climate risks factor is zero $E(F_2) = 0$. We now have the following expected returns for the factors:
            """)


selected_risk_premia_2 = [F1_risk_premia_val_2, F2_risk_premia_val_2]
lambda_factors_2 = sp.Matrix(selected_risk_premia_2)

# Convert the SymPy matrix to a LaTeX string
lambda_latex_2 = sp.latex(lambda_factors_2)

# Display the equation for risk premia
st.latex(rf"\lambda = \begin{{bmatrix}} {F1_risk_premia_val_2:.2f} \\ {F2_risk_premia_val_2:.2f} \end{{bmatrix}}")

st.write("Keeping all the same except now that $E(F_2) = 0$, the new tangency portfolio displays the following statistics:")

# Step 1: Recalculate the optimal portfolio with the new expected returns

# Use the new selected risk premia for the factors
selected_risk_premia_2 = [F1_risk_premia_val_2, F2_risk_premia_val_2]
lambda_factors_2 = sp.Matrix(selected_risk_premia_2)

# Recompute the expected returns vector (mu)
mu_2 = B * lambda_factors_2

# Recompute the covariance matrix of assets (Sigma remains the same)
Sigma = B * Omega * B.T + D

# Compute the new optimal portfolio weights (tangency portfolio)
numerator_2 = Sigma.inv() * mu_2
denominator_2 = sp.Matrix([1]*n).T * Sigma.inv() * mu_2
w_star_2 = numerator_2 / denominator_2[0]

# Step 2: Compute the new portfolio statistics

# Expected return of the new portfolio
portfolio_expected_return_2 = w_star_2.T * mu_2   # Portfolio expected return
portfolio_variance_2 = w_star_2.T * Sigma * w_star_2  # Portfolio variance

# Extract scalar values from the 1x1 matrices
portfolio_expected_return_scalar_2 = round(portfolio_expected_return_2[0] * 100, 2)
portfolio_variance_scalar_2 = round(portfolio_variance_2[0] * 100, 2)

# Compute Sharpe ratio
portfolio_sharpe_ratio_2 = portfolio_expected_return_scalar_2 / sp.sqrt(portfolio_variance_scalar_2)

# Compute portfolio betas and R-squared as before
portfolio_betas_2 = B.T * w_star_2

# Compute the overall R-squared of the portfolio
numerator_r_squared_c_2 = (w_star_2.T * B * Omega * B.T * w_star_2)[0]  # Extract scalar
denominator_r_squared_c_2 = (w_star_2.T * Sigma * w_star_2)[0]  # Extract scalar

# Compute R-squared (overall)
r_squared_c_2 = numerator_r_squared_c_2 / denominator_r_squared_c_2 * 100

# Compute R-squared for each factor
r_squared_factors_2 = []
for j in range(B.shape[1]):  # Loop through each factor
    numerator_r_squared_j_2 = (w_star_2.T * B[:, j] * B[:, j].T * Omega[j, j] * w_star_2)[0]  # Extract scalar
    denominator_r_squared_j_2 = denominator_r_squared_c_2  # The denominator is the same
    r_squared_j_2 = numerator_r_squared_j_2 / denominator_r_squared_j_2
    r_squared_factors_2.append(r_squared_j_2)

# Convert values to latex-friendly strings
r_squared_c_latex_2 = sp.latex(round(r_squared_c_2, 2))
r_squared_factors_latex_2 = [sp.latex(round(r2 * 100, 2)) for r2 in r_squared_factors_2]

# Step 3: Display the results in a LaTeX table

table_latex_2 = r"\begin{array}{|c|c|} \hline \text{Metric} & \text{Value} \\ \hline"
table_latex_2 += rf"\text{{Expected Return}} & {sp.latex(portfolio_expected_return_scalar_2)} \\"  # New expected return in %
table_latex_2 += rf"\text{{Volatility}} & {sp.latex(sp.sqrt(portfolio_variance_scalar_2))} \\"  # New volatility in %
table_latex_2 += rf"\text{{Sharpe Ratio}} & {sp.latex(portfolio_sharpe_ratio_2)} \\"  # New Sharpe ratio in %
table_latex_2 += rf"\beta_1 & {sp.latex(round(portfolio_betas_2[0], 2))} \\"  # Beta 1
table_latex_2 += rf"\beta_2 & {sp.latex(round(portfolio_betas_2[1], 2))} \\"  # Beta 2
table_latex_2 += rf"\text{{R}}^2_1 & {r_squared_factors_latex_2[0]} \\"  # R-squared factor 1
table_latex_2 += rf"\text{{R}}^2_2 & {r_squared_factors_latex_2[1]} \\"  # R-squared factor 2
table_latex_2 += r"\hline"
table_latex_2 += r"\end{array}"

st.latex(table_latex_2)

st.write("Note: Expected return, variance, Sharpe ratio, and R-squared values are expressed in percentages.")

st.write(r"""
         As you can see, the tangency portfolio no longer loads on the new climate risks factor $F_2$.
         This is because the expected return of the factor is zero. The investor is not compensated for bearing the risk. 
         In this case, the mean-variance efficient portfolio
         does not load on the factor. Indeed, loading on the factor would increase the volatility of the portfolio without
            increasing the expected return. The investor would be taking on more risk without being compensated for it.
         """)

st.write(r"""
         If climate risks is an unrewarded risk factor, then the mean-variance efficient portfolio will not load on it.
            In that case, the investor can choose to exclude climate risks from the investment universe without affecting the
            expected return of the portfolio. The investor would be taking on less risk without sacrificing expected return.
            """)


st.subheader("When $F$ loads on Climate Risks")

st.write(r"""
Now the problem, in the case where climate risks is an unrewarded risk factor, lies in the way $F$ is defined.
         Those are the excess returns of long-short portfolios. The common 
         practice in the academic finance literature is 
         to create factor-mimicking portfolios by sorting on characteristics positively associated with expected returns.
         The resultant portfolios, which go long a portfolio of high characteristic stocks and short a portfolio of low characteristic stocks,
         helps to estimate the associated risk premia (see Fama and French 1992 and 2015).
         """)


st.write(r"""Daniel $\textit{et al.}$ (2020) have shown that, if the characteristics are positively correlated with the unrewarded factor,
         the resulting $F$ (factor-mimicking portfolios) will load on the unrewarded factor. This is because the long-short portfolios
            will load on the unrewarded factor.""")

st.write(r"""

         """)

N = 12  # Number of assets

# Predefined fixed values for C
B_1 = sp.Matrix([1, 1, 1, 1, 1,1, -1, -1, -1,-1,-1,-1])

st.sidebar.header("Input Desired Cross-Correlation")

# Ask user for the desired correlation coefficient
correlation = st.sidebar.selectbox(
    "Select the correlation between $B_1$ and $B_2$", 
    ("0", "1/3", "2/3")
)

# Predefined sets of gamma based on the correlation choices
B_2_sets = {
    "0": [1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1],
    "1/3": [1, 1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1],
    "2/3": [1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1],
}

selected_B_2 = B_2_sets[correlation]

# Define the matrix B_2
B_2 = sp.Matrix(selected_B_2)

# concat B_1 and B_2 to form B
B = B_1.row_join(B_2)

st.write(r"""Too see this, let's define a universe of 12 assets. We have the following matrix $B$ of factor loadings:""")

st.latex(fr"""B = {sp.latex(B)}""")


# Step 1: Compute the means of beta and gamma
B1_mean = sp.Rational(sum(B_1), N)
B2_mean = sp.Rational(sum(B_2), N)
# Step 2: Compute the covariance between beta and gamma
cov_B1_B2 = sp.Rational(0, 1)
for i in range(N):
    cov_B1_B2 += (B_1[i] - B1_mean) * (B_2[i] - B2_mean)
cov_B1_B2 /= N

# Step 3: Compute the standard deviations of beta and gamma
std_B1 = sp.sqrt(sum((B_1[i] - B1_mean)**2 for i in range(N)) / N)
std_B2 = sp.sqrt(sum((B_2[i] - B2_mean)**2 for i in range(N)) / N)

# Step 4: Compute the correlation
correlation = cov_B1_B2 / (std_B1 * std_B2)

beta_np = np.array(B_1).flatten()

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
w_ls = w_long + w_short

# Display the weights
st.write(r"""
         The weights of the factor-mimicking portfolio are:
         """)

weights_latex = r"\begin{bmatrix} "
for i in range(N):
    weights_latex += f"{sp.latex(w_ls[i])} & "
weights_latex = weights_latex[:-2] + r" \end{bmatrix}"

st.latex(r"""
w_f^T = """ + weights_latex)

# Recompute the expected returns and portfolio statistics as before

# Define lambda_factors_2 based on the new risk premia
lambda_factors_2 = sp.Matrix([0.04, 0.00])

# Recompute expected returns (mu)
mu_3 = B * lambda_factors_2

# Define the covariance matrix Omega for the factors
Omega = sp.diag(0.20**2, 0.2**2)

# Define the diagonal matrix D for idiosyncratic risks
D = sp.diag(0.12**2, 0.10**2, 0.08**2, 0.10**2, 0.12**2, 0.15**2, 0.10**2, 0.09**2, 0.13**2, 0.14**2, 0.11**2, 0.16**2)

# Recompute the covariance matrix of assets (Sigma)
Sigma = B * Omega * B.T + D

# Compute the expected return and variance of the portfolio
portfolio_expected_return_f = w_ls.T * mu_3   # Portfolio expected return
portfolio_variance_f = w_ls.T * Sigma * w_ls  # Portfolio variance

# Extract scalar values from the 1x1 matrices
portfolio_expected_return_f_scalar = portfolio_expected_return_f[0]
portfolio_variance_f_scalar = portfolio_variance_f[0] 


# Compute portfolio betas
portfolio_betas_f = B.T * w_ls
# Display the correlation formula
st.write(r"""
The correlation between $B_1$ and $B_2$ is:
""")
st.latex(r"\rho(B_1, B_2) = " + sp.latex(correlation.simplify()))


st.write(r"""
         Therefore the portfolio $f$ has the following $\beta$ coefficients:""")

portfolio_betas_f_latex = sp.latex(portfolio_betas_f)
st.latex(fr'\beta_f = {portfolio_betas_f_latex}')

st.write(r"""
The higher the correlation between the characteristic and $B_2$, the higher the loading of the portfolio $f$ on the factor $B_2$.
         """)

st.write(r"""The expected return of $f$ is directly linked to its loading on the factors $\beta_f$: """)


portfolio_expected_return_f_scalar_latex = sp.latex(portfolio_expected_return_f_scalar)
st.latex(fr"""E(f) = \beta^\top \lambda = {portfolio_betas_f_latex}^T {lambda_latex_2} = {portfolio_expected_return_f_scalar_latex}""")

st.write(r"""and the portfolio variance is:""")

portfolio_variance_f_scalar_latex = sp.latex(portfolio_variance_f_scalar)

st.latex(fr"""\sigma_f^2 = {portfolio_variance_f_scalar_latex}""")

st.write(r"""
The variance of the portfolio $f$ explained by the unrewarded factor is:
         """)

numerator_r_squared_c_3 = (w_ls.T * B_2 * Omega[1, 1] * B_2.T * w_ls)[0]  # Extract scalar
denominator_r_squared_c_3 = (w_ls.T * Sigma * w_ls)[0]  # Extract scalar

r_squared_c_3 = numerator_r_squared_c_3 / denominator_r_squared_c_3 

st.latex(rf"\text{{R}}^2 = \frac{{w_f^\top B_2 \Omega_2 B_2^\top w_f}}{{w_f^\top \Sigma w_f}} = {sp.latex(r_squared_c_3)}")

st.write(r"""
So now we have a situation where the factor-mimicking portfolio $F_1$ loads on the unrewarded factor $F_2$.
         """)

st.write(fr"""
We keep all the same except now that $\sigma_1^2 = {round(portfolio_variance_f_scalar,2)}$, the new tangency portfolio displays the following statistics:
         """)


# Matrix of factor loadings B (5 assets, 2 factors)
B = sp.Matrix([[0.5, -0.3], [1.2, 0.6], [1.4, -0.3], [0.7, 0.3], [0.8, -0.9]])

Omega = sp.diag(portfolio_variance_f_scalar, 0.2**2)

# Idiosyncratic risk matrix D (5x5)
D = sp.diag(0.12**2, 0.10**2, 0.08**2, 0.10**2, 0.12**2)

# Expected returns vector for factors lambda (2x1)
lambda_factors = sp.Matrix([portfolio_expected_return_f_scalar, 0.00])

# Compute expected returns for the 5 assets
mu = B * lambda_factors  # This will be a (5x1) vector

# Compute the covariance matrix of asset returns
Sigma = B * Omega * B.T + D  # Sigma will be a (5x5) matrix

# Compute the new optimal portfolio weights (tangency portfolio)
numerator_2 = Sigma.inv() * mu_2
denominator_2 = sp.Matrix([1]*n).T * Sigma.inv() * mu_2
w_star_2 = numerator_2 / denominator_2[0]

# Step 2: Compute the new portfolio statistics

# Expected return of the new portfolio
portfolio_expected_return_2 = w_star_2.T * mu_2   # Portfolio expected return
portfolio_variance_2 = w_star_2.T * Sigma * w_star_2  # Portfolio variance

# Extract scalar values from the 1x1 matrices
portfolio_expected_return_scalar_2 = round(portfolio_expected_return_2[0] * 100, 2)
portfolio_variance_scalar_2 = round(portfolio_variance_2[0] * 100, 2)

# Compute Sharpe ratio
portfolio_sharpe_ratio_2 = portfolio_expected_return_scalar_2 / sp.sqrt(portfolio_variance_scalar_2)

# Compute portfolio betas and R-squared as before
portfolio_betas_2 = B.T * w_star_2

# Compute the overall R-squared of the portfolio
numerator_r_squared_c_2 = (w_star_2.T * B * Omega * B.T * w_star_2)[0]  # Extract scalar
denominator_r_squared_c_2 = (w_star_2.T * Sigma * w_star_2)[0]  # Extract scalar

# Compute R-squared (overall)
r_squared_c_2 = numerator_r_squared_c_2 / denominator_r_squared_c_2 * 100

# Compute R-squared for each factor
r_squared_factors_2 = []
for j in range(B.shape[1]):  # Loop through each factor
    numerator_r_squared_j_2 = (w_star_2.T * B[:, j] * B[:, j].T * Omega[j, j] * w_star_2)[0]  # Extract scalar
    denominator_r_squared_j_2 = denominator_r_squared_c_2  # The denominator is the same
    r_squared_j_2 = numerator_r_squared_j_2 / denominator_r_squared_j_2
    r_squared_factors_2.append(r_squared_j_2)

# Convert values to latex-friendly strings
r_squared_c_latex_2 = sp.latex(round(r_squared_c_2, 2))
r_squared_factors_latex_2 = [sp.latex(round(r2 * 100, 2)) for r2 in r_squared_factors_2]

# Step 3: Display the results in a LaTeX table

table_latex_2 = r"\begin{array}{|c|c|} \hline \text{Metric} & \text{Value} \\ \hline"
table_latex_2 += rf"\text{{Expected Return}} & {sp.latex(portfolio_expected_return_scalar_2)} \\"  # New expected return in %
table_latex_2 += rf"\text{{Volatility}} & {sp.latex(sp.sqrt(portfolio_variance_scalar_2))} \\"  # New volatility in %
table_latex_2 += rf"\text{{Sharpe Ratio}} & {sp.latex(portfolio_sharpe_ratio_2)} \\"  # New Sharpe ratio in %
table_latex_2 += rf"\beta_1 & {sp.latex(round(portfolio_betas_2[0], 2))} \\"  # Beta 1
table_latex_2 += rf"\beta_2 & {sp.latex(round(portfolio_betas_2[1], 2))} \\"  # Beta 2
table_latex_2 += rf"\text{{R}}^2_1 & {r_squared_factors_latex_2[0]} \\"  # R-squared factor 1
table_latex_2 += rf"\text{{R}}^2_2 & {r_squared_factors_latex_2[1]} \\"  # R-squared factor 2
table_latex_2 += r"\hline"
table_latex_2 += r"\end{array}"

st.latex(table_latex_2)