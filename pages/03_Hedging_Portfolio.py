import streamlit as st
import numpy as np
import plotly.graph_objs as go
import numpy as np
import sympy as sp
import sympy.stats as stats
import plotly.graph_objs as go
import pandas_datareader as pdr
import statsmodels.api as sm
from plotnine import *
from mizani.formatters import date_format
from mizani.breaks import date_breaks
import pandas as pd
from statsmodels.regression.rolling import RollingOLS


import yfinance as yf


st.title('Hedging from Unrewarded Risks')

st.write(r"""
We have seen that, as of today, no risk premium seems to be associated with climate risks. It is therefore to be treated as unrewarded risk. 
We have seen that if exposure to the rewarded risk is correlated with exposure to unrewarded risk, a portfolio rightly exposed to rewarded risk is mean-variance inefficient because
         it loads in the unrewarded risk. We can improve upon this. We want a method that (1) let untouched the expected return and (2) reduce the variance of our portfolio.
         """)

st.subheader('Characteristic-Balanced Hedge Portfolio')

st.write(r"""
         We follow Daniel $\textit{et al.}$ (2020) and show 
how we can improve the portfolio $c$ from the first section.
We can form a hedge portfolio $h$ that is long in assets with high exposure to the unrewarded factor $g$ and short in assets with low exposure to $g$. 
To form a hedge portfolio that does not affect the exposure to the rewarded factor of the initial portfolio, 
we need to form a hedge portfolio neutral to the rewarded factor $f$.
""")



st.write(r"""
To do so, we can form a portfolio sorted on the characteristic $x$ as before. Then, 
         within each sleeve (long and short), we can sort the assets on the loading on the unrewarded risk $\gamma$.
We end up with $2\times2$ portfolios. We then go long portfolios with high $\gamma$ and short portfolios with low $\gamma$,
         equally-weighted.

Of course, in our simple framework, depending on the correlation between $\beta$ and $\gamma$, we may end up 
with unrealistic sleeves (ie. assets with different $\gamma$ in the same sleeve). But this will illustrate the idea.      
""")


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


# Create a LaTeX table for the beta and gamma inputs
table_latex = r"\begin{array}{|c|c|c|} \hline Asset & \beta & \gamma \\ \hline "
for i in range(N):
    table_latex += f"{i+1} & {fixed_beta[i]} & {sp.latex(selected_gamma[i])} \\\\ \\hline "
table_latex += r"\end{array}"
st.latex(table_latex)

# Convert beta and gamma inputs to Sympy matrices
gamma = sp.Matrix(selected_gamma)
# Convert beta inputs to Sympy matrices
beta = sp.Matrix(fixed_beta)

# Convert beta and gamma inputs to NumPy arrays
beta_np = np.array(fixed_beta)
gamma_np = np.array(selected_gamma)

# Get the indices of the sorted beta values
sorted_indices = np.argsort(beta_np)

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


st.write(r"""
         The weights of the portfolio are:
         """)

# Prepare weights in LaTeX format as a row vector
weights_latex_h = r"\begin{bmatrix} "
for i in range(N):
    weights_latex_h += f"{w_h[i]} & "
weights_latex_h = weights_latex_h[:-2] + r" \end{bmatrix}"  # Remove the last "&" and close the matrix

st.latex(r"""
w_h^T = """ + weights_latex_h)

# Define priced factor as normal random variable with variance properties
f = stats.Normal('f', 0, sp.symbols('sigma_f'))  # Priced factor f with E[f] = 0 and var(f) = sigma_f^2
# Characteristic premium
lambda_ = sp.symbols('lambda')
# Define idiosyncratic errors epsilon_i as random variables with zero mean and variance sigma_epsilon^2
epsilon = sp.Matrix([stats.Normal(f'epsilon{i+1}', 0, sp.symbols('sigma_epsilon')) for i in range(N)])
# Define priced and unpriced factors as normal random variables with variance properties
g = stats.Normal('g', 0, sp.symbols('sigma_g'))  # Unpriced factor g with E[g] = 0 and var(g) = sigma_g^2

# Define symbols for variances of the factors and idiosyncratic error
sigma_g = sp.symbols('sigma_g')
# Define symbols for variances of the factors and idiosyncratic error
sigma_f, sigma_epsilon = sp.symbols('sigma_f sigma_epsilon')


# Step 1: Define the portfolio return formula symbolically
r_h = w_h.dot(beta * (f + lambda_) + gamma * g + epsilon)

st.write(r"""
         We can now compute the return of the hedge portfolio as:
                """)

st.latex(r"""
         \begin{equation}
         \begin{aligned}
r_h = w_h^\top (\beta (f + \lambda) + \gamma g + \epsilon) \\
         = w_h^\top (\gamma g + \epsilon)
            \end{aligned}
         \end{equation}
            """)

st.latex(f"""r_h = {sp.latex(r_h)}""")

st.write(r"""
         because $w_h^\top \beta = 0$. Therefore the expected return is:
            """)    

# Step 2: Take the expectation using sympy.stats
expected_r_h = w_h.dot(beta) * lambda_

st.latex(r"""
\begin{equation}
            \mathbb{E}[r_h] = w_h^\top \mu = w_h^\top \beta \lambda
\end{equation}
         """)

st.latex(f"E[r_h] = {sp.latex(expected_r_h)}")

st.write(r"""which is nill as expected. The hedge portfolio $h$ is designed to not alter the expected return of the portfolio $c$.
The double sorting on the rewarded and unrewarded factors allows us to construct a portfolio that is neutral to the rewarded factor, and
         therefore could be use without affecting the exposure to the rewarded factor of the initial portfolio.
         """)

st.write(r"""
The hedge portfolio $h$ has a variance given by:
         """)

st.latex(r"""
\begin{equation}
         \begin{aligned}
         \sigma_h^2 = w_h^\top \Sigma w_h = w_h^\top \left( \beta \beta^\top \sigma_f^2 + \gamma \gamma^\top \sigma_g^2 + \sigma_\epsilon^2 I \right) w_h \\
            \end{aligned}
         \end{equation}
         """)

covariance_matrix_f = beta * beta.T * sigma_f**2

covariance_matrix_g = gamma * gamma.T * sigma_g**2

# Define the covariance matrix for idiosyncratic errors (N x N identity matrix scaled by sigma_epsilon**2)
covariance_matrix_epsilon = sigma_epsilon**2 * sp.eye(N)

# Combine the covariance matrices
covariance_matrix = covariance_matrix_f + covariance_matrix_g +  covariance_matrix_epsilon

# Calculate the total portfolio variance as w.T * covariance_matrix * w
var_h = (w_h.T * covariance_matrix * w_h)[0]

st.latex(f"\\sigma^2_h = {sp.latex(var_h)}")

st.write(r"""
         because $w_h^\top \beta = 0$. 
         """)

st.write(r"""
Thus, because it loads on the unrewarded factors, the variance of the portfolio is partially driven by 
         the variance of the unrewarded factor $g$ (ie. the unrewarded risk).
         """)

st.subheader('Optimal Hedge Ratio')

st.write(r"""
We now have an investment tool - a hedging portfolio - that helps to reduce the exposure to unrewarded risks,
while keeping the expected return of the portfolio unchanged. 
         """)


st.write(r"""
         Daniel $\textit{et al.}$ (2020) show that we can improve the portfolio $c$ by combining it with the hedge portfolio $h$ in order to maximize the Sharpe ratio.
         Given that the hedge portfolio has zero expected return, this is equivalent 
         to finding the combination of $c$ and $h$ that minimizes the variance of the resulting portfolio:
            """)

st.latex(r"""\min_{\delta} \sigma^2(r_c - \delta r_h)""")

st.write(r"""
         with:
         """)

st.latex(r"""
         \begin{equation}
            \sigma^2(r_c - \delta r_h) = \sigma^2_c + \delta^2 \sigma^2_h - 2 \delta \text{Cov}(r_c, r_h)
        \end{equation}
    """)

st.write(r"""
         Taking the derivative with respect to $\delta$ and setting it to zero, we find the optimal hedge ratio $\delta^*$:
            """)

st.latex(r"""
            \begin{equation}
         \begin{aligned}
         \frac{\partial}{\partial \delta} \sigma^2(r_c - \delta r_h) = 2 \delta \sigma^2_h - 2 \text{Cov}(r_c, r_h) = 0 \\
            \delta^* = \frac{\text{Cov}(r_c, r_h)}{\sigma^2_h} \\
         \end{aligned}   
         \end{equation}
        """)

st.write(r"""
We know that $w^\top_h \beta = 0$ and $\text{Cov}(f, g) = 0$. Therefore, we can simplify a bit the covariance 
            between $r_c$ and $r_h$:
         """)

st.latex(r"""
\begin{equation}
            \begin{aligned}
            \text{Cov}(r_c, r_h) = \text{Cov}(w_c^\top \gamma g, w_h^\top \gamma g) + \text{Cov}(w_c^\top \epsilon, w_h^\top \epsilon) \\
         = (w_c^\top \gamma) (w_h^\top \gamma) \text{Cov}(g,g) + w_c^\top w_h \text{Cov}(\epsilon, \epsilon) \\
            = (w_c^\top \gamma) (w_h^\top \gamma) \sigma_g^2 \\
            = \gamma_c \gamma_h \sigma_g^2
\end{aligned}
         \end{equation}
            """)

st.write(r"""
         Similarly, we can simplify the variance of the hedge portfolio:
            """)

st.latex(r"""
\begin{equation}
            \begin{aligned}
            \sigma^2_h = w_h^\top \Sigma w_h = w_h^\top \left( \beta \beta^\top \sigma_f^2 + \gamma \gamma^\top \sigma_g^2 + \sigma_\epsilon^2 I \right) w_h \\
            = w_h^\top \gamma \gamma^\top \sigma_g^2 \\
            = \gamma_h^2 \sigma_g^2 
\end{aligned}
            \end{equation}
""")


st.write(r"""
            Therefore, the optimal hedge ratio is:
            """)

st.latex(r"""
\begin{equation}
            \begin{aligned}
            \delta^* = \frac{\gamma_c \gamma_h \sigma_g^2}{\gamma_h^2 \sigma_g^2} \\
         = \frac{\gamma_c} {\gamma_h} \\ 
\end{aligned}
            \end{equation}
            """)


# Use SymPy's Rational to keep weights as fractions
w_long_c = sp.Matrix([0] * N)
w_short_c = sp.Matrix([0] * N)

# Get the indices of the sorted beta values
sorted_indices_c = np.argsort(beta_np)

# Assign long positions (1/3) to the top 3 assets
for idx in sorted_indices_c[-6:]:
    w_long_c[idx] = sp.Rational(1, 6)

# Assign short positions (-1/3) to the bottom 3 assets
for idx in sorted_indices_c[:6]:
    w_short_c[idx] = sp.Rational(-1, 6)

# Combine long and short positions to form the final weight vector
w_c = w_long_c + w_short_c

var_c = (w_c.T * covariance_matrix * w_c)[0]

st.latex(f"\\sigma^2_c = {sp.latex(var_c)}")

gamma_c = w_c.dot(gamma)
gamma_h = w_h.dot(gamma)

delta_optim = gamma_c / gamma_h

st.write(fr"""
Which, in our case with $\gamma_c = {gamma_c}$ and $\gamma_h = {gamma_h}$, gives:
         """)

st.latex(f"\delta^* = {sp.latex(delta_optim)}")


gamma_optim = gamma_c - delta_optim * gamma_h


w_optim = w_c - delta_optim * w_h

gamma_optim = w_optim.dot(gamma)

st.write(r"""
         The loading on the unrewarded factor of the combined portfolio is:""")

st.latex(f"\gamma^* = {sp.latex(gamma_optim)}")

st.write(r"""
         The variance of the combined portfolio is:
            """)

var_optim = (w_optim.T * covariance_matrix * w_optim)[0]

st.latex(f"\\sigma^2(r_c - \delta^* r_h) = {sp.latex(var_optim)}")

st.write(r"""
         While the expected return of the combined portfolio is:
         """)

expected_r_optim = w_optim.dot(beta) * lambda_

st.latex(f"E[r_c - \delta^* r_h] = {sp.latex(expected_r_optim)}")

st.write(r"""
         Therefore the new Sharpe ratio is:
         """)

sharpe_optim = expected_r_optim / sp.sqrt(var_optim)

st.latex(f"\\text{{Sharpe ratio}}^* = {sp.latex(sharpe_optim)}")


st.write(r""" while the Sharpe ratio of the initial portfolio $c$ was:
         """)

sharpe_c = w_c.dot(beta) * lambda_ / sp.sqrt(var_c)

st.latex(f"\\text{{Sharpe ratio}} = {sp.latex(sharpe_c)}")

# # Function to retrieve S&P 500 tickers from Wikipedia
# @st.cache_data
# def get_sp500_tickers():
#     url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
#     sp500_table = pd.read_html(url)[0]  # Get the first table on the page
#     tickers = sp500_table['Symbol'].tolist()  # Convert the 'Symbol' column to a list
#     return tickers

# # Function to download stock prices using yfinance and cache it
# @st.cache_data
# def download_stock_data(tickers, start_date, end_date):
#     # Download stock prices
#     prices_daily = yf.download(
#         tickers=tickers, 
#         start=start_date, 
#         end=end_date, 
#         progress=False
#     )
    
#     # Process the downloaded prices
#     prices_daily = (prices_daily
#         .stack()
#         .reset_index(level=1, drop=False)
#         .reset_index()
#         .rename(columns={
#             "Date": "date",
#             "Ticker": "symbol",
#             "Open": "open",
#             "High": "high",
#             "Low": "low",
#             "Close": "close",
#             "Adj Close": "adjusted",
#             "Volume": "volume"}
#         )
#     )
    
#     return prices_daily

# @st.cache_data
# def load_ff3_data():
#     """Load the Fama-French factors data."""
#     start_date = '2000-01-01'
#     end_date = '2019-06-30'
#     factors_ff3_daily_raw = pdr.DataReader(
#         name="F-F_Research_Data_Factors_daily",
#         data_source="famafrench",
#         start=start_date,
#         end=end_date)[0]
#     factors_ff3_daily = (factors_ff3_daily_raw
#                          .divide(100)
#                          .reset_index(names="date")
#                          .rename(str.lower, axis="columns")
#                          .rename(columns={"mkt-rf": "mkt_excess"}))
#     return factors_ff3_daily

# # Retrieve the S&P 500 tickers
# sp500_tickers = get_sp500_tickers()

# # Set the date range
# start_date = '2000-01-01'
# end_date = '2019-06-30'

# # Download stock price data for the selected tickers
# prices_daily = download_stock_data(sp500_tickers, start_date, end_date)

# # Compute daily returns
# returns_daily = (prices_daily
#   .assign(ret=lambda x: x.groupby("symbol")["adjusted"].pct_change())
#   .get(["symbol", "date", "ret"])
#   .dropna(subset=["ret"])
# )

# factors_ff3_daily = load_ff3_data()
# bmg = pd.read_excel('data/carbon_risk_factor_updated.xlsx', sheet_name='daily', index_col=0)

# # Merge HML and bmg
# data = pd.merge(factors_ff3_daily[['date', 'hml']], bmg, on='date')
# data = pd.merge(data, returns_daily, on='date')

# st.write(data.columns)

# # Caching the rolling beta estimation
# @st.cache_data
# def roll_beta_estimation(data, window_size, min_obs, factor):
#     """Calculate rolling beta estimation."""
#     data = data.sort_values("date")

#     # Ensure the group has at least the minimum number of observations
#     if len(data) < window_size:
#         return pd.Series([np.nan] * len(data), index=data.index)  # Return NaNs if not enough data


#     result = (RollingOLS.from_formula(
#       formula=f"ret ~ {factor}",
#       data=data,
#       window=window_size,
#       min_nobs=min_obs,
#       missing="drop")
#       .fit()
#       .params.get(f"{factor}")
#     )
    
#     result.index = data.index
#     return result


# # Cache the full chain for beta_hml computation
# @st.cache_data
# def compute_beta(data, window_size, min_obs, factor):
#     return (data
#       .groupby(["symbol"])
#       .apply(
#              lambda x: x.assign(
#                  beta=roll_beta_estimation(x, window_size, min_obs, factor))
#       )
#       .reset_index(drop=True)
#       .dropna()
#     )


# window_size = 126  # 6-month rolling window
# min_obs = 10  # Minimum number of observations required for each window

# # Compute rolling beta for HML factor with caching
# beta_hml = compute_beta(data, window_size, min_obs, "hml")

# # Compute rolling beta for BMG factor with caching
# beta_bmg = compute_beta(beta_hml, window_size, min_obs, "BMG")

# st.write(beta_hml.head())
# st.write(beta_bmg.head())

st.subheader('Conclusion')

st.write(r"""
To manage exposure to unrewarded risk, we have seen that we can form a hedge portfolio that is not exposed 
         to the rewarded factor - such that it will not affect expected returns of the initial portfolio - and 
         that loads on the unrewarded factor - such that it will affect the initial portfolio variance. 
            """)
