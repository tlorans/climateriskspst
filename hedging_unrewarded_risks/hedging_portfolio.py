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
         it loads in the unrewarded risk. We can improve upon this. We want a method that (1) let untouched the rewarded risk exposure and (2) eliminate the unrewarded risk exposure.
         """)

st.subheader('Characteristic-Balanced Hedge Portfolio')

st.write(r"""
         We follow Daniel $\textit{et al.}$ (2020) and show 
how we can improve the portfolio $c$ from the first section.
We can form a hedge portfolio $h$ that is long in assets with high exposure to the unrewarded factor $g$ and short in assets with low exposure to $g$. 
To form a hedge portfolio that does not affect the exposure to the rewarded factor of the initial portfolio, 
we need to form a hedge portfolio neutral to the rewarded factor $f$. That is, we want:
""")

st.latex(
    r"""
\begin{equation}
w_h^T x = 0
\end{equation}
"""
)

st.write(r"""
To do so, we can form a portfolio sorted on the characteristic $x$ as before. Then, 
         within each sleeve (long and short), we can sort the assets on the loading on the unrewarded risk $\gamma$.
We end up with $2\times3$ portfolios. We then go long portfolios with high $\gamma$ and short portfolios with low $\gamma$,
         equally-weighted.
         
""")


default_gamma = [1, 1, -1, 1, -1, -1]
N = 6  # Number of assets

# Predefined values for beta and gamma
default_beta = [1, 1, 1, -1, -1, -1]



# Create a LaTeX table for the beta and gamma inputs
table_latex = r"\begin{array}{|c|c|c|} \hline Asset & x & \gamma \\ \hline "
for i in range(N):
    table_latex += f"{i+1} & {default_beta[i]} & {default_gamma[i]} \\\\ \\hline "
table_latex += r"\end{array}"
st.latex(table_latex)

# Convert beta and gamma inputs to Sympy matrices
gamma = sp.Matrix(default_gamma)
# Convert beta inputs to Sympy matrices
beta = sp.Matrix(default_beta)
# Convert beta and gamma inputs to NumPy arrays
beta_np = np.array(default_beta)
gamma_np = np.array(default_gamma)

# Get the indices of the sorted beta values
sorted_indices = np.argsort(beta_np)

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
w = w_long + w_short


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
         We can now compute the return of the hedge portfolio as:
                """)

st.latex(f"""r_h = {sp.latex(portfolio_return_with_g)}""")

# Step 2: Take the expectation using sympy.stats
expected_portfolio_return_with_g = stats.E(portfolio_return_with_g)

loading_on_f = w.dot(beta)
loading_on_g = w.dot(gamma)
st.write(f"""

The loading of the portfolio $h$ on the rewarded factor $f$ is zero,
and the loading on the unrewarded factor $g$ is {loading_on_g}. Therefore, 
expected return of the portfolio $h$ is :
         """)


# Step 2: Take the expectation using sympy.stats
expected_portfolio_return_with_g = stats.E(portfolio_return_with_g) 

st.latex(f"E[r_h] = {sp.latex(expected_portfolio_return_with_g)}")



# Function to retrieve S&P 500 tickers from Wikipedia
@st.cache_data
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500_table = pd.read_html(url)[0]  # Get the first table on the page
    tickers = sp500_table['Symbol'].tolist()  # Convert the 'Symbol' column to a list
    return tickers

# Function to download stock prices using yfinance and cache it
@st.cache_data
def download_stock_data(tickers, start_date, end_date):
    # Download stock prices
    prices_daily = yf.download(
        tickers=tickers, 
        start=start_date, 
        end=end_date, 
        progress=False
    )
    
    # Process the downloaded prices
    prices_daily = (prices_daily
        .stack()
        .reset_index(level=1, drop=False)
        .reset_index()
        .rename(columns={
            "Date": "date",
            "Ticker": "symbol",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adjusted",
            "Volume": "volume"}
        )
    )
    
    return prices_daily

@st.cache_data
def load_ff3_data():
    """Load the Fama-French factors data."""
    start_date = '2000-01-01'
    end_date = '2019-06-30'
    factors_ff3_daily_raw = pdr.DataReader(
        name="F-F_Research_Data_Factors_daily",
        data_source="famafrench",
        start=start_date,
        end=end_date)[0]
    factors_ff3_daily = (factors_ff3_daily_raw
                         .divide(100)
                         .reset_index(names="date")
                         .rename(str.lower, axis="columns")
                         .rename(columns={"mkt-rf": "mkt_excess"}))
    return factors_ff3_daily

# Retrieve the S&P 500 tickers
sp500_tickers = get_sp500_tickers()

# Set the date range
start_date = '2000-01-01'
end_date = '2019-06-30'

# Download stock price data for the selected tickers
prices_daily = download_stock_data(sp500_tickers, start_date, end_date)

# Compute daily returns
returns_daily = (prices_daily
  .assign(ret=lambda x: x.groupby("symbol")["adjusted"].pct_change())
  .get(["symbol", "date", "ret"])
  .dropna(subset=["ret"])
)

factors_ff3_daily = load_ff3_data()
bmg = pd.read_excel('data/carbon_risk_factor_updated.xlsx', sheet_name='daily', index_col=0)

# Merge HML and bmg
data = pd.merge(factors_ff3_daily[['date', 'hml']], bmg, on='date')
data = pd.merge(data, returns_daily, on='date')

st.write(data.columns)

# Caching the rolling beta estimation
@st.cache_data
def roll_beta_estimation(data, window_size, min_obs, factor):
    """Calculate rolling beta estimation."""
    data = data.sort_values("date")

    # Ensure the group has at least the minimum number of observations
    if len(data) < window_size:
        return pd.Series([np.nan] * len(data), index=data.index)  # Return NaNs if not enough data


    result = (RollingOLS.from_formula(
      formula=f"ret ~ {factor}",
      data=data,
      window=window_size,
      min_nobs=min_obs,
      missing="drop")
      .fit()
      .params.get(f"{factor}")
    )
    
    result.index = data.index
    return result


# Cache the full chain for beta_hml computation
@st.cache_data
def compute_beta(data, window_size, min_obs, factor):
    return (data
      .groupby(["symbol"])
      .apply(
             lambda x: x.assign(
                 beta=roll_beta_estimation(x, window_size, min_obs, factor))
      )
      .reset_index(drop=True)
      .dropna()
    )


window_size = 126  # 6-month rolling window
min_obs = 10  # Minimum number of observations required for each window

# Compute rolling beta for HML factor with caching
beta_hml = compute_beta(data, window_size, min_obs, "hml")

# Compute rolling beta for BMG factor with caching
beta_bmg = compute_beta(beta_hml, window_size, min_obs, "BMG")

st.write(beta_hml.head())
st.write(beta_bmg.head())

st.write(r"""
The double sorting on the rewarded and unrewarded factors allows us to construct a portfolio that is neutral to the rewarded factor, and
         therefore could be use without affecting the exposure to the rewarded factor of the initial portfolio.
         """)

st.subheader('Hedge Portfolio Variance')

st.write(r"""
So, we now have our hedge portfolio $h$ that is neutral to the rewarded factor $f$ and long in assets with high exposure to the unrewarded factor $g$ and short in assets with low exposure to $g$.
The resulting hedge portfolio $h$ has a variance given by:
         """)

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

st.latex(f"\\sigma^2_h = {sp.latex(total_portfolio_variance_with_g)}")


st.write(r"""
Thus, because it loads on the unrewarded factors, the variance of the portfolio is partially driven by 
         the variance of the unrewarded factor $g$ (ie. the unrewarded risk).

         """)

st.subheader('Conclusion')

st.write(r"""
To manage exposure to unrewarded risk, we have seen that we can form a hedge portfolio that is not exposed 
         to the rewarded factor - such that it will not affect expected returns of the initial portfolio - and 
         that loads on the unrewarded factor - such that it will affect the initial portfolio variance. 
            """)