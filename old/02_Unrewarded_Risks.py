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

st.title('Unrewarded Risks')


st.subheader('Rewarded and Unrewarded Risks Cross-Correlation')

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

# Display the table of beta and gamma
table_latex = r"\begin{array}{|c|c|c|} \hline Asset & \beta & \gamma \\ \hline "
for i in range(N):
    table_latex += f"{i+1} & {fixed_beta[i]} & {sp.latex(selected_gamma[i])} \\\\ \\hline "
table_latex += r"\end{array}"
st.latex(table_latex)


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
w = w_long + w_short

# Display the weights
st.write(r"""
         The weights of the portfolio are:
         """)

weights_latex = r"\begin{bmatrix} "
for i in range(N):
    weights_latex += f"{sp.latex(w[i])} & "
weights_latex = weights_latex[:-2] + r" \end{bmatrix}"

st.latex(r"""
w_c^T = """ + weights_latex)

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
portfolio_return_with_g = w.dot(beta * (f + lambda_) + gamma * g + epsilon)

st.write(r"""
         We can now compute the return of the portfolio $c$:
                """)

st.latex(r"""
\begin{equation}
         \begin{aligned}
         r_c = w^\top r \\
         = w^\top (\beta (f + \lambda) + \gamma g +  \epsilon)
         \end{aligned}
\end{equation}
         """)
st.latex(f"""r_c = {sp.latex(portfolio_return_with_g)}""")

# Step 2: Take the expectation using sympy.stats
# expected_portfolio_return_with_g = stats.E(portfolio_return_with_g)
expected_portfolio_return_with_g = w.dot(beta) * lambda_

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
# total_portfolio_variance = variance_f + variance_g + variance_epsilon

covariance_matrix_f = beta * beta.T * sigma_f**2

covariance_matrix_g = gamma * gamma.T * sigma_g**2

# Define the covariance matrix for idiosyncratic errors (N x N identity matrix scaled by sigma_epsilon**2)
covariance_matrix_epsilon = sigma_epsilon**2 * sp.eye(N)

# Combine the covariance matrices
covariance_matrix = covariance_matrix_f + covariance_matrix_g +  covariance_matrix_epsilon

# Calculate the total portfolio variance as w.T * covariance_matrix * w
total_portfolio_variance = (w.T * covariance_matrix * w)[0]

# Calculate the Sharpe ratio
sharpe_ratio_with_g = expected_portfolio_return_with_g / sp.sqrt(total_portfolio_variance)

gamma_portfpoliio = gamma.dot(w)
st.latex(f"\\gamma_c = {sp.latex(gamma_portfpoliio)}")
st.write(r"""
         
The portfolio returns captures the expected returns 
because it loads on the rewarded factor $f$, but it also loads 
on the unrewarded factor $g$.
Indeed, in our example there exists a cross-sectional correlation
between the characteristics and the loadings on the unrewarded factor.
Most assets with positive loadings on the rewarded factor have positive loadings on the unrewarded factor.
The expected return of the portfolio is:
""")


st.latex(r"""
\begin{equation}
            \mathbb{E}[r_c] = w^\top \mu = w^\top \beta \lambda
\end{equation}
         """)

st.latex(f"E[r_c] = {sp.latex(expected_portfolio_return_with_g)}")


st.write(r"""
         The variance of the portfolio is:
                """)


st.latex(r"""
\begin{equation}
         \sigma_c^2 = w^\top \Sigma w = w^\top \left( \beta \beta^\top \sigma_f^2 + \gamma \gamma^\top \sigma_g^2 + \sigma_\epsilon^2 I \right) w
\end{equation}
         """)

st.latex(f"\\sigma^2_c = {sp.latex(total_portfolio_variance)}")

st.write(r"""
         which give us the Sharpe ratio of the portfolio:
                """)

st.latex(f"\\text{{Sharpe Ratio}} = {sp.latex(sharpe_ratio_with_g)}")

st.write(r"""
         The portfolio $c$ is not efficient because it loads on the unrewarded factor $g$.
Loading on the unrewarded factor $g$ is a source of risk (additional variance in the denominator of the Sharpe ratio)
that is not rewarded by the market (no risk premium $\lambda_g$ on $g$, and therefore no supplementary expected return in the numerator of the Sharpe ratio).
"""
         )


# Define variance of the portfolio due to g
variance_g = (w.dot(gamma))**2 * sigma_g**2  # Contribution from unpriced factor g

# Total portfolio variance, including contributions from f, g, and epsilon
variance_f = (w.dot(beta))**2 * sigma_f**2  # Contribution from priced factor f
variance_epsilon = w.dot(w) * sigma_epsilon**2  # Contribution from idiosyncratic errors

# Total variance of the portfolio:
total_portfolio_variance_with_g = variance_f + variance_g + variance_epsilon

# Compute R^2, which is the proportion of portfolio variance explained by the unrewarded factor g
# r_squared = variance_g / total_portfolio_variance_with_g
r_squared = (w.T * covariance_matrix_g * w)[0] / (w.T * covariance_matrix * w)[0]
# Display R^2
st.write(r"""
         The proportion of portfolio variance explained by the unrewarded risk factor $g$ (R²) is:
                """)

st.latex(r"""
\begin{equation}
         R^2 = \frac{w^\top (\gamma \gamma^\top \sigma_g^2) w}{w^\top \left( \beta \beta^\top \sigma_f^2 + \gamma \gamma^\top \sigma_g^2 + \sigma_\epsilon^2 I \right) w}
\end{equation}
         """)

st.latex(f"R^2 = {sp.latex(r_squared.simplify())}")



# Sidebar: Choose the volatility of the unrewarded risk
sigma_g_val = st.sidebar.slider("Select the volatility of the unrewarded risk factor (σg)", 0.1, 5.0, 1.0)
sigma_f_val = 1.0  # Default to 1

st.latex(f"\\sigma_g = {sigma_g_val}")

r_squared_eval = r_squared.subs({sigma_g: sigma_g_val, sigma_f: sigma_f_val, sigma_epsilon: 1})
st.latex(f"R^2 = {sp.latex(round(r_squared_eval.simplify(), 2))}")


# Further explanation
st.write(r"""
This $R^2$ shows how much of the portfolio's total variance can be attributed to the variance in the unrewarded risk factor $g$. 
The influence of $g$ depends on its volatility, ($\sigma_g^2$), and the cross-correlation between the rewarded and unrewarded risks.
         """)


st.write(r"""

Issue: 
- Exposure to $g$ increase the variance of the portfolio, but does not increase the expected return of the portfolio.

Source of the issue:
- The higher the cross-correlation between loadings on the rewarded factor and loadings on the unrewarded factor, the higher the loading of the portfolio to the unrewarded factor $\gamma_c$.

Magnitude of the issue: 
- the higher $\gamma_c$, the higher the variance explained by the unrewarded factor ($R^2$).
- the higher the volatility of the unrewarded factor, $\sigma^2_g$, the higher the variance explained by the unrewarded factor ($R^2$).
""")


st.subheader('Climate Risks Factor as $g$')


st.write(r"""
We have seen that the BMG factor is not a rewarded risk factor. May it acts as an unrewarded risk factor as $g$?
To investigate this, let's first see if any of the traditional Fama-French factors loads on the BMG factor.
What we are going to do is to run a regression of the BMG factor on the Fama-French factors.
We want to see if $\gamma_g$ is significantly different for any of the Fama-French factors.
         """)

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

bmg = pd.read_excel('data/carbon_risk_factor_updated.xlsx', sheet_name='daily', index_col=0)
ff3_data = load_ff3_data()
data = pd.merge(ff3_data, bmg, on='date')

# st.write(ff3_data.head())

@st.cache_data
def compute_rolling_beta(f, g, rolling_window=126):
    """Compute rolling beta of HML on Money Industry portfolio."""
    beta_values = [np.nan] * (rolling_window - 1)  # Prepend NaN for the first observations
    for i in range(len(f) - rolling_window + 1):
        y = f.iloc[i:i + rolling_window]
        X = sm.add_constant(g.iloc[i:i + rolling_window])
        model = sm.OLS(y, X).fit()
        beta_values.append(model.params[1])  # The slope (beta) coefficient
    return np.array(beta_values)

# Assuming 'data' is your DataFrame with columns ['date', 'hml', 'BMG']
gamma_values = (data 
               .get(["date","smb","hml"])
               .assign(
                   gamma_hml = compute_rolling_beta(data['hml'], data['BMG']),
                     gamma_smb = compute_rolling_beta(data['smb'], data['BMG'])
            )
            .dropna()
)

# Create a long-form DataFrame to plot both gamma_hml and gamma_smb
gamma_long = pd.melt(gamma_values, id_vars=['date'], value_vars=['gamma_hml', 'gamma_smb'],
                     var_name='Factor', value_name='Gamma')

# Plot both gamma_hml and gamma_smb
plot_values = (
    ggplot(gamma_long, aes(x='date', y='Gamma', color='Factor')) +   
    geom_line() +
    labs(title="Rolling Gamma of HML and SMB on BMG Factor",
         x="", y="$\gamma$") +
    scale_x_datetime(breaks=date_breaks('1 year'), labels=date_format('%Y')) +
    theme(axis_text_x=element_text(rotation=45, hjust=1))
)

# Display the plot in Streamlit
st.pyplot(ggplot.draw(plot_values))

st.write(r"""
We are now going to investigate the magnitude of it, such as the $\sigma_g^2$.
         """)

@st.cache_data
def compute_rolling_volatility(returns, rolling_window=126):
    """Compute rolling annualized volatility of a portfolio."""
    # Compute rolling standard deviation of daily returns
    rolling_volatility = returns.rolling(window=rolling_window).std()
    
    # Annualize the volatility: Multiply by the square root of 252 (trading days per year)
    annualized_volatility = rolling_volatility * np.sqrt(252)
    
    return annualized_volatility

# Select the "Money Industry" portfolio (replace 'money' with actual column name)
bmg_returns = data['BMG']  # Replace 'money' with the actual column name

# Compute rolling annualized volatility (252-day window)
annualized_volatility = compute_rolling_volatility(bmg_returns)

# Prepare data for plotting
data_volatility = data.copy()
data_volatility['annualized_volatility'] = annualized_volatility

# Create the plot for annualized volatility
plot_volatility = (ggplot(data_volatility, aes(x='date', y='annualized_volatility')) +
                    geom_line(color='green') +
                    labs(title="126-Day Rolling Annualized Volatility: BMG",
                        x="Date", y="Annualized Volatility") +
                    scale_x_datetime(breaks=date_breaks('1 year'), labels=date_format('%Y')) +
                    theme(axis_text_x=element_text(rotation=45, hjust=1)))

st.pyplot(ggplot.draw(plot_volatility))

st.write(r"""
         Results in the $R^2$.
            """)

@st.cache_data
def compute_rolling_r2(f, g, rolling_window=126):
    """Compute rolling R-squared values from regression of HML on industry portfolios."""
    r2_values = [np.nan] * (rolling_window - 1)  # Prepend NaN for the first observations
    for i in range(len(f) - rolling_window + 1):
        y = f.iloc[i:i + rolling_window]
        X = g.iloc[i:i + rolling_window]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        r2_values.append(model.rsquared)
    return np.array(r2_values)


# Assuming 'data' is your DataFrame with columns ['date', 'hml', 'BMG']
r_values = (data 
               .get(["date","smb","hml"])
               .assign(
                   r_hml = compute_rolling_r2(data['hml'], data['BMG']),
                     r_smb = compute_rolling_r2(data['smb'], data['BMG'])
            )
            .dropna()
)

# Create a long-form DataFrame to plot both gamma_hml and gamma_smb
r_values_long = pd.melt(r_values, id_vars=['date'], value_vars=['r_hml', 'r_smb'],
                     var_name='Factor', value_name='R_squared')

# Plot both gamma_hml and gamma_smb
plot_r_values = (
    ggplot(r_values_long, aes(x='date', y='R_squared', color='Factor')) +   
    geom_line() +
    labs(title="Rolling $R^2$ of HML and SMB on BMG Factor",
         x="", y="$R^2$") +
    scale_x_datetime(breaks=date_breaks('1 year'), labels=date_format('%Y')) +
    theme(axis_text_x=element_text(rotation=45, hjust=1))
)

# Display the plot in Streamlit
st.pyplot(ggplot.draw(plot_r_values))

# st.write(r"""
#          First plot show the $R^2$ from 126-days rolling regressions of daily HML returns on the twelve daily Fama and French (1997) value-weighted industry excess returns.
#          The plot shows not only short periods where the realized $R^2$ dips below 50%, but also several periods during which it exceeds 90%. While the $R^2$ fluctuates considerably,
#          the average is well above 70%.
#          """)


# # Display the code snippet
# code_snippet = '''
# import pandas as pd
# import numpy as np
# import pandas_datareader as pdr
# import statsmodels.api as sm
# from plotnine import *
# from mizani.formatters import date_format
# from mizani.breaks import date_breaks

# start_date = '2000-01-01'
# end_date = '2019-06-30'

# factors_ff3_daily_raw = pdr.DataReader(
#   name="F-F_Research_Data_Factors_daily",
#   data_source="famafrench", 
#   start=start_date, 
#   end=end_date)[0]

# factors_ff3_daily = (factors_ff3_daily_raw
#   .divide(100)
#   .reset_index(names="date")
#   .rename(str.lower, axis="columns")
#   .rename(columns={"mkt-rf": "mkt_excess"})
# )

# industries_ff_daily_raw = pdr.DataReader(
#   name="10_Industry_Portfolios_daily",
#   data_source="famafrench", 
#   start=start_date, 
#   end=end_date)[0]

# industries_ff_daily = (industries_ff_daily_raw
#   .divide(100)
#   .reset_index(names="date")
#   .assign(date=lambda x: pd.to_datetime(x["date"].astype(str)))
#   .rename(str.lower, axis="columns")
# )

# data = pd.merge(factors_ff3_daily[['date', 'hml']], industries_ff_daily, on='date')

# rolling_window = 126  # 126 days

# def rolling_r2(hml, industry_portfolios):
#     r2_values = []
    
#     for i in range(len(hml) - rolling_window + 1):
#         y = hml.iloc[i:i + rolling_window]
#         X = industry_portfolios.iloc[i:i + rolling_window]
#         X = sm.add_constant(X)
#         model = sm.OLS(y, X).fit()
#         r2_values.append(model.rsquared)
    
#     return np.array(r2_values)

# r2_values = rolling_r2(data['hml'], data.drop(columns=['date', 'hml']))

# data_rolling_r2 = data.iloc[rolling_window - 1:].copy()
# data_rolling_r2['r2'] = r2_values

# plot = (ggplot(data_rolling_r2, aes(x='date', y='r2')) +
#         geom_line(color='blue') +
#         labs(title="126-Day Rolling R^2: HML on 12 Industry Portfolios",
#              x="Date", y="R-squared") +
#         scale_x_datetime(breaks=date_breaks('5 years'), labels=date_format('%Y')) +
#         theme(axis_text_x=element_text(rotation=45, hjust=1)))

# plot.show()
# '''

# # Display the code snippet
# st.code(code_snippet, language='python')

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

# @st.cache_data
# def load_industry_data():
#     """Load the Fama-French industry portfolios data."""
#     start_date = '2000-01-01'
#     end_date = '2019-06-30'
#     industries_ff_daily_raw = pdr.DataReader(
#         name="12_Industry_Portfolios_daily",
#         data_source="famafrench",
#         start=start_date,
#         end=end_date)[0]
#     industries_ff_daily = (industries_ff_daily_raw
#                            .divide(100)
#                            .reset_index(names="date")
#                            .assign(date=lambda x: pd.to_datetime(x["date"].astype(str)))
#                            .rename(str.lower, axis="columns"))
#     return industries_ff_daily

# @st.cache_data
# def compute_rolling_r2(hml, industry_portfolios, rolling_window=126):
#     """Compute rolling R-squared values from regression of HML on industry portfolios."""
#     r2_values = []
#     for i in range(len(hml) - rolling_window + 1):
#         y = hml.iloc[i:i + rolling_window]
#         X = industry_portfolios.iloc[i:i + rolling_window]
#         X = sm.add_constant(X)
#         model = sm.OLS(y, X).fit()
#         r2_values.append(model.rsquared)
#     return np.array(r2_values)

# @st.cache_data
# def compute_rolling_beta(hml, money_industry, rolling_window=126):
#     """Compute rolling beta of HML on Money Industry portfolio."""
#     beta_values = []
#     for i in range(len(hml) - rolling_window + 1):
#         y = hml.iloc[i:i + rolling_window]
#         X = sm.add_constant(money_industry.iloc[i:i + rolling_window])
#         model = sm.OLS(y, X).fit()
#         beta_values.append(model.params[1])  # The slope (beta) coefficient
#     return np.array(beta_values)

# @st.cache_data
# def compute_rolling_volatility(returns, rolling_window=126):
#     """Compute rolling annualized volatility of a portfolio."""
#     # Compute rolling standard deviation of daily returns
#     rolling_volatility = returns.rolling(window=rolling_window).std()
    
#     # Annualize the volatility: Multiply by the square root of 252 (trading days per year)
#     annualized_volatility = rolling_volatility * np.sqrt(252)
    
#     return annualized_volatility

# # Main logic
# factors_ff3_daily = load_ff3_data()
# industries_ff_daily = load_industry_data()

# # Merge HML and industry portfolios
# data = pd.merge(factors_ff3_daily[['date', 'hml']], industries_ff_daily, on='date')

# # Run rolling regression and compute R-squared values
# if st.button('Run Rolling Regression and Plot'):
#     r2_values = compute_rolling_r2(data['hml'], data.drop(columns=['date', 'hml']))
    
#     # Add the R-squared values to the data
#     data_rolling_r2 = data.iloc[126 - 1:].copy()
#     data_rolling_r2['r2'] = r2_values

#     # Create the plot
#     plot = (ggplot(data_rolling_r2, aes(x='date', y='r2')) +
#             geom_line(color='blue') +
#             labs(title="126-Day Rolling $R^2$: HML on 12 Industry Portfolios",
#                  x="Date", y="R-squared") +
#             scale_x_datetime(breaks=date_breaks('5 years'), labels=date_format('%Y')) +
#             theme(axis_text_x=element_text(rotation=45, hjust=1)))

#     st.pyplot(ggplot.draw(plot))


# st.write(r"""
# The Money industry is a striking example of the large industry effect on HML. The figure below shows that the regression coefficient associated with Money increased
#          dramatically between 2007 and 2009. 
#          """)

# # Add a button to compute rolling beta and plot it
# if st.button('Run Rolling Beta Regression and Plot'):
#     # Select the "Money Industry" portfolio (replace 'money_industry' with actual column name)
#     money_industry = data['money']  # Make sure to replace 'money' with the actual column name in your dataset

#     # Compute rolling beta
#     beta_values = compute_rolling_beta(data['hml'], money_industry)

#     # Prepare data for plotting
#     data_rolling_beta = data.iloc[126 - 1:].copy()
#     data_rolling_beta['beta'] = beta_values

#     # Create the plot for beta
#     plot_beta = (ggplot(data_rolling_beta, aes(x='date', y='beta')) +
#                  geom_line(color='red') +
#                  labs(title="126-Day Rolling Beta: HML on Money Industry Portfolio",
#                       x="Date", y="Beta") +
#                  scale_x_datetime(breaks=date_breaks('5 years'), labels=date_format('%Y')) +
#                  theme(axis_text_x=element_text(rotation=45, hjust=1)))

#     st.pyplot(ggplot.draw(plot_beta))


# st.write(r"""
#          As shown as the Figure below, the volatility of returns also increased dramatically during the financial crisis.""")

# # Add a button to compute and plot annualized volatility of the Money portfolio
# if st.button('Run Annualized Volatility Calculation and Plot'):
#     # Select the "Money Industry" portfolio (replace 'money' with actual column name)
#     money_industry_returns = data['money']  # Replace 'money' with the actual column name

#     # Compute rolling annualized volatility (252-day window)
#     annualized_volatility = compute_rolling_volatility(money_industry_returns)

#     # Prepare data for plotting
#     data_volatility = data.copy()
#     data_volatility['annualized_volatility'] = annualized_volatility

#     # Create the plot for annualized volatility
#     plot_volatility = (ggplot(data_volatility, aes(x='date', y='annualized_volatility')) +
#                        geom_line(color='green') +
#                        labs(title="126-Day Rolling Annualized Volatility: Money Industry Portfolio",
#                             x="Date", y="Annualized Volatility") +
#                        scale_x_datetime(breaks=date_breaks('5 years'), labels=date_format('%Y')) +
#                        theme(axis_text_x=element_text(rotation=45, hjust=1)))

#     st.pyplot(ggplot.draw(plot_volatility))

# st.write(r"""
# As a result of increasing loading into the Money portfolio and higher volatility of the Money portfolio, Money explained a substantial amount of the variation of HML returns during those years.
#          """)

st.subheader("Conclusion")

st.write(r"""
We have seen that unrewarded risks matter.
In fact, asset pricing theory suggest that one of the main challenge 
in finance is the efficient diversification of unrewarded risks, 
where "diversification" means "reduction" or "cancellation" (as in "diversify away")
and "unrewarded" means "not compensated by a risk premium".
Indeed, unrewarded risks are by definition not attractive for investors 
who are inherently risk-averse and therefore only willing to take 
risks if there is an associated reward to be expected in exchange for such 
risk-taking, as shown by Markowitz (1952). (Amenc $\textit {et al.}$, 2014)
         """)