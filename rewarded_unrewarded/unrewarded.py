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


st.write(r"""
         The weights of the portfolio are:
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

st.latex(f"""r_c = {sp.latex(portfolio_return_with_g)}""")

# Step 2: Take the expectation using sympy.stats
expected_portfolio_return_with_g = stats.E(portfolio_return_with_g)

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
"""
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
The correlation between $\beta$ and $\gamma$ is:
""")
st.latex(r"\rho(\beta, \gamma) = " + sp.latex(correlation.simplify()))

st.subheader("Illustration with Industry")

st.write(r"""
         Asness $\textit{et al.}$ (2000) have shown that if book-to-market ratios (the characteristic used to build the value factor) 
         are decomposed into an across-industry component and a within-industry comopnent, then only the within-industry component is related 
            to expected returns. This literature then suggests that the exposure of HML to industry returns is unpriced, that is, that industry is one 
         unpriced source of common variation $g$. 

            """)


# Display the code snippet
code_snippet = '''
import pandas as pd
import numpy as np
import pandas_datareader as pdr
import statsmodels.api as sm
from plotnine import *
from mizani.formatters import date_format
from mizani.breaks import date_breaks

start_date = '1963-07-01'
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
  .rename(columns={"mkt-rf": "mkt_excess"})
)

industries_ff_daily_raw = pdr.DataReader(
  name="10_Industry_Portfolios_daily",
  data_source="famafrench", 
  start=start_date, 
  end=end_date)[0]

industries_ff_daily = (industries_ff_daily_raw
  .divide(100)
  .reset_index(names="date")
  .assign(date=lambda x: pd.to_datetime(x["date"].astype(str)))
  .rename(str.lower, axis="columns")
)

data = pd.merge(factors_ff3_daily[['date', 'hml']], industries_ff_daily, on='date')

rolling_window = 126  # 126 days

def rolling_r2(hml, industry_portfolios):
    r2_values = []
    
    for i in range(len(hml) - rolling_window + 1):
        y = hml.iloc[i:i + rolling_window]
        X = industry_portfolios.iloc[i:i + rolling_window]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        r2_values.append(model.rsquared)
    
    return np.array(r2_values)

r2_values = rolling_r2(data['hml'], data.drop(columns=['date', 'hml']))

data_rolling_r2 = data.iloc[rolling_window - 1:].copy()
data_rolling_r2['r2'] = r2_values

plot = (ggplot(data_rolling_r2, aes(x='date', y='r2')) +
        geom_line(color='blue') +
        labs(title="126-Day Rolling R^2: HML on 12 Industry Portfolios",
             x="Date", y="R-squared") +
        scale_x_datetime(breaks=date_breaks('5 years'), labels=date_format('%Y')) +
        theme(axis_text_x=element_text(rotation=45, hjust=1)))

plot.show()
'''

# Display the code snippet
st.code(code_snippet, language='python')

# Define a function for rolling regression and plotting
def run_rolling_regression_and_plot():
    start_date = '1963-07-01'
    end_date = '2019-06-30'

    # Load Fama-French factors data
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

    # Load 10 Industry Portfolios data
    industries_ff_daily_raw = pdr.DataReader(
        name="10_Industry_Portfolios_daily",
        data_source="famafrench",
        start=start_date,
        end=end_date)[0]

    industries_ff_daily = (industries_ff_daily_raw
                           .divide(100)
                           .reset_index(names="date")
                           .assign(date=lambda x: pd.to_datetime(x["date"].astype(str)))
                           .rename(str.lower, axis="columns"))

    # Merge HML and Industry portfolios
    data = pd.merge(factors_ff3_daily[['date', 'hml']], industries_ff_daily, on='date')

    # Define the window for rolling regressions
    rolling_window = 126  # 126 days

    # Function to compute R^2 from regression of HML on industry portfolios
    def rolling_r2(hml, industry_portfolios):
        r2_values = []

        for i in range(len(hml) - rolling_window + 1):
            y = hml.iloc[i:i + rolling_window]
            X = industry_portfolios.iloc[i:i + rolling_window]
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            r2_values.append(model.rsquared)

        return np.array(r2_values)

    # Perform the rolling regression and get R^2 values
    r2_values = rolling_r2(data['hml'], data.drop(columns=['date', 'hml']))
    
    # Add the R-squared values to the data
    data_rolling_r2 = data.iloc[rolling_window - 1:].copy()
    data_rolling_r2['r2'] = r2_values

    # Create the plot with plotnine (ggplot style)
    plot = (ggplot(data_rolling_r2, aes(x='date', y='r2')) +
            geom_line(color='blue') +
            labs(title="126-Day Rolling $R^2$: HML on 12 Industry Portfolios",
                 x="Date", y="R-squared") +
            scale_x_datetime(breaks=date_breaks('5 years'), labels=date_format('%Y')) +
            theme(axis_text_x=element_text(rotation=45, hjust=1)))

    # Show the plot in Streamlit
    st.pyplot(ggplot.draw(plot))

# Add a button to trigger the function
if st.button('Run Rolling Regression and Plot'):
    run_rolling_regression_and_plot()

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