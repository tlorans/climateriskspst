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


st.title('Hedging from Climate Risks')

st.write(r"""
We have seen that, as of today, no risk premium seems to be associated with climate risks. It is therefore to be treated as unrewarded risk. 
We have seen that if exposure to the rewarded risk is correlated with exposure to unrewarded risk, a portfolio rightly exposed to rewarded risk is mean-variance inefficient because
it loads in the unrewarded risk. We can improve upon this. We want a method that (1) let untouched the expected return and (2) reduce the variance of our portfolio.
         """)



st.subheader('Hedging Portfolio')

st.write(r"""Our final objective is to obtain a portfolio that is orthogonalized (or beta-hedged) from climate risks. 
         To do so, we first isolate the part of the portfolio that is exposed to climate risks.
         """
   )

st.latex(r"""
\begin{equation}
         R_t = \alpha + \beta BMG_t + \epsilon_t 
\end{equation}
         """)

st.latex(r"""
H_t = \beta BMG_t
         """)

st.write(r"""
         Because $BMG_t$ has zero expected returns, the expected returns of the hedging portfolio $H_t$ are zero.
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
               .get(["date","hml","BMG"])
               .assign(
                   gamma = compute_rolling_beta(data['hml'], data['BMG']),
            )
            .dropna()
)

gamma_values['hedging portfolio'] = gamma_values['gamma'] * gamma_values['BMG']

cumulative_returns = (1 + gamma_values[['hml', 'BMG', 'hedging portfolio']]).cumprod() - 1

# Reset index to use 'date' in plotting
cumulative_returns = cumulative_returns.reset_index()

# Convert 'date' to datetime for better plotting
cumulative_returns['date'] = pd.to_datetime(gamma_values['date'])

# Transform the cumulative_returns DataFrame to long format
cumulative_returns_long = pd.melt(cumulative_returns, id_vars=['date'], 
                                  value_vars=['hml', 'BMG', 'hedging portfolio'], 
                                  var_name='Portfolio', value_name='Cumulative Return')

# Create the cumulative returns plot using plotnine
plot = (ggplot(cumulative_returns_long, aes(x='date', y='Cumulative Return', color='Portfolio')) +
        geom_line() +
        labs(title='Cumulative Returns of HML, BMG, and Hedging Portfolio',
             x='Date', y='Cumulative Returns') +
        theme(axis_text_x=element_text(rotation=45, hjust=1)) +
        scale_x_datetime(breaks=date_breaks('1 year'), labels=date_format('%Y')))

# Display the plot in Streamlit
st.pyplot(ggplot.draw(plot))

st.subheader('Portfolio Hedged From Climate Risks')

st.write(r"""
We can now construct a portfolio that is orthogonalized from climate risks.
         """)

st.latex(r"""
\begin{equation}
         R^*_t = R_t - H_t
\end{equation}
         """)

gamma_values['hedged portfolio'] = gamma_values['hml'] - gamma_values['hedging portfolio']

cumulative_returns_hedged = (1 + gamma_values[['hml', 'hedged portfolio']]).cumprod() - 1

# Reset index to use 'date' in plotting
cumulative_returns_hedged = cumulative_returns_hedged.reset_index()

# Convert 'date' to datetime for better plotting
cumulative_returns_hedged['date'] = pd.to_datetime(gamma_values['date'])

# Transform the cumulative_returns DataFrame to long format
cumulative_returns_hedged_long = pd.melt(cumulative_returns_hedged, id_vars=['date'], 
                                  value_vars=['hml', 'hedged portfolio'], 
                                  var_name='Portfolio', value_name='Cumulative Return')

# Create the cumulative returns plot using plotnine
plot = (ggplot(cumulative_returns_hedged_long, aes(x='date', y='Cumulative Return', color='Portfolio')) +
        geom_line() +
        labs(title='Cumulative Returns of HML and Hedged Portfolio',
             x='Date', y='Cumulative Returns') +
        theme(axis_text_x=element_text(rotation=45, hjust=1)) +
        scale_x_datetime(breaks=date_breaks('1 year'), labels=date_format('%Y')))
st.pyplot(ggplot.draw(plot))

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
r_values = (gamma_values 
               .get(["date","hml", "hedged portfolio","BMG"])
               .assign(
                   r_hml = compute_rolling_r2(gamma_values['hml'], gamma_values['BMG']),
                     r_hedged = compute_rolling_r2(gamma_values['hedged portfolio'], gamma_values['BMG'])
            )
            .dropna()
)


# Create a long-form DataFrame to plot both gamma_hml and gamma_smb
r_values_long = pd.melt(r_values, id_vars=['date'], value_vars=['r_hml', 'r_hedged'],
                     var_name='Portfolio', value_name='R_squared')

# Plot both gamma_hml and gamma_smb
plot_r_values = (
    ggplot(r_values_long.query('date < "01-01-2019"'), aes(x='date', y='R_squared', color='Portfolio')) +   
    geom_line() +
    labs(title="Rolling $R^2$ of HML and Hedge Portfolio on BMG Factor",
         x="", y="$R^2$") +
    scale_x_datetime(breaks=date_breaks('1 year'), labels=date_format('%Y')) +
    theme(axis_text_x=element_text(rotation=45, hjust=1))
)

# Display the plot in Streamlit
st.pyplot(ggplot.draw(plot_r_values))


sharpe_ratio_hedged = r_values['hedged portfolio'].mean() / r_values['hedged portfolio'].std()
sharpe_ratio_hml = r_values['hml'].mean() / r_values['hml'].std()

st.write(f"""
The Sharpe ratio of the hedged portfolio is {sharpe_ratio_hedged:.2f}, while the Sharpe ratio of the HML portfolio is {sharpe_ratio_hml:.2f}.
""")