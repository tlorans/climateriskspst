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

st.title('Climate Risks as $g$')

st.subheader('')

st.write(r"""
The empirical findings in Ardia $\textit{et al.}$ (2021) and 
Pastor $\textit{et al.}$ (2021) gives an interpretation of the mixed 
results in the literature. Also Ardia $\textit{et al.}$ (2021)
         and Pastor $\textit{et al.}$ (2021) found that a portfolio 
         that is long green stocks and short brown firms exhibits a positive alpha,
         controlling for other known risk factors in the asset pricing literature.
         They showed that such an outperformance is related to both the preference 
         change channels and thus, to both positive cash-flow and negative discount rate 
         news for green stocks. 

To further show that such an outperformance is due to unexpected revaluations during the 
estimation period, Pastor $\textit{et al.}$ (2021) constructed a counter-factual green 
factor assuming zero shocks to climate concerns. The striking result 
in Pastor $\textit{et al.}$ (2021) was that, in absence of climate news, the green 
factor performance would be essentially flat. 
         """)


st.write(r"""
         Value factor and green factor""")

st.write(r"""
Therefore, the empirical evidence in Ardia $\textit{et al.}$ (2021) and Pastor $\textit{et al.}$ (2021)
alongside the mixed results in chasing the climate risk premium in the literature,
suggest that climate risks are unrewarded in the cross-section of stock returns.
         
We've seen in the previous section that exposure to unrewareded risks can be
detrimental to the investor's portfolio, as it undermines the portfolio's mean-variance efficiency.
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

@st.cache_data
def compute_rolling_r2(hml, industry_portfolios, rolling_window=126):
    """Compute rolling R-squared values from regression of HML on industry portfolios."""
    r2_values = []
    for i in range(len(hml) - rolling_window + 1):
        y = hml.iloc[i:i + rolling_window]
        X = industry_portfolios.iloc[i:i + rolling_window]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        r2_values.append(model.rsquared)
    return np.array(r2_values)

@st.cache_data
def compute_rolling_beta(hml, money_industry, rolling_window=126):
    """Compute rolling beta of HML on Money Industry portfolio."""
    beta_values = []
    for i in range(len(hml) - rolling_window + 1):
        y = hml.iloc[i:i + rolling_window]
        X = sm.add_constant(money_industry.iloc[i:i + rolling_window])
        model = sm.OLS(y, X).fit()
        beta_values.append(model.params[1])  # The slope (beta) coefficient
    return np.array(beta_values)

@st.cache_data
def compute_rolling_volatility(returns, rolling_window=126):
    """Compute rolling annualized volatility of a portfolio."""
    # Compute rolling standard deviation of daily returns
    rolling_volatility = returns.rolling(window=rolling_window).std()
    
    # Annualize the volatility: Multiply by the square root of 252 (trading days per year)
    annualized_volatility = rolling_volatility * np.sqrt(252)
    
    return annualized_volatility


# Main logic
factors_ff3_daily = load_ff3_data()
bmg = pd.read_excel('data/carbon_risk_factor_updated.xlsx', sheet_name='daily', index_col=0)

# Merge HML and bmg
data = pd.merge(factors_ff3_daily[['date', 'hml']], bmg, on='date')


# Run rolling regression and compute R-squared values
if st.button('Run Rolling Regression and Plot'):
    r2_values = compute_rolling_r2(data['hml'], data.drop(columns=['date', 'hml']))
    
    # Add the R-squared values to the data
    data_rolling_r2 = data.iloc[126 - 1:].copy()
    data_rolling_r2['r2'] = r2_values

    # Create the plot
    plot = (ggplot(data_rolling_r2, aes(x='date', y='r2')) +
            geom_line(color='blue') +
            labs(title="126-Day Rolling $R^2$: HML on BMG Portfolio",
                 x="Date", y="R-squared") +
            scale_x_datetime(breaks=date_breaks('1 years'), labels=date_format('%Y')) +
            theme(axis_text_x=element_text(rotation=45, hjust=1)))

    st.pyplot(ggplot.draw(plot))


# # Add a button to compute rolling beta and plot it
# if st.button('Run Rolling Beta Regression and Plot'):
#     # Select the "Money Industry" portfolio (replace 'money_industry' with actual column name)
#     bmg = data['BMG']  # Make sure to replace 'money' with the actual column name in your dataset

#     # Compute rolling beta
#     beta_values = compute_rolling_beta(data['hml'], bmg)

#     # Prepare data for plotting
#     data_rolling_beta = data.iloc[126 - 1:].copy()
#     data_rolling_beta['beta'] = beta_values

#     # Create the plot for beta
#     plot_beta = (ggplot(data_rolling_beta, aes(x='date', y='beta')) +
#                  geom_line(color='red') +
#                  labs(title="126-Day Rolling Beta: HML on BMG Portfolio",
#                       x="Date", y="Beta") +
#                  scale_x_datetime(breaks=date_breaks('1 year'), labels=date_format('%Y')) +
#                  theme(axis_text_x=element_text(rotation=45, hjust=1)))

#     st.pyplot(ggplot.draw(plot_beta))


# # Add a button to compute and plot annualized volatility of the Money portfolio
# if st.button('Run Annualized Volatility Calculation and Plot'):
#     # Select the "Money Industry" portfolio (replace 'money' with actual column name)
#     bmg_returns = data['BMG']  # Replace 'money' with the actual column name

#     # Compute rolling annualized volatility (252-day window)
#     annualized_volatility = compute_rolling_volatility(bmg_returns)

#     # Prepare data for plotting
#     data_volatility = data.copy()
#     data_volatility['annualized_volatility'] = annualized_volatility

#     # Create the plot for annualized volatility
#     plot_volatility = (ggplot(data_volatility, aes(x='date', y='annualized_volatility')) +
#                        geom_line(color='green') +
#                        labs(title="126-Day Rolling Annualized Volatility: BMG Portfolio",
#                             x="Date", y="Annualized Volatility") +
#                        scale_x_datetime(breaks=date_breaks('1 year'), labels=date_format('%Y')) +
#                        theme(axis_text_x=element_text(rotation=45, hjust=1)))

#     st.pyplot(ggplot.draw(plot_volatility))