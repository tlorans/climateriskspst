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


@st.cache_data
def load_ff5_data():
    """Load the Fama-French factors data."""
    start_date = '2000-01-01'
    end_date = '2019-06-30'
    factors_ff3_daily_raw = pdr.DataReader(
        name="F-F_Research_Data_5_Factors_2x3_daily",
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
ff5_data = load_ff5_data()
data = pd.merge(ff5_data, bmg, on='date')
# Add a dropdown to let the user select the factor to hedge against BMG
factor_to_hedge = st.selectbox(
    'Select the factor you want to hedge from BMG:',
    ['hml', 'smb', 'rmw', 'cma', 'mkt_excess'],
    index=0  # Default selection
)

@st.cache_data
def compute_rolling_beta(f, g, rolling_window=126):
    """Compute rolling beta of the chosen factor on the BMG factor."""
    beta_values = [np.nan] * (rolling_window - 1)  # Prepend NaN for the first observations
    for i in range(len(f) - rolling_window + 1):
        y = f.iloc[i:i + rolling_window]
        X = sm.add_constant(g.iloc[i:i + rolling_window])
        model = sm.OLS(y, X).fit()
        beta_values.append(model.params[1])  # The slope (beta) coefficient
    return np.array(beta_values)

# Calculate rolling beta for the selected factor
gamma_values = (data
                .get(["date", factor_to_hedge, "BMG"])
                .assign(
                    gamma=compute_rolling_beta(data[factor_to_hedge], data['BMG']),
                )
                .dropna()
)

# Calculate hedging portfolio based on the rolling beta
gamma_values['hedging portfolio'] = gamma_values['gamma'] * gamma_values['BMG']

st.subheader('Portfolio Hedged From Climate Risks')

st.write(r"""
We can now construct a portfolio that is orthogonalized from climate risks.
         """)

st.latex(r"""
\begin{equation}
         R_t = \alpha + \beta BMG_t + \epsilon_t 
\end{equation}
         """)

st.latex(r"""
\begin{equation}
         R^*_t = R_t - \beta BMG_t
\end{equation}
         """)


# Create the hedged portfolio by subtracting the hedging portfolio from the selected factor
gamma_values['hedged portfolio'] = gamma_values[factor_to_hedge] - gamma_values['hedging portfolio']

# Calculate cumulative returns for both the original and hedged portfolios
cumulative_returns_hedged = (1 + gamma_values[[factor_to_hedge, 'hedged portfolio']]).cumprod() - 1

# Reset index to use 'date' in plotting
cumulative_returns_hedged = cumulative_returns_hedged.reset_index()

# Convert 'date' to datetime for better plotting
cumulative_returns_hedged['date'] = pd.to_datetime(gamma_values['date'])

# Transform the cumulative_returns DataFrame to long format for plotting
cumulative_returns_hedged_long = pd.melt(cumulative_returns_hedged, id_vars=['date'], 
                                         value_vars=[factor_to_hedge, 'hedged portfolio'], 
                                         var_name='Portfolio', value_name='Cumulative Return')

# Create the cumulative returns plot using plotnine
plot = (ggplot(cumulative_returns_hedged_long, aes(x='date', y='Cumulative Return', color='Portfolio')) +
        geom_line() +
        labs(title=f'Cumulative Returns of {factor_to_hedge.upper()} and Hedged Portfolio',
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


# Calculate rolling R-squared for both the selected factor and the hedged portfolio
r_values = (gamma_values 
               .get(["date", factor_to_hedge, "hedged portfolio", "BMG"])
               .assign(
                   r_factor = compute_rolling_r2(gamma_values[factor_to_hedge], gamma_values['BMG']),
                   r_hedged = compute_rolling_r2(gamma_values['hedged portfolio'], gamma_values['BMG'])
               )
               .dropna()
)

# Create a long-form DataFrame to plot both R-squared values
r_values_long = pd.melt(r_values, id_vars=['date'], value_vars=['r_factor', 'r_hedged'],
                     var_name='Portfolio', value_name='R_squared')

# Plot R-squared values
plot_r_values = (
    ggplot(r_values_long.query('date < "01-01-2019"'), aes(x='date', y='R_squared', color='Portfolio')) +   
    geom_line() +
    labs(title=f"Rolling $R^2$ of {factor_to_hedge.upper()} and Hedge Portfolio on BMG Factor",
         x="", y="$R^2$") +
    scale_x_datetime(breaks=date_breaks('1 year'), labels=date_format('%Y')) +
    theme(axis_text_x=element_text(rotation=45, hjust=1))
)

# Display the R-squared plot
st.pyplot(ggplot.draw(plot_r_values))

# Calculate Sharpe ratios
sharpe_ratio_hedged = r_values['hedged portfolio'].mean() / r_values['hedged portfolio'].std() * np.sqrt(252) 
sharpe_ratio_factor = r_values[factor_to_hedge].mean() / r_values[factor_to_hedge].std() * np.sqrt(252) 

st.write(f"""
The Sharpe ratio of the hedged portfolio is {sharpe_ratio_hedged:.2f}, while the Sharpe ratio of the {factor_to_hedge.upper()} portfolio is {sharpe_ratio_factor:.2f}.
""")


@st.cache_data
def compute_rolling_volatility(returns, rolling_window=126):
    """Compute rolling annualized volatility of a portfolio."""
    # Compute rolling standard deviation of daily returns
    rolling_volatility = returns.rolling(window=rolling_window).std()
    
    # Annualize the volatility: Multiply by the square root of 252 (trading days per year)
    annualized_volatility = rolling_volatility * np.sqrt(252)
    
    return annualized_volatility

# Compute rolling volatility for both portfolios (factor and hedged)
rolling_vol_factor = compute_rolling_volatility(gamma_values[factor_to_hedge])
rolling_vol_hedged = compute_rolling_volatility(gamma_values['hedged portfolio'])

# Create a DataFrame with both rolling volatilities
volatility_df = pd.DataFrame({
    'date': gamma_values['date'],
    f'Rolling Volatility {factor_to_hedge.upper()}': rolling_vol_factor,
    'Rolling Volatility Hedged': rolling_vol_hedged
}).dropna()

# Reset index for better plotting
volatility_df = volatility_df.reset_index(drop=True)

# Convert 'date' to datetime for better plotting
volatility_df['date'] = pd.to_datetime(volatility_df['date'])

# Transform the volatility DataFrame to long format for plotting
volatility_df_long = pd.melt(volatility_df, id_vars=['date'], 
                             value_vars=[f'Rolling Volatility {factor_to_hedge.upper()}', 'Rolling Volatility Hedged'], 
                             var_name='Portfolio', value_name='Annualized Volatility')

# Create the rolling volatility plot using plotnine
plot_volatility = (ggplot(volatility_df_long, aes(x='date', y='Annualized Volatility', color='Portfolio')) +
                   geom_line() +
                   labs(title=f'Rolling Annualized Volatility of {factor_to_hedge.upper()} and Hedged Portfolio',
                        x='Date', y='Annualized Volatility') +
                   theme(axis_text_x=element_text(rotation=45, hjust=1)) +
                   scale_x_datetime(breaks=date_breaks('1 year'), labels=date_format('%Y')))

# Display the volatility plot
st.pyplot(ggplot.draw(plot_volatility))


# get the loading of the hedged portfolio on the BMG factor

gamma_values['loading'] = compute_rolling_beta(gamma_values['hedged portfolio'], gamma_values['BMG'])

# melt 'loading' and 'gamma' columns in gamma_values to plot them
plot_values = pd.melt(gamma_values, id_vars=['date'], value_vars=['loading', 'gamma'],
                       var_name='Factor', value_name='Beta')


# Create the plot for the loading of the hedged portfolio on the BMG factor
plot_loading = (ggplot(plot_values.query('date < "01-01-2019"'), aes(x='date', y='Beta', color='Factor')) +
                geom_line() +
                labs(title="Rolling $\\beta$ of Hedged Portfolio on BMG Factor",
                     x="", y="$\\beta$") +
                scale_x_datetime(breaks=date_breaks('1 year'), labels=date_format('%Y')) +
                theme(axis_text_x=element_text(rotation=45, hjust=1))
)

# Display the plot
st.pyplot(ggplot.draw(plot_loading))