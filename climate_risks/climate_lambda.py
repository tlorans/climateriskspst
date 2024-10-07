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

st.title('Are Climate Risks Rewarded or Unrewarded?')

st.write("""

The climate finance literature employ characteristic-sorted portfolio to investigate the relationship between climate risks and cross-sectional stock returns.
Reasoning in terms of portfolio,
         the fundamental debate in the literature is to understand whether climate change risk dynamics 
            are rewarded or unrewarded in the cross-section of stock returns.
         """)


st.write(r"""
Some studies found the evidence of a carbon premium in the cross-section.
         Bolton and Kacperczyk (2021) found that portfolios sorted on the total level and the 
         year-by-year change in emissions were valued at discount. These facts 
         are true regardless of the type of scope emission analysed. Bolton and Kacperczyk (2021)
         further showed that the carbon premium is not linked ot the emission intensity measure.
         """)


# Data for the table
data = {
    "Study": [
        "Bolton and Kacperczyk (2021a)", 
        "Bolton and Kacperczyk (2020)", "Hsu et al. (2020)", 
        "Ardia et al. (2021)", "Cheema-Fox et al. (2021)", "Görgen et al. (2020)",
        "In et al. (2017)", "Pastor et al. (2021)"
    ],
    "Climate risk measure": [
        "Three emission measures", 
        "Three emission measures", "Emission intensity", 
         "Emission intensity", "Two emission measures", 
        "BG scores", 
        "Emission intensity", "E-scores"
    ],
    "Economic rationale": [
        "Transition risk proxies", 
        "Transition risk proxies","Climate policy risk", "Pastor et al. (2020)", "Investor irrationality", 
        "Transition risk proxies", 
        "Investor irrationality", "Pastor et al. (2020)"
    ],

    "Is climate risk priced?": [
        "Yes", "Yes",  
        "Yes", "No", "No", "No", 
        "No",  "No"
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Display the table using Streamlit
st.subheader("Table 1: Climate risk factors and the cross-section of stock returns")
st.write("This table summarizes studies on transition risks factor and their conclusions on the pricing of climate risks.")
st.dataframe(df)

st.write(r"""
         The findings of Bolton and Kacperczyk (2021) are echoed by 
         Bolton and Kacperzcyk (2020) at the international level, although not all countries 
         are pricing cabron risk to the same extent. 
         On the other hand, Hsu $\textit{et al.} (2020)$ discovered that 
         the carbon premium is related to the emission intensity measure.
            """)

st.write(r"""
                  Nevertheless, these findings were not corroborated by the remaining studies in the Table.
""")

st.subheader("_Mean_ Factor (Rewarded Risk)...")


st.write(r"""
Is there a spread in average returns between green and brown firms?
Test:
         """)

st.latex(r"""
         \begin{equation}
\hat{\lambda} = \frac{1}{T} \sum_{t=1}^{T} BMG_t = \bar{BMG}_t
         \end{equation}
         """)         

st.write(r"""
Is this spread explained by exposure to other known risk factors?
Test:
         """)

st.latex(r"""
\begin{equation}
         \alpha_{\text{BMG}} = 0
         \end{equation}
         """)

st.write(r"""
Or, alternatively, does it help to explain average returns ie. change alpha in the model 
         of average (expected) returns?
         """)

st.latex(r"""
         \begin{equation}
         \begin{aligned}
         \mathbb{E}[R_{i,t}] = \alpha + \beta_1 \mathbb{E}[R_{m,t}] + \beta_2 \mathbb{E}[SMB_t] + \\
         \beta_3 \mathbb{E}[HML_t] + \beta_4 \mathbb{E}[RMW_t] + \beta_5 \mathbb{E}[CMA_t] + \beta_6 \mathbb{E}[BMG_t]
         \end{aligned}
         \end{equation}
         """)

st.write(r"""
This is where we stop with BMG factor. Not rewarded risk (ie. not a mean factor ie. not associated with higher expected returns once controlled for known factors).
         """)

bmg = pd.read_excel('data/carbon_risk_factor_updated.xlsx', sheet_name='daily', index_col=0)

if st.button("Cumulative Returns"):
    bmg_cum = (1 + bmg).cumprod() - 1
    plot = (ggplot(bmg_cum.reset_index(), aes(x='date', y='BMG')) +
        geom_line() + 
        scale_x_date(breaks=date_breaks('1 years'), labels=date_format('%Y')) + 
        labs(title="Cumulative Returns of BMG Portfolio", x="Date", y="Cumulative Return") +
        theme(axis_text_x=element_text(rotation=45, hjust=1))
        )
    

    st.pyplot(ggplot.draw(plot))


st.subheader("...or _Variance_ Factor? (Unrewarded Risk)")

st.write(r"""
BMG may not helps in explain average returns but it may help to explain return variance - it can help to increase
         $R^2$ in:
         """)

st.latex(r"""
            \begin{equation}
         R_{i,t} = \alpha + \beta_1 R_{m,t} + \beta_2 SMB_t + \beta_3 HML_t + \beta_4 RMW_t + \beta_5 CMA_t + \beta_6 BMG_t + \epsilon_{i,t}
            \end{equation}
            """)

st.write(r"""
In that case, source of common variation in returns (as $g$). Associated variance 
         but no additional expected return.
         You want to hedge out from your portfolio.
            """)


st.subheader("Value and BMG Portfolio")

st.write(r"""
         Is climate risk an unrewarded factor that may 
         decrease the mean-variance efficiency of investors' portfolios?
            """)

st.write(r"""Use the same recipe than in our unrewarded risk section:
        - Volatility of the BMG portfolio?
        - Correlation between Value and BMG portfolio?
        - Resulting variance of Value explained by BMG?
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
