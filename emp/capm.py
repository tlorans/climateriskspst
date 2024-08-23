import streamlit as st
from pydantic import BaseModel, Field, conlist, field_validator 
from typing import List
import io
import contextlib
import pandas as pd

def run_code_and_capture_output(code: str) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        try:
            exec(code, globals(), locals())
        except Exception as e:
            print(e)
    return buffer.getvalue()

st.set_option('deprecation.showPyplotGlobalUse', False)


# Title of the web app
st.markdown("""
            # Testing the CAPM

            To check empirically the validity of the theory we have discussed, the first thing to test is the validity of the CAPM, which forms the basis of the theory.            
            As we have seen, cross-sectional variation in expected returns should be explained by the covariance between the excess return 
            of the asset and the excess return of the market portfolio. 
            The regression coefficient of this relationship is the market beta of the asset.
            In this part, we will first start by estimating the beta of the assets, and then apply portfolio univariate sort to test the CAPM.

            """)

st.markdown("""
## Estimating the CAPM Beta
            
Our first step is to estimate the exposure of an individual asset to changes in the market portfolio.
To estimate the beta, we will use the following regression model:
            """)

st.latex(r"""
\begin{equation}
         R_{i,t} - R_{f,t} = \alpha_i + \beta_i (R_{m,t} - R_{f,t}) + \epsilon_{i,t}
\end{equation}
         """)

st.write(r"""
where:
- $R_{i,t}$ is the return of asset $i$ at time $t$,
- $R_{f,t}$ is the risk-free rate at time $t$,
- $R_{m,t}$ is the return of the market portfolio at time $t$,
- $\alpha_i$ is the intercept of the regression,
- $\beta_i$ is the beta of asset $i$,
- $\epsilon_{i,t}$ is the error term
        """)

st.markdown(r"""            
### Data Preparation
            
We will use the S&P 500 constituents as our sample of assets. 
We will download the monthly returns of the first 5 constituents. 
In the code below, we download the data and calculate the monthly returns.
We then calculate the descriptive statistics of the returns.
            """)


code_snippet_1 = """
import pandas as pd 
import numpy as np
import yfinance as yf 

### Download S&P 500 constituents
# URL to the Wikipedia page
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Use pandas to read tables from the URL
sp500_constituents = (pd.read_html(url)[0]
                      .get("Symbol")
                      .tolist()
)


### Download stock prices
ret_monthly = (
    yf.download(
        tickers = sp500_constituents[:5],
        start = '2009-12-31',
        end = '2023-12-31',
        interval = '1mo',
        progress = True
    )
    .stack()
    .reset_index(level = 1, drop = False)
    .reset_index()
    .assign(R = lambda df: df.groupby("Ticker")["Adj Close"].pct_change())
    .dropna()
    .assign(Date = lambda x: pd.to_datetime(x['Date']) + pd.offsets.MonthEnd(0))
    .get(["Date", "Ticker", "R"])
)


table = (ret_monthly
       .groupby(['Ticker'])['R'].describe()
       .round(3)
)

"""

st.code(code_snippet_1, language='python')

if st.button('Run Example 1'):
    output = exec(code_snippet_1, globals())
    st.subheader("Output")
    st.dataframe(table)  # Assuming table is the result DataFrame


st.markdown(r"""
            We then download the Fama-French factors to use the market portfolio.
            We will use the excess return of the market portfolio as the independent variable in the regression.
            In the code below, we download the Fama-French factors and calculate the descriptive statistics 
            of the market excess return. We also plot the cumulative return of the market portfolio.
            """)

code_snippet_2 = """
import pandas_datareader as pdr 
factors_ff3_monthly = (pdr.DataReader(
                        name = 'F-F_Research_Data_5_Factors_2x3',
                        data_source = 'famafrench',
                        start = '2010-01-01',
                        end = '2023-12-31',
                    )[0]
                    .divide(100)
                    .reset_index(names = 'Date')
                    .assign(Date = lambda x: pd.to_datetime(x['Date'].astype(str)) + pd.offsets.MonthEnd(0))
                    .rename(columns = {'Mkt-RF':'Rm_minus_Rf',
                                       'RF':'Rf'})
                    .get(['Date', 'Rm_minus_Rf', 'Rf'])
)

output_df = (factors_ff3_monthly['Rm_minus_Rf']
      .groupby(factors_ff3_monthly['Date'].dt.year)
      .describe()
        .round(3)
)

from plotnine import *
from mizani.breaks import date_breaks
from mizani.formatters import date_format

plot_market = (
    ggplot(factors_ff3_monthly.assign(Cumulative_Market_Return = lambda x: (1 + x['Rm_minus_Rf']).cumprod()),
           aes(x = 'Date', y = 'Cumulative_Market_Return')) +
    geom_line() +
    scale_x_datetime(breaks = date_breaks('2 years'), labels = date_format('%Y')) +
    labs(x = "Date", y = "Market Return")
)
"""


st.code(code_snippet_2, language='python')

if st.button('Run Example 2'):
    exec(code_snippet_2, globals())  # Run the code snippet directly

    # Display the DataFrame using st.dataframe
    st.subheader("Output")
    st.dataframe(output_df)  # Assuming output_df is the result DataFrame
    st.pyplot(plot_market.show())  # Assuming plot_market is the result plot



st.markdown("""
            ### OLS Regression

            We will now estimate the beta of the assets using the OLS regression.
            This is a time-series regression where we regress the excess return 
            of the asset on the excess return of the market portfolio.
            """)



code_snippet_3 = """
import pandas as pd 
import numpy as np
import yfinance as yf 
import statsmodels.formula.api as smf
import pandas_datareader as pdr

### Download S&P 500 constituents
# URL to the Wikipedia page
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Use pandas to read tables from the URL
sp500_constituents = (pd.read_html(url)[0]
                      .get("Symbol")
                      .tolist()
)


### Download stock prices
ret_monthly = (
    yf.download(
        tickers = sp500_constituents[:5],
        start = '2009-12-31',
        end = '2023-12-31',
        interval = '1mo',
        progress = True
    )
    .stack()
    .reset_index(level = 1, drop = False)
    .reset_index()
    .assign(R = lambda df: df.groupby("Ticker")["Adj Close"].pct_change())
    .dropna()
    .assign(Date = lambda x: pd.to_datetime(x['Date']) + pd.offsets.MonthEnd(0))
    .get(["Date", "Ticker", "R"])
)

factors_ff3_monthly = (pdr.DataReader(
                        name = 'F-F_Research_Data_5_Factors_2x3',
                        data_source = 'famafrench',
                        start = '2010-01-01',
                        end = '2023-12-31',
                    )[0]
                    .divide(100)
                    .reset_index(names = 'Date')
                    .assign(Date = lambda x: pd.to_datetime(x['Date'].astype(str)) + pd.offsets.MonthEnd(0))
                    .rename(columns = {'Mkt-RF':'Rm_minus_Rf',
                                       'RF':'Rf'})
                    .get(['Date', 'Rm_minus_Rf', 'Rf'])
)

monthly_data = (
    ret_monthly
    .merge(factors_ff3_monthly, on = 'Date', how = 'left')
    .assign(
        R_minus_Rf = lambda x: x['R'] - x['Rf'],
    )
    .get(['Date', 'Ticker', 'R_minus_Rf', 'Rm_minus_Rf'])
)


model_beta = (
    smf.ols("R_minus_Rf ~ Rm_minus_Rf",
    data = monthly_data.query("Ticker == 'ABBV'")
    )
    .fit()
)
"""

st.code(code_snippet_3, language='python')


if st.button('Run Example 3'):
    exec(code_snippet_3, globals())  # Run the code snippet directly
    st.subheader("Output")
    st.write(model_beta.summary())  # Assuming table is the result DataFrame

st.markdown("""
            ### Rolling Window Estimation
            """)


st.markdown("""
## Univariate Portfolio Sort
            """)

st.markdown("""
### Sorting by Market Beta
            """)


st.markdown("""
### Portfolio Returns
            """)