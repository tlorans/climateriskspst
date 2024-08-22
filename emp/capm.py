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

print(ret_monthly.head())
"""

st.code(code_snippet_1, language='python')

if st.button('Run Example 1'):
    output = run_code_and_capture_output(code_snippet_1)
    st.subheader("Output")
    st.code(output, language='text')

st.markdown("""
## Univariate Portfolio Sort
            """)