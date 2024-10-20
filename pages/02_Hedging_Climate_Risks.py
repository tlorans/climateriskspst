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

st.latex(r"""
\begin{equation}
         BMG_t = \alpha + \beta R_t + \epsilon_t 
\end{equation}
         """)

st.latex(r"""
H_t = BMG_t - \beta R_t
         """)

st.write(r"""
We can check that the resulting $H_t$ has zero exposure to the initial portfolio.
         """)


st.subheader('Optimal Hedge Ratio')

st.write(r"""
We now have an investment tool - a hedging portfolio - that helps to reduce the exposure to climate risks.
Because it has no loading on the rewarded risk, it has zero expected retunrs and can be used 
in combination with the initial portfolio without affecting the expected returns. 
         """)


st.write(r"""
         Daniel $\textit{et al.}$ (2020) show that we can improve the the initial portfoio by combining it with the hedge portfolio $h$ in order to maximize the Sharpe ratio.
         Given that the hedge portfolio has zero expected return, this is equivalent 
         to finding the combination of $C$ and $H$ that minimizes the variance of the resulting portfolio:
            """)

st.latex(r"""\min_{\delta} \sigma^2(R_C - \delta R_H)""")

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


