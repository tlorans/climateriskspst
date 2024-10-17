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
import matplotlib.pyplot as plt

st.title('...or _Variance_ Factor? (Unrewarded Risk)')




st.subheader("")

st.write(r"""
BMG may not helps in explain average returns but it may help to explain return variance - it can help to increase
         $R^2$.
         """)

st.write(r"""
         High $R^2$ is interesting. The fact that $R^2$ are high means that 
         the regression used to define the loadings explains most of the variance $\sigma^2_i$ of the asset $i$, 
         even if alpha is big.""")




st.subheader("Project 4: Value and BMG Portfolio")

st.write(r"""
         Is climate risk an unrewarded factor that may 
         decrease the mean-variance efficiency of investors' portfolios?
            """)

st.write(r"""Use the same recipe than in our unrewarded risk section:
        - Volatility of the BMG portfolio?
        - Correlation between Value and BMG portfolio?
        - Resulting variance of Value explained by BMG?
        """)
