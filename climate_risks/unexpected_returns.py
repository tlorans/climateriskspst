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

st.title('Unexpected Returns')

st.write(r"""
Some papers have found risk premia 
         associated with climate risks.
         Why may we have such differences in the literature?
         Pastor $\textit{et al.}$ (2022) and Ardia $\textit{et al.}$ (2022) provide an interpretation of the mixed results in the literature.
         Pastor $\textit{et al.}$ (2022)
         found that a portfolio that is long green stocks and short brown firms exhibits a positive alpha,
            controlling for other known risk factors in the asset pricing literature.
         But are there really expected returns associated with climate risks?
         Or are they unexpected returns that are driving the results?
""")

st.subheader("Expected vs. Unexpected Returns")

st.write(r""" Explain differences""")

st.subheader("Climate Concerns and Unexpected Returns")

st.write(r"""
Climate concerns explain dynamics of unexpected returns.
""")