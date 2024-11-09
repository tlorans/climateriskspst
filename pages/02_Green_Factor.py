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
from regtabletotext import prettify_result

st.title('Green Factor')

st.subheader('Green ETF')


factors_world = {
    "value": "IVLU",
    "size": "IWSZ.L",
    "momentum": "IMTM",
    "quality": "IWQU.L",
}