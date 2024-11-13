import streamlit as st
from plotnine import *
from mizani.formatters import date_format
from mizani.breaks import date_breaks
from mizani.formatters import percent_format
import numpy as np
import pandas as pd
import yfinance as yf
from PIL import Image


st.title('Long-Short Strategy')

st.write(r'''
We have an investable proxy for the green factor, and 
a model that seems to be able to infer the transition regimes.
More precisely, we have a signal indicating a bull or bear market for 
our green ETF. 
We therefore have all the ingredients to implement an investment strategy.
We are going to implement a simple long-short strategy, conditional on the
signal we have.
''')


st.subheader('Turning Signals into Positions')  


st.subheader('Backtesting Protocol')

st.write(r'''
For an active strategy driven by signals, defining the buffer period versus the backtest period is essential:

1. **Buffer Period**: In our case, we use a two-year buffer period to train the regime fitting model. This period helps us to calibrate the model, validate signals, and understand the signal dynamics without executing trades. The two-year period provides enough historical data to reliably detect bull and bear regimes.

2. **Backtest Period**: After the buffer period, we enter the backtest phase, which lasts for one year. During this period, the inferred bull and bear regimes are used to actively adjust the portfolio based on the signals. The strategy dynamically changes positions as new signals are generated, allowing us to evaluate its real-time decision-making effectiveness.

3. **Rolling Forward**: After each backtest year, we roll the window forward by one year. This means that the model is retrained using the most recent two years of data, and the inferred regimes are then used for the next one-year backtest period. This rolling approach ensures that the strategy remains adaptive to changing market conditions and continuously learns from the most up-to-date data.

In this way, the two-year buffer period ensures the model and signals are calibrated properly, while the one-year backtest period allows for a thorough evaluation of the strategy's performance in a dynamic, signal-driven context.
''')


st.subheader('Performance Metrics')


st.subheader('Conclusion')


st.subheader('Exercice')
