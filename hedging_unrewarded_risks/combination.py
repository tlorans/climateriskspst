import streamlit as st
import numpy as np
import plotly.graph_objs as go
import numpy as np


st.title('Factor Efficient Portfolio')

st.write(r"""
         Now that we have a tool to manage exposure to unrewarded risk, while keeping exposure to rewarded risk, we can go further.
         """)

st.subheader('Combined Portfolio')

st.write(r"""
         Our initial portfolio $c$ with default values for the loadings on the rewarded and unrewarded risks is have variance such as:
            """)

st.latex(r"""
\begin{equation}
         \sigma^2_c = \frac{2}{3}\sigma^2_{\epsilon} + 4 \sigma^2_f + \frac{2}{3}\sigma^2_g
\end{equation}
         """)

st.write(r"""
         and the variance of the hedging portfolio $h$ is:
            """)

st.latex(r"""
\begin{equation}
            \sigma^2_h = \sigma^2_{\epsilon} + 4 \sigma^2_g
\end{equation}
            """)

st.subheader('Optimal Hedge Ratio')

st.subheader('Conclusion')