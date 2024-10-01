import streamlit as st
import numpy as np
import plotly.graph_objs as go
import numpy as np
import sympy as sp
import sympy.stats as stats
import plotly.graph_objs as go
import pandas as pd

st.title('Climate Risks as $g$')

st.subheader('')

st.write(r"""
The empirical findings in Ardia $\textit{et al.}$ (2021) and 
Pastor $\textit{et al.}$ (2021) gives an interpretation of the mixed 
results in the literature. Also Ardia $\textit{et al.}$ (2021)
         and Pastor $\textit{et al.}$ (2021) found that a portfolio 
         that is long green stocks and short brown firms exhibits a positive alpha,
         controlling for other known risk factors in the asset pricing literature.
         They showed that such an outperformance is related to both the preference 
         change channels and thus, to both positive cash-flow and negative discount rate 
         news for green stocks. 

To further show that such an outperformance is due to unexpected revaluations during the 
estimation period, Pastor $\textit{et al.}$ (2021) constructed a counter-factual green 
factor assuming zero shocks to climate concerns. The striking result 
in Pastor $\textit{et al.}$ (2021) was that, in absence of climate news, the green 
factor performance would be essentially flat. 
         """)


st.write(r"""
         Value factor and green factor""")

st.write(r"""
Therefore, the empirical evidence in Ardia $\textit{et al.}$ (2021) and Pastor $\textit{et al.}$ (2021)
alongside the mixed results in chasing the climate risk premium in the literature,
suggest that climate risks are unrewarded in the cross-section of stock returns.
         
We've seen in the previous section that exposure to unrewareded risks can be
detrimental to the investor's portfolio, as it undermines the portfolio's mean-variance efficiency.
            """)