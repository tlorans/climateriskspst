import streamlit as st
import numpy as np
import plotly.graph_objs as go
import numpy as np


st.title('Rewarded and Unrewarded Risk in Finance')


st.subheader('Simple One-Period Two-Factor Model')

st.write(r"""
Following Daniel $\textit {et al.}$ (2020), 
we consider a one-period economy with $N$ assets. 
Realized excess returns are given by:
""")

st.latex(r"""
        \begin{equation}
r_i = \beta_i (f + \lambda) + \gamma_i g + \epsilon_i
\end{equation}
         """)

st.write(r"""
with $r_i$ the excess return of asset $i$, 
$\beta_i$ the factor loading on the rewarded factor $f$,
$\lambda$ the risk premium on $f$,
$\gamma_i$ the factor loading on the unrewarded factor $g$,
$\epsilon_i$ the idiosyncratic return. 
We have 
$\mathbb{E}[\epsilon_i] = \mathbb{E}[f] = \mathbb{E}[g] = 0$.
         """)

st.subheader('What About Climate Risks?')

st.write(r"""
Question, is Climate Risks $f$, $g$ or $\epsilon$?
""")