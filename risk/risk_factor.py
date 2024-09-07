import streamlit as st
import numpy as np
import plotly.graph_objs as go
import numpy as np
from scipy.optimize import fsolve


st.title('What is risk in Finance?')

st.write(r"""
Grounded in a solid academic background, factor model provide a general framework 
         to analyse the systematic drivers to explain the cross-section of returns.
         This framework assumes that the return of an asset can be modelled as a combination 
         of underlying risk factors:
""")

st.latex(r"""
\begin{equation}
         r_{i,t} = \alpha_i + \sum_{k=1}^{K} \beta_{i,k} f_{k,t} + \epsilon_{i,t}
\end{equation}
         """)

st.write(r"""
         A common practice in the academic finance literature has been to create characterstic portfolios (CP)
         by sorting on characteristics positively associated with expected returns.
         The resulting portfolios, that goes long on a portfolio of firms with high 
         characteristics and short on a portfolio of firms with low characteristics,
         then serve as a model for returns.

         Fama and French (1993, 2015) are prominent exemples of this approach. They explain (Fama and French, 2015)
         sources of risk from a dividend discount model such as:
         """)

st.latex(r"""
\begin{equation}
         \frac{M_t}{B_t} = \frac{\sum_{\tau = 1}^{\infty} E(Y_{t+\tau} - dB_{t+\tau})}{B_t (1 + r)^{\tau}}
\end{equation}
         """)

st.write(r"""
         where $M_t$ is the market value of equity, $B_t$ is the book value of equity, $E(Y_{t+\tau})$ are the expected earnings,
         $E(dB_{t+\tau})$ are the expected changes in book value of equity, and $r$ is approximately the expected return on equity.
         The rationale behind this is the following: fixed all other variabels except one and the expected returns, 
         and see what should happen to the expected retuns if you change the variable to keep the identity true.

         1. Fix everything except $M_t$ and $r$: if $M_t$ decreases, $r$ should increase. Then, a lower value 
            of $M_t$ - or a higher book-to-market ratio - should be associated with higher expected returns. Investors
            should be compensated for the risk of holding a stock with a high book-to-market ratio.
         This is the value factor.
         2. Fix everything except $Y_{t+\tau}$ and $r$: if $Y_{t+\tau}$ increases, $r$ should increase. Then, a higher value
            of $Y_{t+\tau}$ - or a higher expected earnings - should be associated with higher expected returns. This is 
         the profitability factor.
         3. Fix everything except $dB_{t+\tau}$ and $r$: if $dB_{t+\tau}$ decrease, $r$ should increase. Then, a lower value
            of $dB_{t+\tau}$ - or a lower expected changes in book value of equity - should be associated with higher expected returns.
            This is the investment factor.
         
         And we may add Size and Momentum characteristics as a proxy for expected earnings. 

         This leads to the famous six-factor model of Fama and French (2015):
         """)

st.latex(r"""
\begin{equation}
            r_{i,t} = \alpha_i + \beta_{i,MKT} MKT_t + \beta_{i,SMB} SMB_t + \beta_{i,HML} HML_t + \beta_{i,UMD} UMD_t + 
         \epsilon_{i,t}
\end{equation}
            """)
st.write(r"""
            where $MKT_t$ is the market factor, $SMB_t$ is the size factor, $HML_t$ is the value factor, $UMD_t$ is the momentum factor,
            and $\epsilon_{i,t}$ is the idiosyncratic risk. The $\beta$ coefficients are the factor loadings.
            The market factor is the return on the market portfolio, the size factor is the return on a portfolio of small stocks minus the return
            on a portfolio of big stocks, the value factor is the return on a portfolio of high book-to-market stocks minus the return on a portfolio
            of low book-to-market stocks, and the momentum factor is the return on a portfolio of winners minus the return on a portfolio of losers.
            The idiosyncratic risk is the risk that is specific to an asset and cannot be diversified away.
            """)


st.subheader('Characteristic-sorted portfolio')
st.write(r"""
Fama and French (1993) and subsequent papers construct a portfolio as proxy for the priced risk associated with 
characteristic premia. 
         
We consider a single period economy with $N$ assets. Realized returns are determined by a one-factor structure:
""")

st.latex(r"""
         \begin{equation}
         r_{i} = \beta_i(f+\lambda)+\epsilon_i
            \end{equation}
            """)

st.write(r"""
         where $E(f) = E(\epsilon) = 0$. We have a column vector of returns $r^T = [r_1, r_2, ..., r_N]$.
         Taking expectations of the returns, we have:
         """)

st.latex(r"""
            \begin{equation}
         \mu = E(r) = \beta \lambda
            \end{equation}
            """)
st.write(r"""
            where $\mu$ is the expected return on the portfolio of assets and $\beta$ is the column vector of betas (individual
            asset exposures to the factor $f$).
         

         """
)



st.subheader('Risk premia')
st.write(r"""

""")
