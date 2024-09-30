import streamlit as st
import numpy as np
import plotly.graph_objs as go
import sympy as sp
import sympy.stats as stats

st.title('Climate Risks and Stock Returns')

st.write(r"""
         We start by a short introduction to linear risk factors model in empirical asset pricing, and then briefly 
            discuss the implications of climate risks on asset pricing.
         """)

st.subheader('Modelling Risk in Finance with Linear Factor Model')


st.write(r"""
         Grounded in a solid academic background, factor models provide a general framework to analyse the systematic 
         drivers to explain the cross-section of asset returns. Linear factor models assume that the return of an asset 
         can be modelled as a combination of underlying risk factors:""")

st.latex(r'''
         \begin{equation}
         r_{i,t} = \alpha_i + \sum^{K}_{k=1} \beta_{i,k} f_{k,t} + \epsilon_{i,t}
         \end{equation}
         ''')

st.write(r"""A common practice in the academic finance literature has been 
         to create characteristic portfolios by sorting on characteristics 
         positively associated with expected returns. The methodology involves 
         going long a portfolio of high characteristic firms and short a portfolio of low characteristic firms.
         The returns of the resulting portfolios are supposed to proxy for the returns of the underlying risk factors $f_{k,t}$.
         $\beta_{i,k}$ is the sensitivity of the asset return to the $k$-th factor, and $\epsilon_{i,t}$ is the idiosyncratic
            component of the asset return.
            """)

st.write(r"""
         Fama and French (2015) use the dividend discount model to motivate the use of a combination of firm 
         characteristics based on valuation, profitability and investment to explain the cross-section of stock returns:
""")

st.latex(r'''
            \begin{equation}
         \frac{M_t}{B_t} = \frac{\sum^{\infty}_{\tau = 1} \mathbb{E}(Y_{t+\tau} - dB_{t+\tau})/(1+r)^{\tau}}{B_t}
        \end{equation}
            ''')

st.write(r"""
where $M_t$ is the market value of equity, $B_t$ is the book value of equity, $Y_{t+\tau}$ is the expected future earnings, 
$dB_{t+\tau}$ is the expected change in book value of equity, and $r$ is, approximately, the expected return.
         """)

st.write(r"""
This model implies three statements about expected returns. First, 
         if we fix everything except the current value of the stock, $M_t$,
         and the expected stock return, $r$, then a lower value of $M_t$, or 
         equivalently a higher book-to-market ratio, implies a higher expected return.

Next, if we fix $M_t$ and the values of everything except the expected future earnings and 
         the expected stock return, the equation tells us that higher expected earnings imply 
         a higher expected return.

Finally, for fixed values of $B_t$, $M_t$ and expected earnings, higher expected growth in 
book equity - investment - implies a lower expected return.
         """)

st.write(r"""
This justifies the development of a five-factor model that includes $MktRF$, $SMB$, 
         $HML$, $RMW$ and $CMA$ characteristic portfolios to explain the cross-section of stock returns, based on a 
         time series regressions like:
         """)

st.latex(r'''
            \begin{equation}
            r_{i,t} = \alpha_i + \beta_{i,MktRF} MktRF_t + \beta_{i,SMB} SMB_t + \beta_{i,HML} HML_t + \beta_{i,RMW} RMW_t + \beta_{i,CMA} CMA_t + \epsilon_{i,t}
            \end{equation}
            ''')

st.write(r"""
         where $SMB$, $HML$, $RMW$ and $CMA$ are the returns of the size, value, profitability and investment characteristic-sorted 
         portfolios, respectively.
         """)


st.subheader('Climate Risks and Asset Pricing')



st.write(r"""
Climate risks are categorized into physical and transition risks (Carney, 2015). Physical risks arise from climate and weather events that impact company operations or society (Tankov and Tantet, 2019). These can be:
- Acute (e.g., extreme weather events).
- Chronic (e.g., long-term shifts in climate patterns).

Transition risks relate to scenarios leading to a low-carbon economy and the impact on fossil fuels and related sectors (Curtin et al., 2019).
""")

