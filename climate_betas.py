import streamlit as st


st.title('Climate Betas')

st.write(r"""
We have seen in the previous sections that climate risks hedging 
portfolio is proportional to the climate risks betas $\psi_n$. Now 
comes the question of how to estimate these climate betas.
         """)

st.write(r"""
In PST (2021), the assumption is that 
some characteristics
of stocks, approximated with 
environmental scores in Pastor *et al.* (2022)) $g_n$,
may be negatively correlated with $\psi_n$. 
For example, in the case of transition risks,
"greener" firms may have negative $\psi_n$,
while "browner" firms may have positive $\psi_n$.
In that case, with $\zeta > 0$, we have:
""")

st.latex(r"""
\begin{equation}
    \psi_n = -\zeta g_n
\end{equation}
""")

st.write(r"""
and therefore:
""")

st.latex(r"""

\begin{equation}
    \mu = \mu_m \beta_m - \bar{c}(1 - \rho^2_{mC}) \zeta g
\end{equation}
""")

st.write(r"""
and:
""")

st.latex(r"""
\begin{equation}
    \alpha_n = \bar{c}(1 - \rho^2_{mC}) \zeta g_n
\end{equation}
""")

st.write(r"""
The climate risks hedging portfolio will 
therefore be proportional to the climate risks-related 
characteristics
of the stocks $g$.
""")

st.write(r"""
Another approach is mimicking portfolio from Lamont (2001) [1],
as advocated by Engle *et al.* (2020) [2]. 
We may not observe $\tilde{C}_1$ directly, and 
therefore estimate $\psi_n$ by regressing the unexpected returns
$\tilde{\epsilon}_1$ on the climate risks $\tilde{C}_1$. But 
we can observe the change in perception of climate risks 
($\bar{c}_1 - E_0(\bar{c}_1)$) by taking unexpected change of climate concerns 
from news articles, as in Pastor *et al.* (2022). 
We can therefore estimate $\psi_n$ by regressing unexpected returns
$\tilde{\epsilon}_1$ on the change in perception of climate risks 
($\bar{c}_1 - E_0(\bar{c}_1)$).
""")


st.markdown("""

## References
            
[1]: Lamont, O. A. (2001). Economic tracking portfolios. Journal of Econometrics, 105(1), 161-184.
            
[2]: Engle, R. F., Giglio, S., Kelly, B., Lee, H., & Stroebel, J. (2020). Hedging climate change news. The Review of Financial Studies, 33(3), 1184-1216.
            
            """)
