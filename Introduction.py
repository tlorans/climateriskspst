import streamlit as st

st.set_page_config(page_title="Climate Risks")

st.title('Mean-Variance Efficient Portfolio with Multi-factor Model')

st.write(r"""We follow Rjaily $\textit{et al.}$ (2024).""")

st.write(r"""In an investment universe with $N$ assets, the investor assumes 
         that asset returns follow a multi-factor risk model:""")

st.latex(r"""
\begin{equation}
         R = B F + \epsilon
\end{equation}
         """)

st.write(r"""where $R$ is the $N \times 1$ vector of asset excess returns,
            $F$ is the $K \times 1$ vector of factor excess returns,
            $B$ is the $N \times K$ matrix of factor loadings, and
            $\epsilon$ is the $N \times 1$ vector of idiosyncratic returns.""")

st.write(r"""
We assume that $E(F) = \lambda$, $\text{Cov}(F) = \Omega$, $E(\epsilon) = 0$, $\text{Cov}(\epsilon) = D$,
and $\text{Cov}(F, \epsilon) = 0$.
         """)

st.write(r"""The vector of expected returns is given by:""")

st.latex(r"""
\begin{equation}
         \mu = B \lambda
\end{equation}
         """)

st.write(r"""While the covariance matrix of asset returns is given by:""")

st.latex(r"""
\begin{equation}
         \Sigma = B \Omega B^\top + D
\end{equation}
         """)


st.write(r"""Given a portfolio $x$ with a vector of weight $w$, it's expected returns is given by:""")

st.latex(r"""
\begin{equation}
         \mu_x = w^\top \mu = w^\top B \lambda
\end{equation}
         """)

st.write(r"""The variance of the portfolio is given by:""")

st.latex(r"""
\begin{equation}
         \sigma_x^2 = w^\top \Sigma w = w^\top(B \Omega B^\top w + D) w
\end{equation}
         """)

st.write(r"""The vector of beta coefficients is given by:""")

st.latex(r"""
\begin{equation}
         \beta_x = B^\top w
\end{equation}
         """)

st.write(r"""The Sharpe ratio of the portfolio is given by:""")

st.latex(r"""
\begin{equation}
         \text{SR}_x = \frac{\mu_x}{\sqrt{\sigma_x^2}}
\end{equation}
         """)

st.write(r"""We can assess the importance of common factors by calculating the propotyion 
         of the portfolio variance that is explained by the factors.""")

st.latex(r"""
         \begin{equation}
         \text{R}_c^{2} = \frac{w^\top B \Omega B^\top w}{w^\top(B \Omega B^\top w + D) w}
         \end{equation}
         """)

st.write(r"""We can uncover the individual contribution of each factor to the portfolio variance:""")

st.latex(r"""
            \begin{equation}
         \text{R}_j^{2} = \frac{w^\top B_j B_j^\top \sigma_j^2 w}{w^\top(B \Omega B^\top w + D) w}
            \end{equation}
            """)

st.write(r"""where $B_j$ is the $N \times 1$ vector of factor loadings on factor $j$,
            and $\sigma_j^2$ is the variance of factor $j$.""")

st.write(r"""
         The investor seeks to maximize the Sharpe ratio of the portfolio by choosing the optimal weights $w$.
         """)

st.write(r"""We consider the following optimization problem:""")

st.latex(r'''
         \begin{equation}
         \begin{aligned}
w^*(\gamma) = \underset{w}{\text{arg min}} \left( \frac{1}{2} w^\top \Sigma w - \gamma w^\top \mu \right) \\
\text{ subject to } \mathbf{1}_n^\top w = 1
\end{aligned}
\end{equation}
''')

st.write("where $\gamma$ is the risk tolerance parameter. We have the following solution:")

st.latex(r'''
w^* = \gamma \Sigma^{-1} \mu
''')

st.write(r"""where $\gamma = (1_n^\top \Sigma^{-1}\mu)^{-1}$.""")

