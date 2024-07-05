import streamlit as st 
import numpy as np
import plotly.graph_objs as go
import numpy as np


st.title("Climate Risks Beta and Alpha with Aggregated Beliefs")

st.write(r"""
We now turn to the implication in terms of climate risks beta 
(sensitivity of stocks to climate risks) and alpha (climate risks premium) 
in the context of the PST model (2021) \cite{pastor2021sustainable} (hereafter PST).

We first need to determine the implication of aggregation of investors' beliefs
on the market portfolio weights.
We define $\omega_i := W_{0,i}/W_0$ the proportion of 
wealth of investor $i$ at time 0, where $W_0 = \int_i W_{0,i} di$.
Market clearing conditions implies that 
the vector of market portfolio weights $w_m$ is:
         """)

st.latex(r"""
\begin{equation}
    \begin{aligned}
    w_m = \int_i \omega_i X_i di\\
    = \frac{1}{a} \Sigma^{-1} \mu - \frac{1}{a} \Sigma^{-1} \bar{c} \sigma_{\tilde{\epsilon}_1, \tilde{C}_1}
    \end{aligned}
\end{equation}
         """)

st.write(r"""
where $\bar{c} = \int_i \omega_i c_i di \geq 0$ is the wealth-weighted average
expectation about climate risks across investors
(details in Appendix \ref{sec:expectedreturns}) 
and $\mathbf{1}^T w_m = 1$.

Solving for the expected returns $\mu$ gives:
            """)

st.latex(r"""
\begin{equation}
    \mu = a \Sigma w_m + a \Sigma \bar{c} \sigma_{\tilde{\epsilon}_1, \tilde{C}_1}
\end{equation}
""")

st.write(r"""
Details on the derivation of the expected returns are in the companion paper.
Multiplying by $w_m$, we find the market equity premium ($\mu_m = w_m^T \mu$):
""")

st.latex(r"""
\begin{equation}
   \mu_m = a \sigma^2_m + \bar{c} \sigma_{mC}
\end{equation}
         """)

st.write(r"""

where we have the market return variance 
$\sigma^2_m = w_m^T \Sigma w_m$ and 
$w_m^T \sigma_{\tilde{\epsilon}_1, \tilde{C}_1} = \text{Cov}(\tilde{\epsilon}_m, \tilde{C}_1) = \sigma_{mC}$,
the covariance between the market portfolio unexpected return and the climate risks.
We use the previous equation 
to solve for $a$:
""")

st.latex(r"""

\begin{equation}
    \begin{aligned}
        a = \frac{\mu_m - \bar{c} \sigma_{mC}}{\sigma^2_m}
    \end{aligned}
\end{equation}
         
""")

st.write(r"""

Combining this with the equation for $\mu$ and 
noting that $\beta_m = (\frac{1}{\sigma^2_m}\sigma_{\tilde{\epsilon}_1, m})$:
""")

st.latex(r"""
\begin{equation}
    \mu = \mu_m \beta_m + \bar{c}(1 - \rho^2_{mC}) \psi 
\end{equation}
""")

st.write(r"""
where $\psi$ is the $N \times 1$ vector of climate
risks betas 
(slope coefficients on $\tilde{C}_1$ in a multivariate 
regressions of $\tilde{\epsilon}_1$ on $\tilde{\epsilon}_m$ and $\tilde{C}_1$),
and $\rho_{mC}$ is the correlation between $\tilde{\epsilon}_m$ and $\tilde{C}_1$
(details in the companion paper.
         """)

st.write(r"""

Expected returns depend on climate risks betas, $\psi$, which represent 
firms' exposures to non-market climate risks. 
$\tilde{\epsilon}_1$ is the vector of unexpected 
returns, and $\tilde{\epsilon}_m$ is the 
market unexpected return.
A firm climate risks beta $\psi_n$ therefore measures 
its loading on $\tilde{C}_1$, 
after controlling for the market return. 
Stock $n$'s climate risks beta $\psi_n$
enter expected returns positively. Thus:
""")

st.write(r"""
A stock with a negative $\psi_n$ that 
    provides investors with a climate risks hedge,
    has a lower expected return than it would in 
    the absence of climate risks.
         Conversely,
    a stock with a positive $\psi_n$, 
    which performs particularly poorly 
    when the climate worsens unexpectedly, 
    has a higher expected return.
""")

st.write(r"""
Because the vector of stocks' CAPM alphas is 
defined as $\alpha := \mu - \mu_m \beta_m$, we have:
""")

st.latex(r"""
\begin{equation}
    \alpha_n = \bar{c}(1 - \rho^2_{mC}) \psi_n
\end{equation}
""")

st.write(r"""
With the assumption that $\bar{c} > 0$, 
stocks with positive $\psi_n$ have positive alphas,
and stocks with negative $\psi_n$ have negative alphas.
         """)


Sigma = np.array([
    [0.1, 0.01, 0.02, 0.01, 0.02],
    [0.01, 0.1, 0.01, 0.02, 0.01],
    [0.02, 0.01, 0.1, 0.01, 0.02],
    [0.01, 0.02, 0.01, 0.1, 0.01],
    [0.02, 0.01, 0.02, 0.01, 0.1]
])  # Covariance matrix
a = 2  # Risk aversion coefficient
mu = np.array([0.12, 0.10, 0.08, 0.11, 0.09])  # Expected returns

# Alpha Plot
# Define additional parameters
psi = np.array([-0.8, 0.9, -0.6, 0.7, -0.4])  # Climate transition risk betas
rho_mC = 0.5  # Correlation between market unexpected return and climate transition risk

# Calculate the inverse of the covariance matrix
Sigma_inv = np.linalg.inv(Sigma)

# Assume market portfolio
w_m = Sigma_inv.dot(mu)
w_m = w_m / np.sum(w_m)
sigma_m2 = w_m.T.dot(Sigma).dot(w_m)
mu_m = a * sigma_m2

# Calculate beta_m
beta_m = Sigma.dot(w_m) / sigma_m2

# Slider for bar_c value
bar_c = st.sidebar.slider("Average Climate Risk Perception", 0.0, 1., 0.5, 0.01)

# Calculate the alphas for the chosen bar_c
alphas = bar_c * (1 - rho_mC ** 2) * psi

# Prepare data for the alpha plot
alpha_data = {
    "psi": psi,
    "alpha": alphas
}

# Plot the relationship between alphas and climate transition risk betas
fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=alpha_data["psi"],
    y=alpha_data["alpha"],
    mode='lines+markers',
    name=f'Alphas for bar_c = {bar_c}',
    line=dict(shape='linear')
))

# Add the benchmark (bar_c = 0)
alpha_benchmark = np.zeros_like(alphas)
fig2.add_trace(go.Scatter(
    x=psi,
    y=alpha_benchmark,
    mode='lines+markers',
    name='Benchmark (bar_c = 0)',
    line=dict(shape='linear', dash='dash')
))

# Update layout with fixed y-axis
fig2.update_layout(
    title='Relationship between Alphas and Climate Transition Risk Betas',
    xaxis_title='Climate Transition Risk Betas (psi)',
    yaxis_title='Alphas',
    yaxis=dict(range=[-max(abs(alphas)) - 0.5, max(abs(alphas)) + 0.5]),  # Fixed y-axis
    template='plotly_white'
)

# Display the alpha plot in Streamlit
st.plotly_chart(fig2)

st.write(r"""
The higher 
    the wealth-weighted average perception
    of the impact of climate risks $\bar{c}$,
    the more the alphas of stocks with positive climate risks betas $\psi_n$
    are positive. 
Conversely, the higher the wealth-weighted average perception
    of the impact of climate risks $\bar{c}$,
    the more the alphas of stocks with negative climate risks betas $\psi_n$
    are negative.
            """)    