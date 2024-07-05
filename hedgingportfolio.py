import streamlit as st 
import numpy as np
import plotly.graph_objs as go
import numpy as np

st.title("Climate Risks Hedging Portfolio")

st.write(r"""
We can reexpress the investor's optimal 
portfolio weights $X_i$ in terms of
climate risks betas. Substituting the 
expected returns $\mu = \mu_m \beta_m + \bar{c}(1 - \rho^2_{mC}) \psi$
into the portfolio weight, we get:
            """)    

st.latex(r"""
\begin{equation}
    X_i = w_m  - \frac{\gamma_i}{a}\Sigma^{-1}\sigma_{\tilde{\epsilon}_1, \tilde{C}_1}
\end{equation}
""")

st.write(r"""
where $\gamma_i = c_i - \bar{c}$ and 
$\sigma_{\tilde{\epsilon}_1, \tilde{C}_1}$ 
is the vector of covariances between 
unexpected returns and climate risks.

The climate risks hedging portfolio, $\Sigma^{-1}\sigma_{\tilde{\epsilon}_1, \tilde{C}_1}$,
is a natural mimicking portfolio for $\tilde{C}_1$.
Indeed, note that the $N$ elements of $\Sigma^{-1}\sigma_{\tilde{\epsilon}_1, \tilde{C}_1}$
are the slopes of the multivariate regression of $\tilde{C}_1$ on $\tilde{\epsilon}_1$.
Therefore, the return on the hedging portfolio 
has the highest correlation with $\tilde{C}_1$,
among all portfolios of the $N$ stocks. 
Investors in this model hold this maximum-correlation 
portfolio, to various degree, determined by their 
$\gamma_i$, to hedge climate risks. 
Specifically, 
the investor allocates a fraction $\phi_i$ 
of her remaining wealth to the climate risks hedging 
portfolio, and a fraction $1 - \phi_i$ to 
the market portfolio.
To see this, we note that the $N \times 1$ vector 
of weights within investor $i$'s stock portfolio $w_i$ is
$X_i$ normalized by the sum of its elements:
""")

st.latex(r"""
\begin{equation}
    \begin{aligned}
        w_i = X_i / \mathbf{1}^T X_i \\
        = \frac{w_m - \frac{\gamma_i}{a} \Sigma^{-1} \sigma_{\tilde{\epsilon}_1, \tilde{C}_1}}{\mathbf{1}^T (w_m - \frac{\gamma_i}{a} \Sigma^{-1} \sigma_{\tilde{\epsilon}_1, \tilde{C}_1})} \\
    \end{aligned}
\end{equation}
""")

st.write(r"""
and we obtain (see the companion paper for details):
""")

st.latex(r"""
\begin{equation}
    \begin{aligned}
        w_i = w_m \phi_i + w_{\psi} (1 - \phi_i) \\
    \end{aligned}
\end{equation}
""")

st.write(r"""
with $\phi_i = \frac{ \frac{\gamma_i}{a} \mathbf{1}^T \Sigma^{-1} \sigma_{\tilde{\epsilon}_1, \tilde{C}_1}}{1 - \frac{\gamma_i}{a} \mathbf{1}^T \Sigma^{-1} \sigma_{\tilde{\epsilon}_1, \tilde{C}_1}}$.
""")

Sigma = np.array([
    [0.1, 0.01, 0.02, 0.01, 0.02],
    [0.01, 0.1, 0.01, 0.02, 0.01],
    [0.02, 0.01, 0.1, 0.01, 0.02],
    [0.01, 0.02, 0.01, 0.1, 0.01],
    [0.02, 0.01, 0.02, 0.01, 0.1]
])
mu = np.array([0.12, 0.10, 0.08, 0.11, 0.09])  # Expected returns
sigma_epsilon_C1 = np.array([-0.8, 0.9, -0.6, 0.7, -0.4])  # Covariance with climate risk
a = 2  # Risk aversion coefficient

gamma_i_values = np.linspace(-4, 4, 100)  # Different values of gamma_i

# Calculate the inverse of the covariance matrix
Sigma_inv = np.linalg.inv(Sigma)

# Assume market portfolio
w_m = Sigma_inv.dot(mu)
w_m = w_m / np.sum(w_m)

# Calculate the transition risk hedging portfolio weights
hedging_portfolio = Sigma_inv.dot(sigma_epsilon_C1)

# Calculate \mathbf{1}^T \Sigma^{-1} \sigma_{\tilde{\epsilon}_1, \tilde{C}_1}
one_T_Sigma_inv_sigma_epsilon_C1 = np.sum(hedging_portfolio)

# Calculate the allocation fractions
phi_values = []
market_allocations = []
hedging_allocations = []

for gamma_i in gamma_i_values:
    phi_i = (gamma_i / a) * one_T_Sigma_inv_sigma_epsilon_C1 / (1 + (gamma_i / a) * one_T_Sigma_inv_sigma_epsilon_C1)
    phi_values.append(phi_i)
    market_allocations.append(1 - phi_i)
    hedging_allocations.append(phi_i)

# Interactive slider for gamma_i
selected_gamma_i = st.sidebar.slider("Select Deviation from Average Climate Risks Perception", float(gamma_i_values.min()), float(gamma_i_values.max()), 0.0, 0.1)

# Find the closest value in gamma_i_values to the selected_gamma_i
closest_gamma_i_index = (np.abs(gamma_i_values - selected_gamma_i)).argmin()
selected_market_allocation = market_allocations[closest_gamma_i_index]
selected_hedging_allocation = hedging_allocations[closest_gamma_i_index]

# Create the bar plot
fig = go.Figure(data=[
    go.Bar(name='Market Portfolio Allocation', x=['Market Portfolio'], y=[selected_market_allocation], marker_color='blue'),
    go.Bar(name='Hedging Portfolio Allocation', x=['Hedging Portfolio'], y=[selected_hedging_allocation], marker_color='green')
])

# Update layout
fig.update_layout(
    title='Allocation Between Market and Hedging Portfolio',
    xaxis_title='Portfolio',
    yaxis_title='Allocation Fraction',
    # yaxis=dict(range=[0, 1]),  # Fixed y-axis
    barmode='group',
    template='plotly_white'
)

# Display the plot in Streamlit
st.plotly_chart(fig)

st.write(r"""

The climate risks hedging portfolio have weights 
proportional to $\Sigma^{-1}\sigma_{\tilde{\epsilon}_1, \tilde{C}_1}$.
Investors with $\gamma_i > 0$, whose climate risks 
expectation 
is higher than the market average, 
will short the hedging portfolio, whereas investors with
$\gamma_i < 0$ will go long on the hedging portfolio.
The allocation fractions $\phi_i$ are shown in Figure \ref{fig:transitionriskhedgingportfolio},
as a function of $\gamma_i$:
""")

st.write(r"""
If the the wealth-weighted average investor's 
    perception of climate risks $\bar{c} = 0$, 
    then everyone holds the market portfolio.*
If the investor has the same perception as the market, 
    then $\gamma_i = 0$ and the investor holds the market portfolio.
         
For a specific climate risks industry to exist, 
there must be investors with $\gamma_i > 0$ and $\gamma_i < 0$,
that is, it must exist diversity in the perception of climate risks,
due to climate change ambiguity.
""")