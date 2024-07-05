import streamlit as st 
import numpy as np
import plotly.graph_objs as go
import numpy as np


st.title("Optimal Portfolio with Climate Change Uncertainty")

st.write(r"""

To see the implications of climate change uncertainty in 
a CAPM model, we first need to extend a bit the model.
We have $N$ firms, and we denote $\tilde{r}_1$ the
vector of excess returns of the $N$ firms from time 0 to time 1.
$\tilde{r}_1$ is assumed to be normally distributed:

         """)

st.latex(r"""
\begin{equation}
    \tilde{r}_1 = \mu + \tilde{\epsilon}_1
\end{equation}
""")

st.write(r"""
with the unexpected returns $\tilde{\epsilon}_1 \sim N(0, \Sigma)$ and $\mu$ 
the expected returns.  
$X_i$ is a $N \times 1$ vector of portfolio weights,
and $W_{0,i}$ is the wealth of investor $i$ at time 0.
The wealth of investor $i$ at time 1 is:
         """)

st.latex(r"""

\begin{equation}
    \tilde{W}_{1,i} = W_{0,i}(1 + r_f + X_i^T \tilde{r}_1)
\end{equation}
""")

st.write(r"""

where $r_f$ is the risk free rate.

The investor has an exponential utility function:
             
    """)

st.latex(r"""

\begin{equation}
    V(\tilde{W}_{1,i}, X_i, \tilde{C}_1) = -\exp{(-A_i \tilde{W}_{1,i} - c_i \tilde{C}_1)}
\end{equation}
""")

st.write(r"""
where $A_i$ is the investor's absolute risk aversion.
To obtain the optimal portfolio weights $X_i$ at time 0,
investor $i$ seeks to maximize its expected utility.
We obtain the first order conditions 
by computing the expectation of the utility function 
and differentiating it with respect to $X_i$ (
see the companion paper for details).
The investor $i$'s portfolio weights $X_i$ are:
""")

st.latex(r"""

\begin{equation}
    X_i = \frac{1}{a}\Sigma^{-1} ( \mu - c_i \sigma_{\tilde{\epsilon}_1, \tilde{C}_1})
\end{equation}
""")


# Function to calculate the optimal portfolio
def optimal_portfolio(a, c_i, mu, Sigma, sigma_epsilon_C1):
    Sigma_inv = np.linalg.inv(Sigma)
    X_i = (1 / a) * Sigma_inv.dot(mu - c_i * sigma_epsilon_C1)
    # Normalize the weights to sum up to 1
    X_i = X_i / np.sum(X_i)
    return X_i

    # Define specific parameters for 5 assets
num_assets = 5
mu = np.array([0.12, 0.10, 0.08, 0.11, 0.09])  # Expected returns
Sigma = np.array([
    [0.1, 0.01, 0.02, 0.01, 0.02],
    [0.01, 0.1, 0.01, 0.02, 0.01],
    [0.02, 0.01, 0.1, 0.01, 0.02],
    [0.01, 0.02, 0.01, 0.1, 0.01],
    [0.02, 0.01, 0.02, 0.01, 0.1]
])  # Covariance matrix
sigma_epsilon_C1 = np.array([-0.8, 0.9, -0.6, 0.7, -0.4])  # Covariance with climate risk

a = 2  # Risk aversion coefficient

# Interactive slider for c_i
c_i = st.sidebar.slider("Climate Risk Perception ($c_i$)", -1.0, 1.0, 0.0, 0.01)

# Calculate the optimal portfolio
X_i = optimal_portfolio(a, c_i, mu, Sigma, sigma_epsilon_C1)
# Calculate the optimal portfolio for the benchmark (c_i = 0)
X_i_benchmark = optimal_portfolio(a, 0, mu, Sigma, sigma_epsilon_C1)

# Plot the portfolio weights using Plotly
fig = go.Figure()

# Plot for the variable c_i
fig.add_trace(go.Scatter(
    x=sigma_epsilon_C1,
    y=X_i,
    mode='markers',
    marker=dict(size=10, color='blue', opacity=0.6),
    name=f'Investor with $c_i$ = {c_i}'
))

# Plot for the benchmark (c_i = 0)
fig.add_trace(go.Scatter(
    x=sigma_epsilon_C1,
    y=X_i_benchmark,
    mode='markers',
    marker=dict(size=10, color='red', opacity=0.6),
    name='Benchmark ($c_i$ = 0)'
))

fig.update_layout(
    title=f'Portfolio Weights vs. Covariance with Climate Transition Risk',
    xaxis_title='Covariance Between Unexpected Returns and Climate Transition Risk',
    yaxis_title='Portfolio Weight',
    template='plotly_white'
)

# Display the plot in Streamlit
st.plotly_chart(fig)

st.write(r"""
with $a_i = A_i W_{0,i}$, the relative 
investor $i$'s risk aversion and we assume 
$a_i = a$ for simplicity. We have $\sigma_{\tilde{\epsilon}_1, \tilde{C}_1}$
the covariance between the unexpected returns and the climate risks.
         """)

st.write(r"""
The higher 
the investor $i$'s perception 
of the impact of climate risks $c_i$,
the more the investor will long stocks whose unexpected returns 
are negatively correlated with the climate risks
and short stocks whose unexpected returns are positively correlated.
For the investor who does not think that climate 
risks will materialize ($c_i = 0$), 
the optimal portfolio weights are not affected by the
covariance between the unexpected returns and the climate risks.
""")
