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
c_i = st.sidebar.slider("Climate Risk Perception ($c_i$)", 0., 2., 0.0, 0.01)

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



# Footnote explanation
st.write(r"""
---  
**Derivation of the Optimal Portfolio:**  
Taking the expectation of the utility function from period 0,
we get: 

$$
    E_0(V(\tilde{W}_{1,i}, X_i, \tilde{C}_1)) = E_0(-\exp{(-A_i W_{0,i} - c_i \tilde{C}_1)})
$$

We can replace $\tilde{W}_{1,i}$
with the relation $\tilde{W}_{1,i} = W_{0,i}(1 + r_f + X_i^T \tilde{r}_1)$
and define $a_i := A_i W_{0,i}$. 
We want to make out from the expectation the 
terms that we know about in period 0, and 
reexpress the terms with the expectation as a function 
of the portfolio weights $X_i$. 

$$
\begin{aligned}
    E_0(V(\tilde{W}_{1,i}, X_i, \tilde{C}_1)) = E_0(-\exp{(-A_i W_{0,i} - c_i \tilde{C}_1)}) \\
    = E_0(-\exp{(-a_i(1 + r_f + X_i^T \tilde{r}_1) - c_i \tilde{C}_1)}) \\
    = -\exp{(-a_i(1 + r_f))} E_0(-\exp{(-a_i X_i^T \tilde{r}_1 - c_i \tilde{C}_1)}) \\
    = -\exp{(-a_i(1 + r_f))} -\exp(a_i X_i^T E_0(\tilde{r}_1) +\\
    \frac{1}{2}a^2_i X_i^T \text{Var}(\tilde{\epsilon}_1)X_i + a_i c_i X_i^T \text{Cov}(\tilde{\epsilon}_1, \tilde{C}_1) + \frac{1}{2}c_i^2 \text{Var}(\tilde{C}_1)) \\
    = -\exp(-a_i(1 + r_f)) - \exp(-a_i X_i^T\mu  + \\
    \frac{1}{2}a_i^2 X_i^T \Sigma X_i + a_i c_i X_i^T \sigma_{\tilde{\epsilon}_1, \tilde{C}_1} + \frac{1}{2}c_i^2 \sigma^2_{\tilde{C}_1})
\end{aligned}
$$

where $\sigma_{\tilde{\epsilon}_1, \tilde{C}_1} = \text{Cov}(\tilde{\epsilon}_1, \tilde{C}_1)$.

Investor $i$ seeks to maximize its expected utility,
by choosing the optimal portfolio weights $X_i$ at time 0.
We need to find the first order conditions for the optimization problem.

We combine the exponential terms:
         
$$
    \begin{aligned}
    E_0(V(\tilde{W}_1, X_i, \tilde{C}_1)) = -\exp(-a_i(1 + r_f) -a_i X_i^T\mu + \\
    \frac{1}{2}a_i^2 X_i^T \Sigma X_i + a_i c_i X_i^T \sigma_{\tilde{\epsilon}_1, \tilde{C}_1} + \frac{1}{2}c_i^2 \sigma^2_{\tilde{C}_1})
    \end{aligned}
$$

and let $f(X_i)$ denotes the exponent: 
         
$$
    E_0(V(\tilde{W}_1, X_i, \tilde{C}_1)) = -\exp(f(X_i))
$$
 
To differentiate $f(X_i)$ with respect to $X_i$, 
 we use the chain rule $\frac{\partial h}{\partial X_i} = \frac{\partial h}{\partial f} \frac{\partial f}{\partial X_i}$.
If $h = - \exp(f)$, then $\frac{\partial h}{\partial f} = -\exp(f)$. Thus: 
         
$$
    \frac{\partial h}{\partial X_i} = -\exp(f) \frac{\partial f}{\partial X_i}
$$

We use the rules that $\frac{\partial x^T b}{\partial x} = b$ and 
$\frac{\partial x^T A x}{\partial x} = 2Ax$:
         
$$
    \begin{aligned}
        \frac{\partial f}{\partial X_i} = -a_i \mu + a_i^2 \Sigma X_i + a_i c_i \sigma_{\tilde{\epsilon}_1, \tilde{C}_1} \\
    \end{aligned}
$$
         
Combining:
         
$$
    \frac{\partial h}{\partial X_i} = -\exp(f) (-a_i\mu + a_i^2 \Sigma X_i + a_i c_i \sigma_{\tilde{\epsilon}_1, \tilde{C}_1})
$$

We set the derivative to zero:
         
$$
    \begin{aligned}
        -\exp(f)(-a_i\mu + a_i^2 \Sigma X_i + a_i c_i \sigma_{\tilde{\epsilon}_1, \tilde{C}_1}) = 0 \\
        -a_i\mu  + a_i^2 \Sigma X_i + a_i c_i \sigma_{\tilde{\epsilon}_1, \tilde{C}_1} = 0 \\
    \end{aligned}
$$
         
because the exponential term is always positive.
 
We solve for $X_i$:
         
$$
    \begin{aligned}
        a_i^2 \Sigma X_i = a_i\mu - c_i \sigma_{\tilde{\epsilon}_1, \tilde{C}_1}) \\
        a_i \Sigma X_i = \mu  - c_i \sigma_{\tilde{\epsilon}_1, \tilde{C}_1} \\
        \Sigma X_i = \frac{1}{a_i}( \mu  - c_i \sigma_{\tilde{\epsilon}_1, \tilde{C}_1}) \\
        X_i = \frac{1}{a_i}\Sigma^{-1} (\mu - c_i \sigma_{\tilde{\epsilon}_1, \tilde{C}_1})
    \end{aligned}
$$
\end{enumerate}


We assume that $a_i = a$ for all investors:

$$
    X_i = \frac{1}{a}\Sigma^{-1} ( \mu - c_i \sigma_{\tilde{\epsilon}_1, \tilde{C}_1})
$$
""", unsafe_allow_html=True)
