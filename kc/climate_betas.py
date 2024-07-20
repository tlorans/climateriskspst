import streamlit as st
import numpy as np
import plotly.graph_objs as go
import numpy as np


st.title('Climate Betas')

st.write(r"""
We have seen in the previous sections that climate risks hedging 
portfolio is proportional to the climate risks betas $\psi_n$. Now 
comes the question of how to estimate these climate betas.
        """)

st.markdown("""
## The Narrative Approach
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


# Slider for zeta
zeta = st.slider('Select a value for zeta', min_value=-1., max_value=1., value=1.0, step=0.1)
c_bar = st.slider('Select a value for c_bar', min_value=0., max_value=10.0, value=1.0, step=0.1)


st.latex(r"""
\begin{equation}
    \psi_n = -\zeta g_n
\end{equation}
""")


# Generate sample data
g_n = np.linspace(-1, 1, 100)
mu_m = 1.0
beta_m = 0
rho_mC_squared = 0.5

# Calculate the variables
psi_n = -zeta * g_n
alpha_n = - c_bar * (1 - rho_mC_squared) * zeta * g_n
alpha_n_benchmark = - 0 * (1 - rho_mC_squared) * zeta * g_n

# Create scatter plot between psi_n and g_n
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=g_n, y=psi_n, mode='lines', name='psi_n'))
fig1.update_layout(
    title="Scatter Plot between psi_n and g_n",
    xaxis_title="Environmental Score (g_n)",
    yaxis_title="psi_n",
    legend_title="Variables"
)

st.plotly_chart(fig1)


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
    \alpha_n = - \bar{c}(1 - \rho^2_{mC}) \zeta g_n
\end{equation}
""")

st.write(r"""
The climate risks hedging portfolio will 
therefore be proportional to the climate risks-related 
characteristics
of the stocks $g$.
""")


# Create plot for alpha_n and g_n
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=g_n, y=alpha_n, mode='lines', name='alpha_n'))
fig2.add_trace(go.Scatter(x=g_n, y=alpha_n_benchmark, mode='lines', name='alpha_n (c_bar = 0)', line=dict(dash='dash')))
fig2.update_layout(
    title="Relationship between alpha_n and g_n",
    xaxis_title="Environmental Score (g_n)",
    yaxis_title="alpha_n",
    legend_title="Variables"
)

# Display the plots in Streamlit
st.plotly_chart(fig2)

st.markdown("""
            ## The Mimicking Approach
            """)


st.write(r"""
Another approach is mimicking portfolio from Lamont (2001) [1],
as advocated by Engle *et al.* (2020) [2]. 
We may not observe $\tilde{C}_1$ directly, and 
therefore estimate $\psi_n$ by regressing the unexpected returns
$\tilde{\epsilon}_1$ on the climate risks $\tilde{C}_1$. 
        
But we can observe the change in perception of climate risks 
($\bar{c}_1 - E_0(\bar{c}_1)$) by taking unexpected change of climate concerns 
from news articles, as in Pastor *et al.* (2022). 

We can therefore estimate $\psi_n$ by regressing unexpected returns
$\tilde{\epsilon}_1$ on the change in perception of climate risks 
($\bar{c}_1 - E_0(\bar{c}_1)$).
""")

# Slider for psi_n
psi_n = st.slider('Select a value for psi_n (slope of regression line)', min_value=-2.0, max_value=2.0, value=0.5, step=0.1)

# Generate sample data
np.random.seed(42)
climate_risk_perception_change = np.random.normal(0, 1, 100)
unexpected_returns = psi_n * climate_risk_perception_change + np.random.normal(0, 0.5, 100)

# Create scatter plot with regression line
fig = go.Figure()

# Add scatter plot for data points
fig.add_trace(go.Scatter(x=climate_risk_perception_change, y=unexpected_returns, mode='markers', name='Data Points'))

# Add regression line
x_vals = np.array([-3, 3])
y_vals = psi_n * x_vals
fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='Regression Line', line=dict(color='red')))

# Update layout
fig.update_layout(
    title="Unexpected Returns vs. Change in Climate Risk Perception",
    xaxis_title="Change in Climate Risk Perception (Δc)",
    yaxis_title="Unexpected Returns (ε)",
    legend_title="Legend"
)

# Display the plot in Streamlit
st.plotly_chart(fig)


st.markdown("""
            ## The Quantity-Based Approach
            """)


st.write(r"""
The idea here, exposed by Alekseev et al. (2022), is to exploit changes from 
individual portfolio holdings to estimate climate betas.
        
Going back to the optimal portfolio of investor $i$ in PST (2021), we have:
        """)

st.latex(r"""

\begin{equation}
    X_i = \frac{1}{a}\Sigma^{-1} ( \mu - c_i \sigma_{\tilde{\epsilon}_1, \tilde{C}_1})
\end{equation}
""")

st.write(r"""
The investor $i$ may be affected by an idiocyncratic shock that may change
his perception of climate risks $c_i$. We can then estimate the climate betas
as the change in the optimal portfolio weights following the change in perception
of climate risks.
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
c_i = 0.5
delta_c_i = st.slider("Change in Climate Risk Perception (Δ$c_i$)", 0., 1.0, 0.1, 0.01)

# Calculate the optimal portfolio for the initial c_i
X_i_initial = optimal_portfolio(a, c_i, mu, Sigma, sigma_epsilon_C1)
# Calculate the optimal portfolio for the changed c_i
X_i_changed = optimal_portfolio(a, c_i + delta_c_i, mu, Sigma, sigma_epsilon_C1)

# Estimate climate betas as the change in the optimal portfolio weights
climate_betas = X_i_changed - X_i_initial

# Plot the portfolio weights using Plotly
fig = go.Figure()

# Plot initial portfolio weights
fig.add_trace(go.Scatter(
    x=np.arange(num_assets),
    y=X_i_initial,
    mode='lines+markers',
    name='Initial Portfolio Weights',
    marker=dict(size=10, color='blue', opacity=0.6)
))

# Plot changed portfolio weights
fig.add_trace(go.Scatter(
    x=np.arange(num_assets),
    y=X_i_changed,
    mode='lines+markers',
    name=f'Portfolio Weights after Change in $c_i$',
    marker=dict(size=10, color='red', opacity=0.6)
))

# Plot climate betas
fig.add_trace(go.Bar(
    x=np.arange(num_assets),
    y=climate_betas,
    name='Estimated Climate Betas',
    marker=dict(color='green', opacity=0.6)
))

fig.update_layout(
    title=f'Portfolio Weights and Estimated Climate Betas',
    xaxis_title='Asset',
    yaxis_title='Weight / Beta',
    legend_title="Legend",
    template='plotly_white'
)

# Display the plot in Streamlit
st.plotly_chart(fig)


st.markdown("""

## References
            
[1]: Lamont, O. A. (2001). Economic tracking portfolios. Journal of Econometrics, 105(1), 161-184.
            
[2]: Engle, R. F., Giglio, S., Kelly, B., Lee, H., & Stroebel, J. (2020). Hedging climate change news. The Review of Financial Studies, 33(3), 1184-1216.
            
            """)