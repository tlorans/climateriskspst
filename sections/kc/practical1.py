import streamlit as st 
import numpy as np
import plotly.graph_objs as go
import numpy as np


    
# Function to calculate and plot
def plot_exclusion(climate_betas, initial_weights, percentage_to_exclude):
    num_assets = len(climate_betas)
    
    # Calculate the threshold climate beta score
    threshold_index = int(np.ceil((1 - percentage_to_exclude) * num_assets)) - 1
    sorted_betas = np.sort(climate_betas)
    threshold_beta = sorted_betas[threshold_index]

    # Determine which stocks to keep
    stocks_to_keep = climate_betas <= threshold_beta
    
    # Recalculate weights for the remaining stocks
    remaining_weights = initial_weights[stocks_to_keep]
    remaining_weights /= remaining_weights.sum()  # Normalize to sum to 1
    adjusted_weights = np.zeros(num_assets)
    adjusted_weights[stocks_to_keep] = remaining_weights
    
    # Calculate weight deviations
    weight_deviations = adjusted_weights - initial_weights
    
    # Create Plotly graph
    trace = go.Scatter(
        x=climate_betas,
        y=weight_deviations,
        mode='markers',
        marker=dict(color='blue', size=10, line=dict(width=2, color='DarkSlateGrey'))
    )
    
    layout = go.Layout(
        title='Climate Beta vs. Weight Deviation',
        xaxis=dict(title='Climate Beta'),
        yaxis=dict(title='Weight Deviation'),
        shapes=[
            dict(
                type='line',
                x0=threshold_beta,
                y0=min(weight_deviations),
                x1=threshold_beta,
                y1=max(weight_deviations),
                line=dict(
                    color="Red",
                    width=2,
                    dash="dashdot",
                )
            )
        ],
        annotations=[
            dict(
                x=threshold_beta,
                y=0,
                xref='x',
                yref='y',
                text='Threshold: {:.2f}'.format(threshold_beta),
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40
            )
        ]
    )
    
    fig = go.Figure(data=[trace], layout=layout)
    return fig

st.title("Practical Portfolio 1: Manage My Sensitivity!")

st.write(r"""

We propose a simple equivalent of a portfolio allocation between the
market portfolio and the climate risks hedging portfolio from PST (2021)
in a long-only and practical context.

Let $\psi_1, \psi_2, \ldots, \psi_n$ be the climate beta of $n$ assets, where a higher climate beta indicates 
greater exposure to climate risks. The climate betas are ordered such that:
        """)

st.latex(r"""
\begin{equation}
\max(\psi_i) = \psi_{1:n} \geq \psi_{2:n} \geq \ldots \geq \psi_{i:n} \geq \ldots \geq \psi_{n:n} = \min(\psi_i)
\end{equation}
""")

st.write(r"""
Given a percentage $p$ representing the proportion of assets 
to exclude based on high climate risks exposure, 
the threshold climate beta $\psi(p,n)$ is defined as the 
climate beta at the $\lceil pn \rceil$-th position 
from the highest score, where $\lceil \cdot \rceil$
denotes the ceiling function:
""")

st.latex(r"""
\begin{equation}
\psi(p,n) = \psi_{\lfloor p n \rfloor : n}
\end{equation}
""")

st.write(r"""
Assets with a climate beta higher than $\psi(p,n)$ are excluded from the portfolio, i.e., their weights are set to zero:
""")

st.latex(r"""
\begin{equation}
w_i = 0 \quad \text{if} \quad \psi_i > \psi(p,n)
\end{equation}
""")

st.write(r"""
The weights of the remaining assets are then recalculated to sum to one, proportionally based on some baseline weights $b_i$:
""")

st.latex(r"""
\begin{equation}
w'_i = \frac{1\{\psi_i \leq \psi(p,n)\} \cdot b_i}{\sum_{k=1}^n 1\{\psi_k \leq \psi(p,n)\} \cdot b_k}
\end{equation}
""")

st.write(r"""
where $1\{\}$ takes the value 1 if the condition is true, and 0 otherwise.
""")


# Slider for percentage of stocks to exclude
percentage_to_exclude = st.sidebar.slider('Percentage of Stocks to Exclude', 0.0, 1.0, 0.20, 0.01)

# Fixed parameters
num_assets = 20
np.random.seed(42)
climate_betas = 2 * np.random.rand(num_assets) - 1  # Uniform distribution from -1 to 1
initial_weights = np.ones(num_assets) / num_assets

# Generate plot
fig = plot_exclusion(climate_betas, initial_weights, percentage_to_exclude)
st.plotly_chart(fig)
