import streamlit as st 
import numpy as np
import plotly.graph_objs as go
import numpy as np


# Define the function to simulate economic impacts
def simulate_economic_impact(mu, phi, sigma, num_simulations):
    # Generate random shocks from a normal distribution
    C_tilde = np.random.normal(mu, sigma, num_simulations)
    # Calculate economic impact
    return phi * C_tilde

# Function to plot a single histogram using Plotly
def plot_single_histogram(data, title):
    trace = go.Histogram(x=data, opacity=0.75, name=f'phi=1')
    layout = go.Layout(
        title=title,
        xaxis=dict(title='Economic Impact'),
        yaxis=dict(title='Frequency'),
        template='plotly_white',
        bargap=0.2
    )
    fig = go.Figure(data=[trace], layout=layout)
    return fig

# Function to plot comparative histograms using Plotly
def plot_comparative_histogram(data1, data2, phi2, title):
    trace1 = go.Histogram(x=data1, opacity=0.5, name='Model A (phi=1)')
    trace2 = go.Histogram(x=data2, opacity=0.5, name=f'Model B (phi={phi2})')
    layout = go.Layout(
        title=title,
        xaxis=dict(title='Economic Impact'),
        yaxis=dict(title='Frequency'),
        template='plotly_white',
        bargap=0.2,
        barmode='overlay'
    )
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    return fig

# Function to plot a single histogram using Plotly
def plot_utility(data, c_i, title):
    trace = go.Histogram(x=data, opacity=0.75, name=f'c_i={c_i}')
    layout = go.Layout(
        title=title,
        xaxis=dict(title='Impact on Investor i Utility'),
        yaxis=dict(title='Frequency'),
        template='plotly_white',
        bargap=0.2
    )
    fig = go.Figure(data=[trace], layout=layout)
    return fig


st.title("Climate Change Uncertainty")
st.write(r"""
In a one-period model, the economy is hit by a climate
shock at time 1, $\tilde{C}_1$, which is unknown at time 0.
We assume that $\tilde{C}_1$ 
is normally distributed, with $E(\tilde{C}_1) = 0$ and
$\text{Var}(\tilde{C}_1) = 1$. 
""")

st.latex(
    r"""   
\begin{equation}
    \Delta y = \mu_y + \phi \tilde{C}_1
\end{equation}
"""
)

st.write(r"""
             
where $\Delta y$ is the output growth, $\mu_y$ is the expected growth rate 
of the economy, and $\phi$ is a scaling factor that converts the climate shock 
into economic terms. In the case of physical climate risks, $\phi$ could represent
the impact of a climate disaster on the economy (damage function). 
In the case of transition risks, $\phi$ could represent the impact of a policy
change on the economy (e.g. carbon tax).
Climate risks represent the uncertainty about the value of the shock $\tilde{C}_1$.
We do not know the value of the shock, but we know its probability distribution (Normal distribution).
We assume that we know the value of $\phi$.
This is the risk component of uncertainty. 
Risk represent the uncertainty about outcomes, with known probabilities.
         """)


# Sidebar controls
num_simulations = st.sidebar.slider("Number of Simulations", 100, 10000, 1000, key='num_sim')
phi_variable = st.sidebar.slider("Scaling Factor (Phi) for Comparative Plot", 0., 2.0, 1.0, 0.1, key='phi_slider')
c_i_variable = st.sidebar.slider("Investor's Perception of Climate Risks (c_i)", 0., 2.0, 1.0, 0.1, key='c_i_slider')

# Simulation for the first plot
economic_impact_baseline = simulate_economic_impact(0, 1, 1, num_simulations)
fig1 = plot_single_histogram(economic_impact_baseline, 'Distribution of Economic Impact with Phi=1')
st.plotly_chart(fig1)

st.write(r"""
         However, uncertainty is not only about risk. Especially 
in the context of climate change, that requires to model complex dynamic systems.
For example, it may be that alternative models
predict different values of $\phi$. 
This is the ambiguity component of uncertainty. 
Ambiguity refers to the uncertainty associated with how to 
weight alternative models. 
""")

# Simulation for the second plot
economic_impact_variable = simulate_economic_impact(0, phi_variable, 1, num_simulations)
fig2 = plot_comparative_histogram(economic_impact_baseline, economic_impact_variable, phi_variable, 'Comparative Distribution of Economic Impacts')
st.plotly_chart(fig2)

st.write(r"""
         This is the specificity of climate risks, taken into account 
in the PST model. In this model, there is a continuum of investors $i$. 
Investors have different beliefs about future climate risks. 
$c_i \geq 0$ is a scalar that represents the investor's perception 
of the impact of climate risks. If 
$c_i = 0$, the investor does not think that 
climate risks will have an impact on the economy,
or will happen at all. Climate risks enter the investor $i$ utility function
as $c_i \tilde{C}_1$.
Figure below shows the impact of climate shocks 
on the investor $i$ utility. This is exactly the same as the
previous one, representing the ambiguity in the economic impact
of climate shocks. This formulation, with $c_i$ representing the investor's perception
of the impact of climate risks, allows to take into account the ambiguity
component of climate change uncertainty.
""")

utility_impact = simulate_economic_impact(0, c_i_variable, 1, num_simulations)
fig3 = plot_utility(utility_impact, c_i_variable, 'Climate Shocks Impact on Investor i Utility')
st.plotly_chart(fig3)


# st.markdown("""

# ### Ambiguity

# Ambiguity deals with the uncertainty associated with evaluating different models. 
# We may not only lack knowledge about outcomes but also about the underlying probabilities themselves. 
# In our model, ambiguity surrounds the parameters like the variance of the climate shock \(J\) and the scaling factor \(\phi\).

#             """)

# # Plot 2 controls
# mu_user = st.slider("Mean of Shock (mu)", -0.1, 0.1, 0.0, 0.01, key='mu')
# phi_user = st.slider("Scaling Factor (phi)", -1.0, 0., -0.5, 0.1, key='phi')
# sigma_user = st.slider("Standard Deviation of Shock (sigma)", 0.01, 0.5, 0.1, 0.01, key='sigma')
# # Generate user-defined economic impact
# economic_impact_user = simulate_economic_impact(mu_user, phi_user, sigma_user, num_simulations)

# # Plot 2
# st.plotly_chart(plot_histogram([economic_impact_base, economic_impact_user], ['Model A', 'Model B'], 'Comparison of Economic Impact Distributions'))


# st.markdown("""
# ### Misspecification

# Misspecification addresses the inherent limitations of models as approximations. 
# Every model, by necessity, omits certain aspects of reality. 
# In the case of our model, misspecification might arise if, for instance, the climate shock is not actually normally distributed as assumed.
# """)

# # Plot 3 controls
# # Correcting the select box options to match expected values
# distribution_choice = st.selectbox("Distribution Type", ['normal', 'uniform'], key='dist_type')

# # Generate distribution based on user choice
# economic_impact_dist = simulate_economic_impact(mu_user, phi_user, sigma_user, num_simulations, distribution_choice)

# # Plot 3
# st.plotly_chart(plot_histogram([economic_impact_base, economic_impact_dist], ['Model A', f'{distribution_choice.capitalize()} Distribution'], 'Impact of Different Distributions'))
