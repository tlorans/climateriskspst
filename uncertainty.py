import streamlit as st 
import numpy as np
import plotly.graph_objs as go
import numpy as np

def simulate_economic_impact(mu, phi, sigma, num_simulations, distribution_type='normal'):
    if distribution_type == 'normal':
        J = np.random.normal(mu, sigma, num_simulations)
    elif distribution_type == 'uniform':
        J = np.random.uniform(mu - 3*sigma, mu + 3*sigma, num_simulations)
    else:
        # Provide a default behavior or raise an error for unsupported types
        J = np.random.normal(mu, sigma, num_simulations)  # Default to normal distribution
    economic_impact = phi * J
    return economic_impact


# Function to plot histograms
def plot_histogram(data, labels, title):
    traces = []
    for i, d in enumerate(data):
        traces.append(go.Histogram(x=d, opacity=0.75, name=labels[i]))
    layout = go.Layout(barmode='overlay',
                       title=title,
                       xaxis=dict(title='Economic Impact'),
                       yaxis=dict(title='Frequency'),
                       template='plotly_white')
    fig = go.Figure(data=traces, layout=layout)
    return fig


st.markdown("""
# Climate Change Uncertainty

Climate economic models often feature random shocks as a source of uncertainty. 
However, as noted by Brock and Hansen (2017), there are other sources of uncertainty. 
For example, we may not know parameters within a given model or we may not know which among 
alternative model specifications gives the best or most reliable answers to the question. 
Because climate economics models are dynamic and the potential consequences of emissions are long-lasting, 
the impact of random shocks and model uncertainty compounds over time. Last but not least, 
the quantitative models used are approximations of the underlying physical, biological, and economic systems dynamics, 
and are therefore necessarily misspecified. Following Brock and Hansen (2017), we can distinguish three types of uncertainty:

1. **Risk:** What probabilities does a specific model assign to events in the future?
2. **Ambiguity:** How much confidence do we place in each model?
3. **Misspecification:** How do we use models that are not perfect?

For pedagogical purposes, we will explore the implications of each type of uncertainty in the context of climate change using a simplified 
version of the illustrative from Giglio *et al.* (2021).

## Simple One Period Model

In this simple one-period model, the economy is affected by a climate shock \(J\), and \(\phi\) is a scaling factor that translates this shock into economic impact:

**Output change = Expected economic growth + Impact of climate shock**

Where:
- **Output change** represents the change in economic output.
- **Expected economic growth** is the anticipated growth rate of the economy.
- **Impact of climate shock** is calculated as the product of the shock \(J\) and the scaling factor \(\phi\). 
The shock \(J\) itself is a variable drawn from a normal distribution with mean zero and a specified variance.

### Risk

Risk represents the uncertainty about outcomes, rather than the probabilities. 
It encapsulates the probabilities inferred by a model given its parameters. 
Known probability shocks within a model are termed risks. Outcomes, decided by random draws from these distributions, are unknown; 
however, their probabilities are known. 
In the context of our model, risk specifically pertains to the understood probability distribution of the climate shock \(J\), which represents stochastic climate-related disasters.
""")

num_simulations = st.slider("Number of Simulations", 100, 1000, 500, 50, key='sim1')
# Fixed parameters for Plot 1
mu_base = 0.0
phi_base = -0.5
sigma_base = 0.1

# Generate baseline economic impact
economic_impact_base = simulate_economic_impact(mu_base, phi_base, sigma_base, num_simulations)

# Plot 1
st.plotly_chart(plot_histogram([economic_impact_base], ['Model A'], 'Baseline Distribution of Economic Impact'))


st.markdown("""

### Ambiguity

Ambiguity deals with the uncertainty associated with evaluating different models. 
We may not only lack knowledge about outcomes but also about the underlying probabilities themselves. 
In our model, ambiguity surrounds the parameters like the variance of the climate shock \(J\) and the scaling factor \(\phi\).

            """)

# Plot 2 controls
mu_user = st.slider("Mean of Shock (mu)", -0.1, 0.1, 0.0, 0.01, key='mu')
phi_user = st.slider("Scaling Factor (phi)", -1.0, 0., -0.5, 0.1, key='phi')
sigma_user = st.slider("Standard Deviation of Shock (sigma)", 0.01, 0.5, 0.1, 0.01, key='sigma')
# Generate user-defined economic impact
economic_impact_user = simulate_economic_impact(mu_user, phi_user, sigma_user, num_simulations)

# Plot 2
st.plotly_chart(plot_histogram([economic_impact_base, economic_impact_user], ['Model A', 'Model B'], 'Comparison of Economic Impact Distributions'))


st.markdown("""
### Misspecification

Misspecification addresses the inherent limitations of models as approximations. 
Every model, by necessity, omits certain aspects of reality. 
In the case of our model, misspecification might arise if, for instance, the climate shock is not actually normally distributed as assumed.
""")

# Plot 3 controls
# Correcting the select box options to match expected values
distribution_choice = st.selectbox("Distribution Type", ['normal', 'uniform'], key='dist_type')

# Generate distribution based on user choice
economic_impact_dist = simulate_economic_impact(mu_user, phi_user, sigma_user, num_simulations, distribution_choice)

# Plot 3
st.plotly_chart(plot_histogram([economic_impact_base, economic_impact_dist], ['Model A', f'{distribution_choice.capitalize()} Distribution'], 'Impact of Different Distributions'))
