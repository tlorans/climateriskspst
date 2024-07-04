import streamlit as st 
import plotly.express as px
import plotly.graph_objects as go
import numpy as np 
import pandas as pd
import statsmodels.api as sm


st.title("CAPM with Climate Risks")

st.markdown("""
            
    We have seen that, putting aside the uncertainty regarding 
    the underlying dynamics of climate change (and the risks associated),
    investors face ambiguity regarding the probability distribution
    of the climate risks. We now turn to the implication 
    for investors, with a Capital Asset Pricing Model (CAPM).
    The Pastor-Stambaugh-Taylor model (2021),
    (hereafter PST) extends the CAPM to the case where investors
    have heterogeneous beliefs about climate risks.


    ## Optimal Portfolio with Heterogenous Climate Risk Perception
        
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
c_i = st.slider("Climate Risk Perception ($c_i$)", -1.0, 1.0, 0.0, 0.01)

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

st.markdown("""
            
    ## Heterogeneous Investors and Expected Returns
            
    The market-weighted average climate risks perception $\\bar{c}$ affect 
    asset prices, and leads to a CAPM alphas.      
    In the absence of strong uncertainty about future climate risks
    (that is, if we can recover the 
    exact probability distribution of 
    transition risk, all investors share the same beliefs about it),
    the CAPM alpha is zero, and
    everyone holds the market portfolio. 
    """)

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
bar_c = st.slider("Average Climate Risk Perception", 0.0, 1., 0.5, 0.01)

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

st.markdown("""
    ## Climate Risks Hedging Portfolio
    The climate risks hedging portfolio weights are proportional 
    to the stocks' sensitivities to the climate risks. Investors with $\\gamma_i > 0$, whose climate risks 
    expectation 
    is higher than the market average, 
    will short the hedging portfolio, whereas investors with
    $\\gamma_i < 0$ will go long on the hedging portfolio.
    The allocation fractions $\\phi_i$ are shown in Figure,
    as a function of $\\gamma_i$.

    If the the wealth-weighted average investor's 
    perception of climate risks $\\bar{c} = 0$, 
    then everyone holds the market portfolio.
    If the investor has the same perception as the market, 
    then $\\gamma_i = 0$ and the investor holds the market portfolio.
    For a specific climate risks industry to exist, 
    there must be investors with $\\gamma_i > 0$ and $\\gamma_i < 0$,
    that is, it must exist diversity in the perception of climate risks.
    """)

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
selected_gamma_i = st.slider("Select Deviation from Average Climate Risks Perception", float(gamma_i_values.min()), float(gamma_i_values.max()), 0.0, 0.1)

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

st.markdown("""
## Unexpected Returns and Climate Risks Perception

Pastor et al (2021) extend their model to study the impact of changes in the perception of climate risks (originally related to ESG concerns) on stock prices. In a nutshell, the model incorporates a characteristic \( g_n \) (referred to as "greenness") that may be negatively correlated with the climate betas \( \psi_n \). They use a simple one-period discounted cash flow model where unexpected changes in climate risk perceptions lead to unexpected returns through two channels:

- **Impact through the cash-flows channel**: Changes in perception affect the expected future cash flows of firms.
- **Impact through the discount rate channel**: Changes in perception alter the discount rates applied to future cash flows.

These dynamics drive a wedge between expected and realized returns. If concerns about climate risks strengthen unexpectedly:

- **Green firms**: A firm's unexpected return is expected to be positive for firms with high greenness.
- **Brown firms**: A firm's unexpected return is expected to be negative for firms with low greenness.

Thus, green stocks tend to perform better than expected, and brown stocks perform worse than expected when concerns about climate risks intensify unexpectedly.

    """)
    # Function to compute unexpected returns based on bar_c_1
def compute_unexpected_returns(g, z_c, bar_c_0, bar_c_1, zeta, rho_mC):
    f_c = z_c + (bar_c_1 - bar_c_0) * (1 - rho_mC**2) * zeta
    beta_m = np.zeros_like(g)  # Simplification as given
    z_m = np.random.normal(0, 0.1, 1)  # Market shock
    u = beta_m * z_m + g * f_c + np.random.normal(0, 0.05, len(g))
    expected_returns = beta_m * z_m + g * z_c
    unexpected_returns = u - expected_returns + (bar_c_1 - bar_c_0) * (1 - rho_mC**2) * zeta * g
    return unexpected_returns

# Parameters
num_firms = 50
g = np.linspace(-1, 1, num_firms)  # Environmental scores from -1 (brown) to 1 (green)
zeta = 1
rho_mC = 0.5
bar_c_0 = 0.3  # Initial average climate risks perception
z_c = np.random.normal(0, 0.1, 1)  # Climate shock

# Slider to change the perception of climate risks
change_in_c = st.slider('Change in Climate Risks Perception', 0.0, 0.7, 0.5, 0.01)

bar_c_1 = bar_c_0 + change_in_c  # New average climate risks perception
# Compute unexpected returns based on the current perception
unexpected_returns = compute_unexpected_returns(g, z_c, bar_c_0, bar_c_1, zeta, rho_mC)

# Create Plotly scatter plot
fig = go.Figure(data=[go.Scatter(x=g, y=unexpected_returns, mode='markers',
                                marker=dict(size=12, color=np.sign(g), colorscale='RdYlGn', line=dict(color='Black', width=1)))])

fig.update_layout(title='Impact of Unexpected Strengthening of Climate Risks Concerns on Returns',
                xaxis_title='g',
                yaxis_title='Unexpected Returns',
                plot_bgcolor='white')

st.plotly_chart(fig)

st.markdown("""
        
    We have seen with PST (2021) model that 
    the expected returns of green firms are lower than those of brown firms 
    at equilibrium, because green firms are an hedge against climate risks.
    We have also seen that changes in climate risks perception can lead to
    positive unexpected returns for green firms, and negative unexpected returns
    for brown firms. Therefore, measuring 
    the "true" climate risks premium is a challenging task, as 
    historical data may be mixed with changes in climate risks perception.

Instead of inferring expected returns as the sample average of returns, 
    PST (2022) run a regression of returns on the unexpected changes in climate risks perception.
    Therefore, the expected returns is inferred as the intercept of the regression plus the residuals,
            that is the realized returns purged from the unexpected returns due to unexpected changes in climate risks perception.
""")     

st.markdown("""
Adjust the slider to change the correlation coefficient between simulated returns and climate concerns, and observe how this affects the cumulative returns.
""")

# Slider for the correlation coefficient
correlation_coefficient = st.slider("Correlation Coefficient", -1.0, 1.0, -0.5)

# Simulate climate concerns
np.random.seed(42)  # For reproducibility
x_t = pd.read_excel('data/climateconcerns.xlsx',sheet_name='monthly',index_col='Date')['TRI_monthly'].dropna()
num_periods = len(x_t)
# Generate correlated returns, influenced by x_t
noise = np.random.normal(0, 0.0005, num_periods)  # Additional noise
returns = correlation_coefficient * x_t + noise  # Daily returns influenced by climate concerns

# Prepare the data for regression to get counterfactual returns
x_t_with_const = sm.add_constant(x_t)  # Add constant to the regressor for intercept
model = sm.OLS(returns, x_t_with_const)
results = model.fit()

# Get the regression coefficients
intercept, beta = results.params

# Calculate counterfactual returns (assuming no influence of climate concerns)
counterfactual_returns = intercept + results.resid

# Calculate compounded cumulative returns
cumulative_actual_returns = np.cumprod(1 + returns) - 1
cumulative_counterfactual_returns = np.cumprod(1 + counterfactual_returns) - 1
cumulative_climate_concerns = np.cumsum(x_t)  # Cumulative sum for shocks

# Plotting with Plotly
# Cumulative Climate Concerns Shocks vs. Cumulative Realized Returns
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=np.arange(num_periods), y=cumulative_climate_concerns, mode='lines', name='Cumulative Climate Concerns Shocks'))
fig1.add_trace(go.Scatter(x=np.arange(num_periods), y=cumulative_actual_returns, mode='lines', name='Cumulative Realized Returns'))
fig1.update_layout(title='Cumulative Climate Concerns Shocks vs. Cumulative Realized Returns',
                xaxis_title='Time Period',
                yaxis_title='Cumulative Change',
                template='plotly_white')
st.plotly_chart(fig1)

# Cumulative Actual vs. Counterfactual Returns
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=np.arange(num_periods), y=cumulative_actual_returns, mode='lines', name='Cumulative Realized Returns', line=dict(color='blue')))
fig2.add_trace(go.Scatter(x=np.arange(num_periods), y=cumulative_counterfactual_returns, mode='lines', name='Cumulative Counterfactual Returns', line=dict(color='red')))
fig2.update_layout(title='Cumulative Actual vs. Counterfactual Returns',
                xaxis_title='Time Period',
                yaxis_title='Cumulative Returns',
                template='plotly_white')
st.plotly_chart(fig2)