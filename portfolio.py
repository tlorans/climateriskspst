import streamlit as st 
import numpy as np
import plotly.graph_objs as go
import numpy as np
import pandas as pd

def simulate_portfolio_adjustment(num_assets, percentage_to_exclude):
    # Generate synthetic greenness scores and initial weights
    np.random.seed(42)  # for reproducibility
    greenness_scores = 2 * np.random.rand(num_assets) - 1  # Uniform distribution from -1 to 1
    initial_weights = np.ones(num_assets) / num_assets  # Equally weighted initially

    # Calculate the threshold greenness score for exclusion
    threshold_index = int(np.floor(percentage_to_exclude * num_assets))
    threshold_score = np.sort(greenness_scores)[threshold_index]

    # Adjust weights based on greenness score
    stocks_to_keep = greenness_scores >= threshold_score
    remaining_weights = initial_weights[stocks_to_keep]
    remaining_weights /= remaining_weights.sum()  # Normalize to sum to 1
    adjusted_weights = np.zeros(num_assets)
    adjusted_weights[stocks_to_keep] = remaining_weights

    # Calculate weight deviations
    weight_deviations = adjusted_weights - initial_weights

    return greenness_scores, weight_deviations, threshold_score

def plot_adjustments(greenness_scores, weight_deviations, threshold_score):
    # Creating the Plotly plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=greenness_scores,
        y=weight_deviations,
        mode='markers',
        marker=dict(size=10, color='blue'),
        name='Weight Deviation'
    ))
    
    # Add a line for the threshold
    fig.add_shape(
        type='line',
        x0=threshold_score, y0=-0.05, x1=threshold_score, y1=0.05,
        line=dict(color='Red', width=2, dash='dash'),
        name='Threshold'
    )

    fig.update_layout(
        title='Deviation of Portfolio Weights by Asset Greenness',
        xaxis_title='Greenness Score (g)',
        yaxis_title='Weight Deviation',
        showlegend=False
    )
    return fig

# Function to calculate compounded cumulative returns
def compound_returns(returns):
    return np.cumprod(1 + returns) - 1

# Function to simulate and plot returns
def simulate_and_plot_returns(specified_weight, x_t):
    num_periods = len(x_t)
    noise_portfolio = np.random.normal(0, 0.005, num_periods)
    noise_climate = np.random.normal(0, 0.005, num_periods)

    # Simulating the returns
    portfolio_returns = -0.5 * x_t + noise_portfolio
    climate_risk_returns = 0.3 * x_t + noise_climate

    # Calculate hedged portfolio returns with the specified weight
    hedged_returns = portfolio_returns + specified_weight * climate_risk_returns

    # Compounded cumulative returns
    cum_portfolio_returns = compound_returns(portfolio_returns)
    cum_climate_risk_returns = compound_returns(climate_risk_returns)
    cum_hedged_returns = compound_returns(hedged_returns)

    # Plotting with Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_t.index, y=cum_portfolio_returns, mode='lines', name='Original Portfolio'))
    fig.add_trace(go.Scatter(x=x_t.index, y=cum_climate_risk_returns, mode='lines', name='Climate Risk Portfolio', line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=x_t.index, y=cum_hedged_returns, mode='lines', name='Hedged Portfolio', line=dict(dash='dash')))

    fig.update_layout(title='Compounded Cumulative Returns Comparison',
                      xaxis_title='Date',
                      yaxis_title='Compounded Cumulative Returns',
                      template='plotly_white')

    return fig

st.markdown("""
# Practical Implications for Portfolio Construction

## A Simple Exclusion Strategy

We propose an approach that approximates portfolio allocation between the market portfolio and the climate risks hedging portfolio 
in a long-only and practical context. This method is inspired by the PST (2021) model.

Let's consider we have `n` assets, each with a "greenness" score. 

**Step-by-step explanation**:

1. **Ordering Scores**: Arrange the greenness scores from the lowest to the highest.
2. **Exclusion Threshold**: Decide on a percentage `p` of assets to exclude based on their poor greenness scores. The threshold score is the score at the position which is `p` percent from the bottom of the ordered list.
3. **Exclusion of Assets**: Assets with a greenness score below this threshold are excluded from the portfolio. Their weights are set to zero.
4. **Weight Recalculation**: The weights of the remaining assets are adjusted to sum up to one, proportionally based on their original weights.

Providing that the greeness score 
is a good proxy for exposure to climate risks, this strategy allows investors to reduce their exposure to climate risks.
            
""")

num_assets = st.number_input('Number of Assets', min_value=10, max_value=100, value=20, step=1)
percentage_to_exclude = st.slider('Percentage of Stocks to Exclude', 0.0, 0.5, 0.2, 0.05)

greenness_scores, weight_deviations, threshold_score = simulate_portfolio_adjustment(num_assets, percentage_to_exclude)
fig = plot_adjustments(greenness_scores, weight_deviations, threshold_score)

st.plotly_chart(fig)

st.markdown("""
## Managing Exposure to Unexpected Returns

Based on empirical findings from Pastor *et al.* (2022), 
investors might wish to manage exposure to unexpected returns that could arise from shifts in market perception of climate risks. 
This can be particularly relevant for investors whose own perceptions of climate risks may differ from the market consensus.

Here's a straightforward strategy for hedging:

- **Baseline Portfolio**: Consider your initial portfolio which is assumed to be negatively correlated with climate risk concerns.
- **Climate Risk Hedging Portfolio**: Compose this portfolio from assets with either the lowest climate risk betas or the highest greenness scores.
- **Combining Portfolios**: Mix the initial portfolio with the climate risk hedging portfolio.

**Formula Representation**:
The total returns of the hedged portfolio at any given time \( t \) are the sum of the returns from the baseline portfolio and a 
fraction of the climate risk hedging portfolio's returns, weighted by \( \omega \). Here, \( \omega \) is determined based on the 
investor's personal view of climate risks.

The performance of this hedged portfolio is tracked by calculating its compounded cumulative returns over time.
""")

# Load data
x_t = pd.read_excel('data/climateconcerns.xlsx', sheet_name='monthly', index_col='Date')['TRI_monthly'].dropna()

# Sidebar to control the weight omega
specified_weight = st.slider('Weight for Climate Risk Portfolio (omega)', 0.0, 1.0, 0.9, 0.05)

# Generate plot
fig = simulate_and_plot_returns(specified_weight, x_t)
st.plotly_chart(fig)