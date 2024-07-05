import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import statsmodels.api as sm


# Assuming you have 'climateconcerns.xlsx' available in the correct directory
def load_data():
    return pd.read_excel('data/climateconcerns.xlsx', sheet_name='monthly', index_col='Date')['TRI_monthly'].dropna()

def calculate_returns(x_t, correlation):
    num_periods = len(x_t)
    noise = np.random.normal(0, 0.0001, num_periods)  # Additional noise
    returns = correlation * x_t + noise  # Correlation controlled by user
    return returns

def regression_analysis(x_t, returns):
    x_t_with_const = sm.add_constant(x_t)  # Add constant for intercept
    model = sm.OLS(returns, x_t_with_const)
    results = model.fit()
    intercept, beta = results.params
    counterfactual_returns = intercept + results.resid
    return counterfactual_returns, results

def compounded_returns(returns):
    return np.cumprod(1 + returns) - 1

def create_plots(x_t, actual_returns, counterfactual_returns):
    # Create cumulative concerns shocks plot
    shock_trace = go.Scatter(
        x=x_t.index, 
        y=compounded_returns(x_t),
        mode='lines',
        name='Cumulative Climate Concerns Shocks'
    )
    
    # Cumulative Realized Returns
    actual_trace = go.Scatter(
        x=x_t.index,
        y=actual_returns,
        mode='lines',
        name='Cumulative Realized Returns'
    )
    
    # Cumulative Counterfactual Returns
    counterfactual_trace = go.Scatter(
        x=x_t.index,
        y=counterfactual_returns,
        mode='lines',
        name='Cumulative Counterfactual Returns',
        line=dict(color='red')
    )

    fig1 = go.Figure(data=[shock_trace, actual_trace])
    fig1.update_layout(title='Cumulative Climate Concerns vs. Realized Returns',
                       xaxis_title='Time',
                       yaxis_title='Cumulative Returns')
    
    fig2 = go.Figure(data=[actual_trace, counterfactual_trace])
    fig2.update_layout(title='Cumulative Realized vs. Counterfactual Returns',
                       xaxis_title='Time',
                       yaxis_title='Cumulative Returns',
                       yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='Black'))

    return fig1, fig2

@st.cache_data
def generate_returns(x_t):
    num_periods = len(x_t)
    noise_portfolio = np.random.normal(0, 0.001, num_periods)  # Additional noise for the portfolio
    noise_climate = np.random.normal(0, 0.001, num_periods)  # Additional noise for the climate risk portfolio

    portfolio_returns = -0.2 * x_t + noise_portfolio  # Portfolio returns negatively correlated with x_t
    climate_risk_returns = 0.4 * x_t + noise_climate  # Climate risk portfolio returns positively correlated with x_t
    
    return portfolio_returns, climate_risk_returns

st.title("Practical Implication 2: Make My Portfolio Great Again!")

st.write(r"""
The PST model predicts that stocks with negative climate betas 
have lower expected returns than stocks with positive climate betas
(climate alpha). The resulting climate risks premium increases 
as the market's concerns about climate risks increase.
We have also seen that the PST also explain that stocks 
with negative climate betas can have higher realized returns 
while the market's concerns about climate risks increase. 
This wedge between expected and realized returns is central to the PST model.

Armed with the previous analytical results, 
Pastor et al. (2022) have shown that outperformance 
of "green" stocks over "brown" stocks during the last decade 
likely reflects an unanticipated 
increase in the market's concerns about climate risks.
Their approach addresses the problem of inferring an asset's unconditional
expected return from historical data $\mu = E(r_t)$ using 
ex post data. To measure it, they are two types of estimators:
- $\bar{r}$, the asset's sample average return 
- $\hat{a} = \bar{r} - \hat{b} \bar{x}$, where the estimated $\hat{a}$ and $\hat{b}$ are obtained from a regression of the asset's return on additional 
information $x_t$:
         """)

st.latex(r"""
\begin{equation}
r_t = a + b x_t + \epsilon_t
\end{equation}
""")

st.write(r"""
and $\bar{x}$ is the sample average of $x_t$.
""")

st.write(r"""
In the second approach, $x_t$ is the unanticipated change in 
perception of climate risks, with $E(x_t) = 0$. From PST (2022), 
$a = \mu$ because $x_t$ has zero mean ex ante. Therefore, the idea 
is to estimate $\mu$ by the sample estimate of $a$. 
To proxy for unexpected changes in climate risks perception, Pastor \textit{et al.} (2022)
uses a sentiment index constructed from news articles, from Ardia \textit{et al.} (2021).
         
To get a sense of the analysis, we simulate a fictional 
portfolio realized returns with a positive correlation to climate transition concern. 
This simulate a long-only version of the climate risks hedging portfolio from the 
theoretical section, formed as a long-only position in the assets with 
the lowest climate betas $\psi_n$.
Figure below shows the cumulative shocks on a climate transition concern index (Apel \textit{et al.}, 2023) 
and the cumulative returns of a simulated assets with negative correlation between climate transition concerns and asset returns.
""")


# Load data
x_t = load_data()

# Slider for correlation adjustment
correlation = st.sidebar.slider('Correlation with Climate Concerns', 0., 1.0, 0.6, 0.01)

# Compute returns based on the selected correlation
returns = calculate_returns(x_t, correlation)
counterfactual_returns, results = regression_analysis(x_t, returns)
actual_returns = compounded_returns(returns)
counterfactual_returns = compounded_returns(counterfactual_returns)

# Generate plots
fig1, fig2 = create_plots(x_t, actual_returns, counterfactual_returns)

# Display plots
st.plotly_chart(fig1)
st.plotly_chart(fig2)

st.write(r"""

Then, the idea is to estimate a counterfactual asset return, 
which is the asset's return in the absence of changes in climate risks perception
(estimated as $\hat{a} + \epsilon_t$). As we see in Figure \ref{fig:counterfactual},
the counterfactual portfolio turned to be negative for most of the period,
illustrating the notion that expected returns (counterfactual returns here)
are negative for stocks with negative climate betas. In a period 
of increasing concerns about climate risks, the cumulative climate risks 
hedging portfolio's realized returns are substantial and positive, 
reflecting the cumulative impact of the unexpected increase in climate risks
concerns on the portfolio's returns.


By contrast, an investor's portfolio may be negatively correlated 
with climate risks concerns. It may be the case for portfolio tilted 
towards value stocks, which are negatively correlated with climate concerns 
as shown by Pastor \textit{et al.} (2022). In this case, the investor's
portfolio may have underperformed during the last decade,
as the market's concerns about climate risks increased.
A simple core-satellite approach may be used to hedge 
unexpected changes in climate risks perception.
         
         """)

st.write(r"""
Let $R_t$ be the returns of the investor's portfolio at time $t$, 
and $C_t$ be the returns of the climate risk hedging portfolio 
at the same time. 
The returns $R_t$ are assumed to be negatively 
correlated with climate risk concerns, whereas $C_t$ 
are positively correlated.

A specified non-negative weight $\omega$ 
is applied to the climate risk hedging portfolio 
to form the hedged portfolio. 
The returns of the hedged portfolio $H_t$ at time $t$ are given by:
""")

st.latex(r"""
\begin{equation}
         H_t = (1 - \omega) R_t + \omega C_t
\end{equation}
""")

st.write(r"""
$\omega$ may be chosen as a function of the investor's own perception 
of climate risks, or future change in climate risks perception. $R_t$ 
acts as the core of the portfolio, while $C_t$ acts as the satellite.
         """)

# Slider for weight adjustment
omega = st.sidebar.slider('Weight for Climate Risk Mimicking Portfolio', 0.0, 1.0, 0.9, 0.01)

# Generate returns
x_t = load_data()
portfolio_returns, climate_risk_returns = generate_returns(x_t)

# Calculate hedged portfolio returns with the specified weight
hedged_returns = (1 - omega) * portfolio_returns + omega * climate_risk_returns

# Calculate compounded cumulative returns
cumulative_portfolio_returns = compounded_returns(portfolio_returns)
cumulative_climate_risk_returns = compounded_returns(climate_risk_returns)
cumulative_hedged_returns = compounded_returns(hedged_returns)


# Create Plotly graphs
trace1 = go.Scatter(
    x=x_t.index,
    y=cumulative_portfolio_returns,
    mode='lines',
    name='Original Portfolio Compounded Cumulative Returns'
)
trace2 = go.Scatter(
    x=x_t.index,
    y=cumulative_climate_risk_returns,
    mode='lines',
    name='Climate Risk Portfolio Compounded Cumulative Returns',
    line=dict(dash='dash')
)
trace3 = go.Scatter(
    x=x_t.index,
    y=cumulative_hedged_returns,
    mode='lines',
    name='Hedged Portfolio Compounded Cumulative Returns',
    line=dict(dash='dot')
)

# Plot layout
layout = go.Layout(
    title='Compounded Cumulative Returns Comparison with Specified Weight',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Compounded Cumulative Returns'),
    template='plotly_white'
)

fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)

# Display plot
st.plotly_chart(fig)