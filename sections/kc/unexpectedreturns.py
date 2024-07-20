import streamlit as st 
import numpy as np
import plotly.graph_objs as go
import numpy as np


def plot_with_changes(psi, rho_mC, z_c, delta_c):
    # Compute unexpected returns for various scenarios
    unexpected_returns_no_change = psi * (0 - (0.5 - 0.5) * (1 - rho_mC**2))
    unexpected_returns_cashflow = psi * (z_c - (0.5 - 0.5) * (1 - rho_mC**2))
    unexpected_returns_discount = psi * (0 - (delta_c) * (1 - rho_mC**2))

    # Create traces for Plotly
    trace1 = go.Scatter(x=psi, y=unexpected_returns_no_change, mode='lines', name='No Shock or Change in Perception')
    trace2 = go.Scatter(x=psi, y=unexpected_returns_cashflow, mode='lines', name='Climate Cash-flow Channel')
    trace3 = go.Scatter(x=psi, y=unexpected_returns_discount, mode='lines', name='Climate Discount Rate Channel')
    
    # Plot layout
    layout = go.Layout(
        title='Impact of Unexpected Strengthening of Climate Risks Concerns on Returns',
        xaxis_title='Climate Beta',
        yaxis_title='Unexpected Returns',
        template='plotly_white'
    )
    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    return fig


st.title("Unexpected Returns: a Consequence of Resolving Climate Change Uncertainty")

st.write(r"""

The previous model can be extended
to study the impact of change in perception of climate risks on stock prices. 
It may reflects the 
progressive learning about climate change by investors, as new information
becomes available. 
        """)

st.image('images_tick-1.png', caption='One-Period Overlapping Generation Model')

st.write(r"""    
PST adopts a one-period overlapping generation 
(OLG) model, 
with two generations, 
$Gen-0$ and $Gen-1$. $Gen-0$ is 
born at time 0 and invests in 
the stock of a firm, 
which it sells to $Gen-1$ 
at the beginning of period 1 ($1^{-}$). 
$Gen-0$ dies at the beginning of period 1 
($1^{-}$), while $Gen-1$ 
is born at the beginning of period 1 
($1^{-}$) and dies at the end of period 1 ($1^{+}$). 
Figure above
shows the timeline of the model.

To model the cash-flows channel, we need to model firm 
profits. We denote $\tilde{u}_{1,n}$ the payoff (profit 
in the one-period setting) that firm $n$ produces at 
time $1$. We assume a simple two-factor structure for the 
$N \times 1$ vector of payoffs:
""")

st.latex(r"""
\begin{equation}
    \tilde{u}_1 - E_0(\tilde{u}_1) = \tilde{z}_{m} \beta_m + \tilde{z}_{c} \psi_n + \tilde{\eta}
\end{equation}
""")

st.write(r"""
where $\tilde{z}_{m}$, $\tilde{z}_{c}$ and $\tilde{\eta}$ have zero 
means and are uncorrelated with each other. The shock $\tilde{z}_{m}$
is a macro output factor, with firms' sensitivities proportional to their 
stocks' market betas $\beta_m$. 
The shock $\tilde{z}_{c}$ represents a climate risks shock. 
A positive value of $\tilde{z}_{c}$
shock increases the payoff of firms with a negative climate beta $\psi_n$,
and hurts firms with a positive climate beta $\psi_n$.
The shock $\tilde{\eta}$ is a firm-specific shock (idiosyncratic).

To model the investor channel, 
we assume that the average climate risks perception 
$\bar{c}$ shifts unpredictably from time 0 to time 1.
We therefore also need to price stocks at time 1, 
after the preference shift in $\bar{c}$ occurs.
The price $p_1$ is calculated at $1^{-}$ when
shocks associated with $\tilde{u}_1$ 
have been realized. Therefore, 
between $1^{-}$ and $1^{+}$, the payoff is
riskless (everything is known).
Stockholders will 
receive the payoff at $1^{+}$.
$Gen-0$ sell stocks to $Gen-1$ investors at price $p_1$ 
that depends on $Gen-1$ perception of climate risks $\bar{c}_1$ 
and the payoff $\tilde{u}_1$.
We compute the price of the stock, assuming $\beta_m = 0$ for simplicity:
""")

st.latex(r"""
\begin{equation}
    \begin{aligned}
    p_{1,n} = \frac{\tilde{u}_n}{1 + \mu} \\
    = \frac{\tilde{u}_n}{1 + (\mu_m \beta_m + \bar{c}_1(1 - \rho^2_{mC})\psi_n)} \\
    = \frac{\tilde{u}_n}{1 + \bar{c}_1(1 - \rho^2_{mC})\psi_n}
    \end{aligned}
\end{equation}
        """)
st.write(r"""
Which can be approximated: [1]
""")

st.latex(r"""
        \begin{equation}
    p_{1,n} \approx \tilde{u}_n - \bar{c}(1 - \rho^2_{mC})\psi_n
\end{equation}
""")

st.write(r"""
        Taking the expectation at time 0:
""")

st.latex(r"""
\begin{equation}
    E_0(p_1) = E_0(\tilde{u}) - E_0(\bar{c}_1)(1 - \rho^2_{mC})\psi
\end{equation}
""")

st.write(r"""
        The unexpected returns for $Gen-0$ is: [2]
""")

st.latex(r"""
\begin{equation}
    \begin{aligned}
    r_1 - E_0(r_1) = \tilde{u} - \bar{c}_1(1 - \rho^2_{mC})\psi - E_0(\tilde{u}) + E_0(\bar{c}_1)(1 - \rho^2_{mC})\psi\\
    = \tilde{u} - E_0(\tilde{u}) - \bar{c}_1(1 - \rho^2_{mC})\psi + E_0(\bar{c}_1)(1 - \rho^2_{mC})\psi \\
    = \tilde{u} - E_0(\tilde{u}) - (\bar{c}_1 - E_0(\bar{c}_1))(1 - \rho^2_{mC})\psi \\ 
    = \beta_m \tilde{z}_m + \psi \tilde{z}_c + \tilde{\eta} - (\bar{c}_1 - E_0(\bar{c}_1))(1 - \rho^2_{mC})\psi \\
    = \beta_m \tilde{z}_m + \psi \tilde{f}_c + \tilde{\eta}
    \end{aligned}
\end{equation}
        """)

st.write(r"""
where $\tilde{f}_c = \tilde{z}_c - (\bar{c}_1 - E_0(\bar{c}_1))(1 - \rho^2_{mC})$.

The two channels of risk in climate risks are identified: 
$\tilde{z}_c$ represents the cash-flow channel, while the other term 
represents the discount rate channel.

This result drive a wedge between expected and realized returns for $Gen-0$ investors. 
If climate risks concerns strenghen unexpectedly, so that $\tilde{f}_c > 0$ 
(ie. $\bar{c}_1 > E_0(\bar{c}_1)$).
        """)

# Sidebar for climate shock
z_c = st.sidebar.slider('Climate Shock (z_c)', -1.0, 0.0, -1.0, key='z_c_slider')

# Sliders for change in perception
bar_c_0 = 0.5
delta_c = st.sidebar.slider('Increase in Perception of Climate Risk', 0.0, 2.0, 1.5, key='delta_c_slider')

# Fixed correlation factor
rho_mC = 0.5

# Parameters
num_firms = 50
psi = np.linspace(-1, 1, num_firms)

# Generate plot
fig = plot_with_changes(psi, rho_mC, z_c, delta_c)
st.plotly_chart(fig)

st.write(r"""
A firm's unexpected return is expected to be positive for firms with 
negative $\psi_n$. 
It is expected to be negative for firms with positive $\psi_n$.
        """)

# Footnote explanation
st.write(r"""
---  
**Footnotes:**  
[1] Where we follow the approximation from Pastor et al. (2021):

$$
\begin{aligned}
\frac{1 + \rho_1}{1 - \rho_2} &= \frac{(1 + \rho_1)(1 + \rho_2)}{1 - \rho_2^2} \\
&\approx (1 + \rho_1)(1 + \rho_2) \approx 1 + \rho_1 + \rho_2
\end{aligned}
$$

If we define $\rho_1 := \tilde{u}_n - 1$ and $\rho_2 := - \bar{c}_1(1 - \rho^2_{mC})\psi_n$, we have:

$$
\begin{aligned}
1 + \rho_1 + \rho_2  &= 1 + \tilde{u}_n - 1 - \bar{c}_1(1 - \rho^2_{mC})\psi_n \\
&= \tilde{u}_n - \bar{c}_1(1 - \rho^2_{mC})\psi_n
\end{aligned}
$$
        
[2] We follow Pastor et al. (2021), 
    where in a one-period model, unexpected loss price 
    is equal to unexpected returns:
    $$
    p_1 - E_0(p_1) = r_1 - E_0(r_1)
    $$
""", unsafe_allow_html=True)

