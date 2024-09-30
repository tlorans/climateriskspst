import streamlit as st
import numpy as np
import plotly.graph_objs as go
import sympy as sp
import sympy.stats as stats

st.title('Climate Risks and Stock Returns')


st.subheader('Characteristics and Cross Section of Stock Returns')


st.write(r"""
The asset pricing literature has found that stock returns across different firms can be explained by their characteristics. 
These characteristics may act as proxies for underlying risk factors, helping to identify stocks that are more likely to perform poorly during bad times, thus carrying higher expected returns.

Fama and French (2015) use the dividend discount model to link firm characteristics—valuation, profitability, and investment—with stock returns. The following equation represents the model:
""")

st.latex(r'''
            \begin{equation}
         M_t = \frac{\mathbb{E}(Y_{t+1}) - \mathbb{E}(dB_{t+1})}{1+r}
        \end{equation}
            ''')

st.write(r"""
Where:
- $M_t$: Market value of equity,
- $Y_{t+1}$: Expected future earnings,
- $dB_{t+1}$: Expected change in book value of equity,
- $r$: Expected return.

By solving for $r$, we obtain the expected return:
""")


# Define the equation: r = (Y_{t+1} - dB_{t+1}) / M_t - 1
M_t, Y_t1, dB_t1, r = sp.symbols('M_t Y_t1 dB_t1 r')
ddm_eq = sp.Eq(r, (Y_t1 - dB_t1) / M_t - 1)

st.latex(r'''
            \begin{equation}
            r = \frac{Y_{t+1} - dB_{t+1}}{M_t} - 1
            \end{equation}
         ''')


default_M = 20
default_Y_t1 = 30
default_dB_t1 = 5

# User inputs for the equation
M_t_val = st.sidebar.number_input('Case 1: Market value of equity ($M_t$)',min_value=15, max_value=25, value=default_M, step=1)
Y_t1_val = st.sidebar.number_input('Case 2: Expected future earnings ($Y_{t+1}$)',min_value = 25, max_value = 35, value=default_Y_t1, step = 1)
dB_t1_val = st.sidebar.number_input('Case 3: Expected change in book value of equity ($dB_{t+1}$)', value=default_dB_t1)

# Substitute user input values into the equation
ddm_numeric_case_1 = ddm_eq.subs({M_t: M_t_val, Y_t1: default_Y_t1, dB_t1: default_dB_t1})

# Solve for r (expected return)
expected_return_case_1 = sp.solve(ddm_numeric_case_1, r)[0]


st.write(r"""
This model suggests the following relationship between expected returns and firm characteristics:
1. **Valuation**: If we hold everything else constant and only vary the market value ($M_t$), a lower $M_t$ (or a higher book-to-market ratio) implies a higher expected return.
""")


# Construct the LaTeX expression using the actual values
latex_eq = r"\begin{equation} r = \frac{" + str(default_Y_t1) + " - " + str(default_dB_t1) + "}{" + str(M_t_val) + "} - 1 =" + str(round(expected_return_case_1,2)) +"\end{equation}" 

# Display the equation in LaTeX format in Streamlit
st.latex(latex_eq)

st.write(r"""
For example, Zhang (2005) explains the value premium by arguing that value firms are more affected by bad economic times, as their assets are harder to adjust compared to growth firms. As a result, value firms tend to offer higher expected returns due to their greater exposure to risks in bad times.
""")

st.write(r"""
2. **Profitability**: If we fix the market value ($M_t$) and only vary the expected future earnings ($Y_{t+1}$), higher expected earnings imply a higher expected return.
""")


# Substitute user input values into the equation
ddm_numeric_case_2 = ddm_eq.subs({M_t: default_M, Y_t1: Y_t1_val, dB_t1: default_dB_t1})

# Solve for r (expected return)
expected_return_case_2 = sp.solve(ddm_numeric_case_2, r)[0]


# Construct the LaTeX expression using the actual values
latex_eq = r"\begin{equation} r = \frac{" + str(Y_t1_val) + " - " + str(default_dB_t1) + "}{" + str(default_M) + "} - 1 =" + str(round(expected_return_case_2,2)) +"\end{equation}"

# Display the equation in LaTeX format in Streamlit
st.latex(latex_eq)

st.write(r"""
While profitability might intuitively suggest lower risk, firms with higher profitability often have more of their cash flow far into the future, making these cash flows more uncertain. Additionally, profitable firms may attract competition, which can threaten future profit margins and increase risk.
""")

st.write(r"""
3. **Investment**: If we hold the market value and expected earnings constant, higher expected growth in book equity (investment) implies a lower expected return.
""")

# Substitute user input values into the equation
ddm_numeric_case_3 = ddm_eq.subs({M_t: default_M, Y_t1: default_Y_t1, dB_t1: dB_t1_val})

# Solve for r (expected return)
expected_return_case_3 = sp.solve(ddm_numeric_case_3, r)[0]

# Construct the LaTeX expression using the actual values
latex_eq = r"\begin{equation} r = \frac{" + str(default_Y_t1) + " - " + str(dB_t1_val) + "}{" + str(default_M) + "} - 1 =" + str(round(expected_return_case_3,2)) +"\end{equation}"

# Display the equation in LaTeX format in Streamlit
st.latex(latex_eq)

st.write(r"""
This relationship suggests that firms investing heavily to sustain profits may have lower free cash flow available for investors, leading to lower expected returns compared to firms with lower investment needs.
""")


st.subheader('Could Climate Risks-Related Characteristics Explain Stock Returns?')


st.write(r"""
The literature on climate finance discusses two broad categories of climate-related risks:
1. **Physical Climate Risk**: Risks arising from the direct impact of climate change on assets, such as sea level rise or extreme weather events damaging production facilities.
2. **Transition Risk**: Risks associated with the transition to a low-carbon economy, such as carbon taxes that may reduce the profitability of fossil fuel companies.

In this course, we will focus on transition risks.
Assets may have different exposures to these risks, meaning that climate risk realizations could create both winners and losers in the market. For example, while coal companies might suffer from transition risks, renewable energy companies may benefit.

Similar to traditional firm characteristics, we can expect that firms more exposed to climate risks may also exhibit higher expected returns, as investors require compensation for these additional risks.
""")


st.write(r"""
Görgen $\textit{et al.}$ construct a score able to proxy for the 
several transition risk drivers. In particular, they developed 
a "Brown-Green-Score" (BG) which is defined as:
         """)

st.latex(r'''
         \begin{equation}
         BGS_{i,t} = 0.7 \text{Value Chain}_{i,t}
         + 0.15 \text{Adaptability}_{i,t}
         + 0.15 \text{Public Perception}_{i,t}
            \end{equation}
            ''')

st.write(r"""
The variables $\text{Value Chain}_{i,t}$,
$\text{Adaptability}_{i,t}$ and $\text{Public Perception}_{i,t}$ are
         proxies for the terms policy risk, technology risk and 
         preference risk, respectively.
         To build the measure, they relied on 10 different ESG variables, retrieved 
         from four different data providers.
         They argued that merging the ESG variables between these datasets should minimise the 
         potential self-reporting bias.
         """)

st.write(r"""
Other studies tend to focus on only one of the three firm-level emissions variables,
         and researchers often argue that these variables should proxy the three transition risks drivers.
Hsu $\textit{et al.}$ (2020) constructed a measure of emission intensity 
at the firm level by aggregating plant-level data from the Toxic Release Inventory (TRI) database in the US.
The Trucost and Thomson Reuters' Asset4 ESG databases provide emission data 
         at the aggregated firm level, both for the United States and the entire world.
Moreover, these databases also provide data related to the three different types of scope emissions. 
         """)

st.write(r"""
         Bolton and Kacperczyk (2020) decomposed the three measures of carbon risk for each type of emissions.
         Notably, as Busch $\textit{et al.}$ (2018) observed, there is little variation 
         in the reported scope 1 and 2 emissions among data providers. 
         """)

st.write(r"""
         Pastor $\textit{et al.}$ (2021) proxies this overall climate risk exposure by means 
         of the E-scores provided by the MSCI and Sustainalytics databases, arguing that they should capture the 
         different dynamics. Engle $\textit{et al.}$ (2020) 
         constructed E-score measures at the firm level by taking the difference between positive and negative 
         E-scores subcategories. 
         """)


st.write(r"""
One must understand how the emission variables are related to the cross-section fo stocks retuns. Bolton and 
         Kacperczyk (2020) argued the total amount of emissions should proxy 
         for the long-term company's exposure to transition risks, as it is likely that regulations 
         aimed to curb emissions are targeted more toward these types of fims. The 
         opposite is true for the year-by-year changes in emissions, as this measure should 
         capture the short-term effects of transition risks on stock returns.
         """)

st.write(r"""
The economic rationale behind the emission intensity measure is explained using two different channels.
         Hsu $\textit{et al.}$ (2020) assumed this measure shoudl proxy for the climate policy risk 
         exposure of pollutant firms, so it is allowed to play a similar role as the total amount of firm 
         emissions as in Bolton and Kacperczyk (2021). 
         """)
