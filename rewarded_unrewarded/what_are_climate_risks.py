import streamlit as st
import numpy as np
import plotly.graph_objs as go
import sympy as sp
import sympy.stats as stats

st.title('Climate Risks and Stock Returns')


st.subheader('Characteristics and Cross Section of Stock Returns')


st.write(r"""
         The literature in asset pricing has found that the cross-section of stock returns is related to firm characteristics.
Those characteristics may proxy exposure to some underlying risk factors. That is, they 
may help to identify stocks that behave poorly in bad times, and thus have high expected returns.
         
Fama and French (2015) use the dividend discount model to motivate the use of a combination of firm 
characteristics based on valuation, profitability and investment to explain the cross-section of stock returns:
""")

st.latex(r'''
            \begin{equation}
         M_t = \frac{\mathbb{E}(Y_{t+1}) - \mathbb{E}(dB_{t+1})}{1+r}
        \end{equation}
            ''')

st.write(r"""
where $M_t$ is the market value of equity, $Y_{t+1}$ is the expected future earnings, 
$dB_{t+1}$ is the expected change in book value of equity, and $r$ is, approximately, the expected return.
You can solve for $r$ to get the expected return:
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
This model implies three statements about expected returns. First, 
         if we fix everything except the current value of the stock, $M_t$,
         and the expected stock return, $r$, then a lower value of $M_t$, or 
         equivalently a higher book-to-market ratio, implies a higher expected return:
""")


# Construct the LaTeX expression using the actual values
latex_eq = r"\begin{equation} r = \frac{" + str(default_Y_t1) + " - " + str(default_dB_t1) + "}{" + str(M_t_val) + "} - 1 =" + str(round(expected_return_case_1,2)) +"\end{equation}" 

# Display the equation in LaTeX format in Streamlit
st.latex(latex_eq)

st.write(r"""
Zhang (2005) provides a rationale for the value premium based on costly 
         reversibility of investments. The stock price 
         of value firms is mainly made up of tangible assets 
         which are hard to reduce while growth firm's stock price is mainly driven by growth options.
         Therefore value firms are much more affected by bad times.
         """)

st.write(r"""
Next, if we fix $M_t$ and the values of everything except the expected future earnings and 
         the expected stock return, the equation tells us that higher expected earnings imply 
         a higher expected return:
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
A problem with risk-based explanations of this link between 
profitability and expected return is that these characteristics would intuitively 
         suggest that these firms are less risky.
However, on the other hand, more profitable firms tend to be growth firms,
         which have more of their cash flow in the distant future. 
         More distant cash flows are more uncertain and should require a risk premium.
         Another risk-based explanation is that higher profitability should attract 
         more competition, threatening profit margins (and thus making future cash flows less certain).
         And that, too, creates more risk and should require a risk premium.
""")

st.write(r"""
Finally, for fixed values of $M_t$ and expected earnings, higher expected growth in 
book equity - investment - implies a lower expected return:
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
Controlling for a firm's market value and expected profitability, 
         a company that must invest heavily to sustain its 
         profits should have lower contemporaneous free cash flows to investors 
         than a company with similar profits but lower investment. 
         """)


st.subheader('Could Climate Risks-Related Characteristics Explain Stock Returns?')



st.write(r"""
The literature considers the price effects of at least two broad categories of climate-related risk 
factors: physical climate risk and transition risk. Physical climate risk includes 
risks of the direct impairment of productive assets resulting from climate change. 
Transition risk includes risks to cash flows arising from a possible transition to a low-carbon economy.
         
A central element is that assets are differentially exposed to the climate 
         risk factors. For example, the threat of damage from rising sea levels 
         to firm's production facilities close to the sea could be considered a physical climate risk.
One example of a transition risk is the possible introduction of a carbon tax that might leave fossil fuel componaies 
         with stranded assets that no longer profitable to operate. 

Different assets may be positively or negatively exposed to these types of climate risks. 
In other words, realizations of both physical and transition risks will have winners and losers 
         in asset markets. For example, while coal companies would likely suffer 
         from realizations of transition risks, renewable enrgy companies might benefit.

A key challenge is therefore to obtain measures of different assets' exposures to both physical and climate risks.
         
For example, 
studies in climate finance tend to assume that the higher a company's emissions, the "browner"
that firm is (Bolton and Kacperczyj, 2021). The opposite holds for "green" firms.  
         
Therefore, as in the case of established characteristics related to the cross-section of stock returns,
we could expect that firms with higher exposure to climate risks, proxied by some climate-related characteristics, would have higher expected returns. 
""")

