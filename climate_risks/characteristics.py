import streamlit as st
import numpy as np
import plotly.graph_objs as go
import numpy as np
import sympy as sp
import sympy.stats as stats
import plotly.graph_objs as go
import pandas as pd

st.title('Which Characteristics to Sort on?')

st.write(r"""
The literature on climate finance discusses two broad categories of climate-related risks:
1. **Physical Climate Risk**: Risks arising from the direct impact of climate change on assets, such as sea level rise or extreme weather events damaging production facilities.
2. **Transition Risk**: Risks associated with the transition to a low-carbon economy, such as carbon taxes that may reduce the profitability of fossil fuel companies.

In this course, we will focus on transition risks.
Assets may have different exposures to these risks, meaning that climate risk realizations could create both winners and losers in the market. For example, while coal companies might suffer from transition risks, renewable energy companies may benefit.

Similar to traditional firm characteristics, we can expect that firms more exposed to climate risks may also exhibit higher expected returns, as investors require compensation for these additional risks.
""")


st.subheader('Carbon Emissions')


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

st.subheader('Environmental Scores')

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