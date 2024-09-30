import streamlit as st
import numpy as np
import plotly.graph_objs as go
import sympy as sp
import sympy.stats as stats

st.title('Climate Risks and Stock Returns')

st.write(r"""
The literature on climate finance is primarily shaped by two key themes:
1. Climate change risks.
2. Financial implications for asset pricing.

Climate risks are categorized into physical and transition risks (Carney, 2015). Physical risks arise from climate and weather events that impact company operations or society (Tankov and Tantet, 2019). These can be:
- Acute (e.g., extreme weather events).
- Chronic (e.g., long-term shifts in climate patterns).

Transition risks relate to scenarios leading to a low-carbon economy and the impact on fossil fuels and related sectors (Curtin et al., 2019). This course focuses on transition risks and their asset pricing implications.
""")

st.write(r"""
The financial aspect of climate risks is divided by asset class. Here, we focus on equity markets and the effect of both physical and transition risks on firm value.
""")

st.subheader('Physical Risks Channel')

st.write(r"""
Physical risks depend on three factors (Tankov and Tantet, 2019):
""")

st.latex(r'''
\text{Physical Risk} = f(\text{Climate Hazard}, \text{Exposure}, \text{Vulnerability})
''')

st.write(r"""
- **Climate Hazard**: The intensity and probability of climate events.
- **Exposure**: The geographical distribution of assets at risk.
- **Vulnerability**: The susceptibility of assets to damage, influenced by adaptation capacity.

Climate hazards are analyzed through climate and natural disaster datasets. Tankov and Tantet (2019) classify climate datasets into four types:
1. Observational,
2. Reanalysis,
3. Projections,
4. Climate indices.

Natural disaster datasets, while traditionally used in disaster management, are gaining traction in climate finance. They provide both location data and historical economic damages, making them useful for financial analysis.
""")

st.write(r"""
To assess exposure and vulnerability, asset-level data is essential. This includes geographical data, asset characteristics, and innovation measures that reflect a firm's adaptation capabilities.
""")

st.subheader('Transition Risks Channel')

st.write(r"""
Transition risks are driven by three factors (Semieniuk et al., 2021):
""")

st.latex(r'''
\text{Transition Risk} = f(\text{Policy Risk}, \text{Technology Risk}, \text{Preference Change})
''')

st.write(r"""
- **Policy Risk**: Risks and opportunities from climate policies aimed at reducing greenhouse gas (GHG) emissions (Nordhaus, 1977, 1993). Policies include carbon taxes, cap-and-trade schemes, regulations, and green subsidies.
- **Technology Risk**: The risk of new technologies that support the transition to low-carbon energy sources. Firms may adopt these technologies due to regulatory pressure, competition, or investor demand.
- **Preference Change**: Shifts in consumer or investor preferences towards environmentally friendly assets (Pastor et al., 2020). Investors may avoid carbon-intensive assets due to higher downside risks or ethical motivations.

Data on firm innovation, emissions, and investor flows towards green funds are critical for modeling these transition risks.
""")

st.subheader('Asset Pricing Implications')

st.write(r"""
The climate risk drivers described above can impact the cross-section of stock returns. Modeling how these risks systematically affect the equity market presents challenges for asset pricing. However, recent literature in climate finance has identified two primary ways to link these risks to equity prices at a macro level.
""")

st.write(r"""
According to Giglio et al. (2020), financial models of climate risks vary based on researchers' beliefs about climate change uncertainty. There is ongoing debate on whether climate risks are a low-probability, catastrophic outcome (Weitzman, 2009) or a stochastic process tied to current aggregate consumption (Nordhaus, 1977). One key objective of these models is to estimate the social cost of carbon (Bansal et al., 2019), a metric with significant policy implications. 

Initially, estimates of the social cost of carbon were largely derived from macroeconomic theories. However, Bansal et al. (2019) suggested that equity markets might offer an independent assessment of climate risks. Empirical evidence in this area could improve macro-financial models and lead to more accurate estimates of the social cost of carbon.
""")

st.write(r"""
Other approaches focus on developing equilibrium models that align environmental, social, and corporate governance (ESG) criteria with asset return dynamics (Avramov et al., 2021; Pastor et al., 2020; Pedersen et al., 2020). ESG investing is closely related to climate considerations. While many models consider the indirect effects of sustainability preferences on asset prices, Pastor et al. (2020) developed a model where climate risks directly influence equilibrium returns, as they are embedded in investors' utility functions.
""")

st.write(r"""
In both approaches, theoretical predictions depend on how financial markets price climate risks. Empirical asset pricing tests of security returns play a crucial role in determining whether and how these risks are reflected in the equity market.
""")
