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
