import streamlit as st
import numpy as np
import plotly.graph_objs as go
import numpy as np
import sympy as sp
import sympy.stats as stats
import plotly.graph_objs as go
import pandas as pd

st.title('Climate Risks: Rewarded or Unrewarded Risks?')

st.write("""

The climate finance literature employ characteristic-sorted portfolio to investigate the relationship between climate risks and cross-sectional stock returns.
Reasoning in terms of portfolio,
         the fundamental debate in the literature is to understand whether climate change risk dynamics 
            are rewarded or unrewarded in the cross-section of stock returns.
         """)

st.subheader('Chasing the Climate Risks Premium')

# Data for the table
data = {
    "Study": [
        "Balvers et al. (2017)", "Bansal et al. (2019)", "Bolton and Kacperczyk (2021a)", 
        "Bolton and Kacperczyk (2020)", "Engle et al. (2020)", "Hsu et al. (2020)", 
        "Nagar and Schoenfeld (2021)", "Ardia et al. (2021)", "Cheema-Fox et al. (2021)", 
        "Ding et al. (2020)", "Görgen et al. (2020)", "Gostlow (2019)", "Hong et al. (2019)", 
        "In et al. (2017)", "Jiang and Weng (2019)", "Kumar et al. (2019)", "Pastor et al. (2021)"
    ],
    "Climate risk measure": [
        "Shocks in temperature", "Temperature anomaly", "Three emission measures", 
        "Three emission measures", "E-scores", "Emission intensity", 
        "Text mining index", "Emission intensity", "Two emission measures", 
        "Soil moisture data", "BG scores", "FTS scores", "PDSI index", 
        "Emission intensity", "ACI index", "Temperature anomaly", "E-scores"
    ],
    "Economic rationale": [
        "Sector and firm dynamics", "Dividend beta", "Transition risk proxies", 
        "Transition risk proxies", "Hedging assets", "Climate policy risk", 
        "Economic tracking", "Pastor et al. (2020)", "Investor irrationality", 
        "Forecast of firm profit", "Transition risk proxies", "Climate risk proxies", 
        "Forecast of firm profit", "Investor irrationality", "Forecast of firm profit", 
        "Forecast of firm profit", "Pastor et al. (2020)"
    ],
    "Countries": [
        "USA", "USA", "USA", "77 countries", "USA", "USA", 
        "USA", "USA", "USA", "Multiple", "Multiple", "USA, EU and Japan", 
        "31 countries", "USA", "USA and Canada", "USA", "USA"
    ],
    "Sectors": [
        "Multiple", "Multiple", "Multiple", "Multiple", "Multiple", "Multiple", 
        "Multiple", "Multiple", "Multiple", "Food", "Multiple", "Multiple", 
        "Food", "Multiple", "Food and forestry", "Multiple", "Multiple"
    ],
    "Period": [
        "1953–2015", "1970–2016", "2005–2017", "2005–2018", "2009–2016", "1991–2016", 
        "2003–2019", "2010–2018", "2013–2020", "1984–2014", "2010–2017", "2008–2017", 
        "1985–2014", "2005–2015", "1993–2018", "1926–2016", "2012–2020"
    ],
    "Sample": [
        "n.a.", "n.a.", "3421", "14,400", "n.a.", "503", 
        "10,000", "500", "1002", "776", "1657", "668", 
        "776", "739", "145", "n.a.", "n.a."
    ],
    "Is climate risk priced?": [
        "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", 
        "Yes", "No", "Yes", "Yes", "Yes", "Yes", 
        "Yes", "Yes", "Yes", "Yes", "Yes"
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Display the table using Streamlit
st.subheader("Table 1: Climate risk factors and the cross-section of stock returns")
st.write("This table summarizes studies on climate risk factors and their effect on stock returns across different sectors, countries, and time periods.")
st.dataframe(df)

st.subheader('Climate Risks and Unexpected Returns')

st.write(r"""
Introduce Ardia and Pastor thesis to come in the next section that explains mixed results. 
         Climate Risks effects are currently unexpected returns.
         """)
