import streamlit as st

st.title('Green Factor')

st.subheader('Climate Concerns and Green Factor')


st.subheader('Green Investment Portfolios')

st.write(r'''
Apel et al. (2023) distinguish between two different construction approaches for available
proxies of 'green' investment portfolios: (i) decarbonized portfolios and (ii) pure-play approaches.
Decarbonized portfolios typically reduce carbon exposure while preserving diversification by exclusion 
         or reweighting constituents on emissions relative to a financial metric.
Pure-play approaches, on the other hand, focus on companies that derive a significant portion of their
            revenue from products and services related to the transition to a low-carbon economy.
''')

st.markdown("""
<table>
    <caption>Index description.</caption>
    <thead>
        <tr>
            <th>Name</th>
            <th>Benchmark</th>
            <th>History</th>
            <th>Rationale</th>
        </tr>
    </thead>
    <tbody>
        <tr><td colspan="4"><strong>(A) Pure-play indices</strong></td></tr>
        <tr>
            <td>MSCI Global Environment Index</td>
            <td>MSCI ACWI IMI</td>
            <td>Nov 08</td>
            <td>Comprised of companies that derive at least 50% of their revenues from environmentally beneficial products and services. Constituent selection is based on data from MSCI ESG Research.</td>
        </tr>
        <tr>
            <td>MSCI Global Alt. Energy Index</td>
            <td>MSCI ACWI IMI</td>
            <td>Jan 09</td>
            <td>Thematic sub-index of the MSCI Global Environment Index and includes companies that derive 50% or more of their revenues from products and services in Alternative Energy.</td>
        </tr>
        <tr>
            <td>S&amp;P Global Clean Energy Index</td>
            <td>S&amp;P Global BMI</td>
            <td>Nov 03</td>
            <td>Index inclusion is based on factors like company’s business description and most recent reported revenue by segment. Companies which exceed a certain carbon emissions threshold are excluded.</td>
        </tr>
        <tr>
            <td>Solactive Climate Change Index</td>
            <td>Solactive GBS LMC</td>
            <td>Nov 05</td>
            <td>Includes 30 largest companies active in sectors like solar and wind energy. Companies are classified according to the percentage of total revenues associated with activities that generate CO₂ avoidance.</td>
        </tr>
        <tr>
            <td>FTSE EO Renewable and Alt. Energy Index</td>
            <td>FTSE Global All Cap</td>
            <td>Nov 08</td>
            <td>Environmental Opportunities (EO) Index requires companies to have at least 20% of their revenues derived from significant involvement in business activities related to Renewable &amp; Alternative Energy.</td>
        </tr>
        <tr>
            <td>FTSE Environmental Technology Index</td>
            <td>FTSE Global All Cap</td>
            <td>Oct 03</td>
            <td>Constituents are required to have at least 50% of their business derived from environmental markets &amp; technologies as defined by FTSE EMCS. Longest-running environmental technology index available.</td>
        </tr>
        <tr><td colspan="4"><strong>(A) Decarbonized indices</strong></td></tr>
        <tr>
            <td>MSCI World Climate Change Index</td>
            <td>MSCI World</td>
            <td>Nov 13</td>
            <td>The index uses the MSCI Low Carbon Transition score to re-weight benchmark constituents to increase (decrease) exposure to companies participating in opportunities (risks) associated with transition.</td>
        </tr>
        <tr>
            <td>MSCI World Low Carbon Leaders Index</td>
            <td>MSCI World</td>
            <td>Nov 10</td>
            <td>By selecting companies with low carbon emissions (relative to sales) and low potential carbon emissions, the index aims to achieve 50% reduction in its carbon footprint while minimizing the tracking error.</td>
        </tr>
        <tr>
            <td>MSCI World Low Carbon Target Index</td>
            <td>MSCI World</td>
            <td>Nov 10</td>
            <td>Same rationale as MSCI World Low Carbon Leader Index but subject to a tracking error constraint of 30 bps relative to the parent index. It uses ESG CarbonMetrics data from MSCI ESG Research Inc.</td>
        </tr>
        <tr>
            <td>S&amp;P Global 1200 Fossil Fuel Free Index</td>
            <td>S&amp;P Global 1200</td>
            <td>Dec 11</td>
            <td>Index is based on its respective underlying index and consists of companies that do not own fossil fuel reserves as measured by S&amp;P Trucost Limited.</td>
        </tr>
        <tr>
            <td>S&amp;P Global LMC Carbon Efficient Index</td>
            <td>S&amp;P Global LMC</td>
            <td>Mar 09</td>
            <td>Index excludes companies classified as high carbon emitters, while overweighting or underweighting those companies that have lower or higher levels of GHG emissions per unit of revenue.</td>
        </tr>
        <tr>
            <td>STOXX Global 1800 Low Carbon</td>
            <td>STOXX Global 1800</td>
            <td>Dec 11</td>
            <td>Offering a reduction in carbon emissions to underlying benchmark by overweighting lower carbon emitters and underweighting higher carbon emitters. STOXX uses CDP and ISS ESG as data sources.</td>
        </tr>
    </tbody>
</table>
<p style="text-align: right; font-style: italic; font-size: 0.9em;">Source: Apel et al. (2023)</p>
""", unsafe_allow_html=True)

st.write(r'''
Apel et al. (2023) propose to analyze the return sensitivity of commonly 
         used portfolio approache towards climate transition risk, 
         without a priori assumptions about the "right" approach 
         to determine the "gree credentials" of firm characteristics.
         The objective is to seek clarification which type of climate 
         investment approaches provide exposure to the risk and opportunities 
         associated with the transition to a low-carbon economy.
         Investors that want to hedge transition risk will desire portfolios 
         that perform well if the public demand to confont the adverse effects 
         of climate change increases.
''')

st.write(r'''
The authors analyze the contemporaneous relationship between innovations 
         (changes unexpected by investors) in a transition risk sentimient index (thereafter TRI)
         and weekly active returns (i.e., returns net of the benchmark) of the green investment
         portfolios. They consider a multivariate time series regression to 
         control for other factors potentially driving active index returns. 
         Therefore, the authors regress the active returns $r_{i,t}$ of the green investment
            portfolios on the innovations $\varepsilon_{t}^{TRI}$ in the TRI and the five 
         Fama-French factors (Fama and French, 1993, 2015) as well as the momentum factor
         (Carhart, 1997). The regression model is given by:
''')

st.latex(r'''
         \begin{equation}
         \begin{aligned}
r_{i,t} = \alpha_{i} + \beta_{1,i} MKT_t + \beta_{2,i} SMB_t + \beta_{3,i} HML_t + \\ \beta_{4,i} RMW_t + \beta_{5,i} CMA_t + \beta_{6,i} MOM_t + \beta_{7,i} \varepsilon_{t}^{TRI} + \varepsilon_{i,t} 
         \end{aligned}   
         \end{equation}
''')

st.write(r'''
where $r_{i,t}$ is the active return of the green investment portfolio $i$ in week $t$, 
         $\alpha_{i}$ is the intercept, $MKT_t$ is the market factor, $SMB_t$ is the size factor, 
         $HML_t$ is the value factor, $RMW_t$ is the profitability factor, $CMA_t$ is the investment factor, 
         $MOM_t$ is the momentum factor, and $\varepsilon_{t}^{TRI}$ is the innovation in the TRI.
''')

st.write(r'''
We are going to try to replicate the analysis. We first download daily 
prices for one green ETF from the data provider 
Yahoo Finance. To download the data, we use the `download()` function 
from the `yfinance` package. Data from Yahoo Finance come as a pandas DataFrame with the date as the index.
''')


st.code(r'''
prices = (yf.download(
    tickers="ICLN", 
    progress=False
  )
  .reset_index()
  .assign(symbol="ICLN")
  .rename(columns={
      "Date": "date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adjusted",
    "Volume": "volume"
  })
  )
        ''')   

import numpy as np
import pandas as pd
import yfinance as yf


prices = (yf.download(
    tickers="ICLN", 
    progress=False
  )
  .reset_index()
  .assign(symbol="ICLN")
  .rename(columns={
      "Date": "date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adjusted",
    "Volume": "volume"
  })
  )

st.dataframe(prices.head().round(3))

st.write(r'''The above snippet of code returns a dataframe 
            with daily prices for the iShares Global Clean Energy ETF (ICLN).
         The open, high, low, close, and adjusted close prices are
            available for each trading day. The adjusted close price
            is the closing price adjusted for dividends and splits.
         We often rely on the adjusted close price to calculate returns.
    ''')

st.write(r'''
Next, we use the `plotnine` (Kibirige, 2023) package to visualize the time 
         series of adjusted prices. 
         ''')

st.code(r'''
from plotnine import *

prices_figure = (
    ggplot(prices, aes(x="date", y="adjusted")) +
    geom_line() +
    labs(
        title="Adjusted Prices of ICLN",
        x="Date",
        y="Adjusted Price"
    )
)
        
prices_figure.draw()
''')
from plotnine import *

prices_figure = (
    ggplot(prices, aes(x="date", y="adjusted")) +
    geom_line() +
    labs(
        title="Adjusted Prices of ICLN",
        x="Date",
        y="Adjusted Price"
    )
)


st.pyplot(prices_figure.draw())