import streamlit as st

st.title('Green Investment Portfolios')


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

st.subheader('ETF Data')

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

st.write(r'''
Instead of analyzing the prices, we computed daily returns defined as:
         ''')

st.latex(r'''
\begin{equation}
         r_t = p_t/p_{t-1} - 1
\end{equation}
''')

st.write(r'''
where $r_t$ is the return on day $t$ and $p_t$ is the adjusted close price on day $t$.
         ''')

st.write(r'''
         We can use the `pct_change()` method from pandas to compute the daily returns.
         ''')


st.code(r'''
returns = (
    prices
    .sort_values("date")
    .assign(
        ret = lambda x: x["adjusted"].pct_change()
    )
    .get(["symbol","date","ret"])
    .dropna()
)
''')

returns = (
    prices
    .sort_values("date")
    .assign(
        ret = lambda x: x["adjusted"].pct_change()
    )
    .get(["symbol","date","ret"])
    .dropna()
)
st.dataframe(returns.head().round(3))

st.write(r'''
         Next, we can visualize the 
         distribution of daily returns in a histogram 
         where we also use the `mizani` package for 
         formatting functions. We also add a dashed line that 
         indicates the five percent quantile of the daily 
         returns to the histogram, which is a proxy for 
         the worst return of the ETF with a 5% probability.
         The 5\% quantile is a common measure of downside risk.
         ''')


st.code(r'''
from mizani.formatters import percent_format

quantile_05 = returns["ret"].quantile(0.05)

returns_figure = (
  ggplot(returns, aes(x="ret")) +
  geom_histogram(bins=100) +
  geom_vline(aes(xintercept=quantile_05), 
                 linetype="dashed") +
  labs(x="", y="",
       title="Distribution of daily ICLN ETF returns") +
  scale_x_continuous(labels=percent_format())
)
returns_figure.draw()
''')

from mizani.formatters import percent_format

quantile_05 = returns["ret"].quantile(0.05)

returns_figure = (
  ggplot(returns, aes(x="ret")) +
  geom_histogram(bins=100) +
  geom_vline(aes(xintercept=quantile_05), 
                 linetype="dashed") +
  labs(x="", y="",
       title="Distribution of daily ICLN ETF returns") +
  scale_x_continuous(labels=percent_format())
)
st.pyplot(returns_figure.draw())

st.write(r'''
`bins = 100` specifies the number of bins in the histogram.
            The `geom_vline()` function adds a vertical line at the
            five percent quantile of the daily returns. The `scale_x_continuous()`
            function formats the x-axis as a percentage.
         ''')

st.write(r'''
A typical task before proceeding is to compute summary 
         statistics of the daily returns. We can use the `describe()`
            method from pandas to compute the mean, standard deviation,
            minimum, 25th percentile, median, 75th percentile, and maximum
            of the daily returns.
         ''')

st.code(r'''
        pd.DataFrame(returns["ret"].describe().round(3)).T
''')

st.write(r'''
         We see that the maximum daily return was 17.4\%. The 
         average daily return is 0\%.
         ''')

pd.DataFrame(returns["ret"].describe().round(3)).T

st.write(r'''
We can also compute these summary statistics 
         for each year individually, by using 
            the `groupby()` method from pandas.
         ''')


st.code(r'''
(returns
  .groupby(returns["date"].dt.year)['ret']
  .describe()
  .round(3)
)
''')

yearly = (returns
  .groupby(returns["date"].dt.year)['ret']
  .describe()
  .round(3)
)

st.dataframe(yearly)

st.write(r'''
         We now generalize the above code 
         such that the computations can handle 
         multiple ETFs. 
         Tidy data makes it easy to generalize the 
         computations to multiple ETFs.
         I have first defined a list of pure play ETFs.
         Next, we can use `yfinance` to download the daily
            prices for the ETFs in the list. We then 
         transform the data into a tidy format.
            ''')


st.code(r'''
list_pure_play_ETFs = ['ICLN','QCLN','PBW','TAN','FAN']

prices_daily = (
    yf.download(
        tickers=list_pure_play_ETFs, 
        progress=False
    )
)

prices_daily = (prices_daily
  .stack()
  .reset_index()
  .rename(columns={
    "Date": "date",
    "Ticker": "symbol",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adjusted",
    "Volume": "volume"}
  )
)
        ''')
        
list_pure_play_ETFs = ['ICLN','QCLN','PBW','TAN','FAN']

prices_daily = (
    yf.download(
        tickers=list_pure_play_ETFs, 
        progress=False
    )
)

prices_daily = (prices_daily
  .stack()
  .reset_index()
  .rename(columns={
    "Date": "date",
    "Ticker": "symbol",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adjusted",
    "Volume": "volume"}
  )
)

st.dataframe(prices_daily.head().round(3))

st.write(r'''
         We again use the `mizani` package but this time to format 
         date to get nicer labels on the x-axis.
            ''')

st.code(r'''
from mizani.breaks import date_breaks
from mizani.formatters import date_format

prices_daily_figure = (
  ggplot(prices_daily, 
         aes(y="adjusted", x="date", color="symbol")) +
 geom_line() +
 labs(x="", y="", color="",
      title="ETF prices") +
 scale_x_datetime(date_breaks="5 years", date_labels="%Y")
)
        
prices_daily_figure.draw()
        ''')

from mizani.breaks import date_breaks
from mizani.formatters import date_format

prices_daily_figure = (
  ggplot(prices_daily, 
         aes(y="adjusted", x="date", color="symbol")) +
 geom_line() +
 labs(x="", y="", color="",
      title="ETF prices") +
 scale_x_datetime(date_breaks="5 years", date_labels="%Y")
)
st.pyplot(prices_daily_figure.draw())


st.subheader('Active Returns')


st.subheader('Risk Factors and TRI')



st.subheader('Multivariate Time Series Regression')
