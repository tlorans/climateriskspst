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
         and monthly active returns (i.e., returns net of the benchmark) of the green investment
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

st.write(r'''
This is also easy to now compute the daily returns for the ETFs.
         ''')

st.code(r'''
   returns_daily = (prices_daily
  .assign(ret=lambda x: x.groupby("symbol")["adjusted"].pct_change())
  .get(["symbol", "date", "ret"])
  .dropna(subset="ret")
)

(returns_daily
  .groupby("symbol")["ret"]
  .describe()
  .round(3)
)
        ''')
     

returns_daily = (prices_daily
  .assign(ret=lambda x: x.groupby("symbol")["adjusted"].pct_change())
  .get(["symbol", "date", "ret"])
  .dropna(subset="ret")
)

desc = (returns_daily
  .groupby("symbol")["ret"]
  .describe()
  .round(3)
)

st.dataframe(desc)


st.subheader('Resampling and Active Returns')

st.write(r'''
We now want to prepare the data for the multivariate time series regression, 
as in Apel et al. (2023). 
We first need to resample the daily returns to monthly returns.
We also need to calculate the active returns of the ETFs, 
that is, the returns net of the benchmark.
         
We can resample the daily returns to monthly returns
by using the `resample()` method from pandas on the price data.
Note that we have added the MSCi World ETF (IWRD.L) to the list of ETFs,
which we use as a benchmark.         
         ''')

st.code(r'''
list_ETFs = ['IWRD.L','ICLN','QCLN','PBW','TAN','FAN']

# Download and process data
prices_monthly = (
    yf.download(
        tickers=list_ETFs, 
        progress=False
    )
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
        "Volume": "volume"
    })
    .set_index("date")
    .groupby("symbol")["adjusted"]  # Only resample the adjusted price
    .resample("M")
    .last()
)

returns_monthly = (prices_monthly
                  .reset_index()
    .assign(
        ret = lambda x: x.groupby("symbol")["adjusted"].pct_change()
    )
    .get(["symbol", "date", "ret"])
    .dropna(subset="ret")
)
        ''')        

list_ETFs = ['IWRD.L','ICLN','QCLN','PBW','TAN','FAN']

# Download and process data
# Function to download and process data with caching
@st.cache_data
def get_monthly_prices():
    prices = (
        yf.download(
            tickers=list_ETFs, 
            progress=False
        )
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
            "Volume": "volume"
        })
        .set_index("date")
        .groupby("symbol")["adjusted"]  # Only resample the adjusted price
        .resample("M")
        .last()
    )
    return prices


returns_monthly = (get_monthly_prices()
                  .reset_index()
    .assign(
        ret = lambda x: x.groupby("symbol")["adjusted"].pct_change()
    )
    .get(["symbol", "date", "ret"])
    .dropna(subset="ret")
)

returns_monthly

st.write(r'''
The above code downloads the monthly prices for the ETFs and the benchmark.
We then calculate the monthly returns and drop the missing values.''')


st.write(r'''
         An easy way to calculate the active returns is to 
         pivot the monthly returns such that we can 
         subtract the benchmark returns column from each ETF column.
         We then use the `melt()` method from pandas to transform the
            data back to a tidy format.
         ''')

st.code(r'''
active_returns = (
    returns_monthly
    .pivot(index="date", columns="symbol", values="ret")
    .apply(lambda x: x- x["IWRD.L"], axis=1)
    .dropna()
    .reset_index()
    .melt(id_vars="date", var_name="symbol", value_name="active_ret")
)
        ''')

active_returns = (
    returns_monthly
    .pivot(index="date", columns="symbol", values="ret")
    .apply(lambda x: x- x["IWRD.L"], axis=1)
    .dropna()
    .reset_index()
    .melt(id_vars="date", var_name="symbol", value_name="active_ret")
)

active_returns

st.write(r'''
         It may be interesting to visualize the difference between the 
         absolute returns and the active returns. 
         We make use again of the `pivot()` and 
            `melt()` methods from pandas.         ''')


st.code(r'''
 cum_absolute_returns = (
    returns_monthly
    .pivot(index="date", columns="symbol", values="ret")
    .dropna()
    .apply(
        lambda x: (1 + x).cumprod() - 1
    )
    .reset_index()
    .melt(id_vars="date", var_name="symbol", value_name="cum_ret")
)       

cum_absolute_returns_figure = (
    ggplot(cum_absolute_returns, 
         aes(y="cum_ret", x="date", color="symbol")) +
 geom_line() +
 # horizontal line at 0
    geom_hline(yintercept=0, linetype="dashed") +
 labs(x="", y="", color="",
      title="Cumulative absolute returns") +
 scale_x_datetime(date_breaks="5 years", date_labels
                    ="%Y")
)

cum_absolute_returns_figure.draw()
        ''')

cum_absolute_returns = (
    returns_monthly
    .pivot(index="date", columns="symbol", values="ret")
    .dropna()
    .apply(
        lambda x: (1 + x).cumprod() - 1
    )
    .reset_index()
    .melt(id_vars="date", var_name="symbol", value_name="cum_ret")
)

cum_absolute_returns_figure = (
    ggplot(cum_absolute_returns, 
         aes(y="cum_ret", x="date", color="symbol")) +
 geom_line() +
 # horizontal line at 0
    geom_hline(yintercept=0, linetype="dashed") +
 labs(x="", y="", color="",
      title="Cumulative absolute returns") +
 scale_x_datetime(date_breaks="5 years", date_labels
                    ="%Y")
)

st.pyplot(cum_absolute_returns_figure.draw())

st.code(r'''
cum_active_returns = (
    active_returns
    .pivot(index="date", columns="symbol", values="active_ret")
    .apply(
        lambda x: (1 + x).cumprod() - 1
    )
    .reset_index()
    .melt(id_vars="date", var_name="symbol", value_name = "cum_ret")
)

cum_active_returns_figure = (
    ggplot(cum_active_returns, 
         aes(y="cum_ret", x="date", color="symbol")) +
 geom_line() +
 # horizontal line at 0
    geom_hline(yintercept=0, linetype="dashed") +
 labs(x="", y="", color="",
      title="Cumulative active returns") +
 scale_x_datetime(date_breaks="5 years", date_labels
                    ="%Y")
)

cum_active_returns_figure.draw()
        ''')

cum_active_returns = (
    active_returns
    .pivot(index="date", columns="symbol", values="active_ret")
    .apply(
        lambda x: (1 + x).cumprod() - 1
    )
    .reset_index()
    .melt(id_vars="date", var_name="symbol", value_name = "cum_ret")
)

cum_active_returns_figure = (
    ggplot(cum_active_returns, 
         aes(y="cum_ret", x="date", color="symbol")) +
 geom_line() +
 # horizontal line at 0
    geom_hline(yintercept=0, linetype="dashed") +
 labs(x="", y="", color="",
      title="Cumulative active returns") +
 scale_x_datetime(date_breaks="5 years", date_labels
                    ="%Y")
)

st.pyplot(cum_active_returns_figure.draw())

st.write(r'''
Obviously, the active returns of the benchmark 
         are zero.
         Over the period, 
         the active returns of the green ETFs are
         nill or negative. This is in line with the 
         negative expected returns for the green 
         factor from Pastor et al. (2021, 2022). 
         ''')

st.subheader('Risk Factors and TRI')

st.write(r'''
         We start by downloading some famous Fama-French factors (Fama and French, 1993, 2015).
         We use the `pandas_datareader` package that provides a simple interface to 
         read data from Kenneth French's website.
         We use the `pd.DataReader()` function of the package to download monthly Fama-French factors.
         ''')


st.code(r'''
import pandas_datareader as pdr
# min date in active returns df as start date
start_date = active_returns["date"].min()
# max date in active returns df as end date
end_date = active_returns["date"].max()

factors_ff5_monthly_raw = pdr.DataReader(
  name="F-F_Research_Data_5_Factors_2x3",
  data_source="famafrench", 
  start=start_date, 
  end=end_date)[0]

factors_ff5_monthly = (factors_ff5_monthly_raw
  .divide(100)
  .reset_index(names="date")
  .assign(
        date=lambda x: pd.to_datetime(x["date"].astype(str)) + pd.offsets.MonthEnd(0)
      )  
   .rename(str.lower, axis="columns")
  .rename(columns={"mkt-rf": "mkt_excess"})
)
        ''')
import pandas_datareader as pdr
# min date in active returns df as start date
start_date = active_returns["date"].min()
# max date in active returns df as end date
end_date = active_returns["date"].max()

@st.cache_data
def get_ff_monthly() -> pd.DataFrame:
    return pdr.DataReader(
    name="F-F_Research_Data_5_Factors_2x3",
    data_source="famafrench", 
    start=start_date, 
    end=end_date)[0]


factors_ff5_monthly = (get_ff_monthly()
  .divide(100)
  .reset_index(names="date")
  # to date time and end of month
  .assign(
        date=lambda x: pd.to_datetime(x["date"].astype(str)) + pd.offsets.MonthEnd(0)
      )
  .rename(str.lower, axis="columns")
  .rename(columns={"mkt-rf": "mkt_excess"})
)

st.dataframe(factors_ff5_monthly.head().round(3))


st.write(r'''
         We can now visualize the cumulative returns of the Fama-French factors.
         ''')

st.code(r'''
cum_returns_factors = (
    factors_ff5_monthly 
    .set_index("date")
    .apply(
        lambda x: (1 + x).cumprod() - 1
    )
    .reset_index()
    .melt(id_vars="date", var_name="factor", value_name="cum_ret")
)

cum_returns_factors_figure = (
    ggplot(cum_returns_factors.query('factor != "rf" & factor != "mkt_excess"'), 
         aes(y="cum_ret", x="date", color="factor")) +
 geom_line() +
 # horizontal line at 0
    geom_hline(yintercept=0, linetype="dashed") +
 labs(x="", y="", color="",
      title="Cumulative returns of Fama-French factors") +
 scale_x_datetime(date_breaks="5 years", date_labels
                    ="%Y")
)

cum_returns_factors_figure.draw()
        ''')        

cum_returns_factors = (
    factors_ff5_monthly 
    .set_index("date")
    .apply(
        lambda x: (1 + x).cumprod() - 1
    )
    .reset_index()
    .melt(id_vars="date", var_name="factor", value_name="cum_ret")
)

cum_returns_factors_figure = (
    ggplot(cum_returns_factors.query('factor != "rf" & factor != "mkt_excess"'), 
         aes(y="cum_ret", x="date", color="factor")) +
 geom_line() +
 # horizontal line at 0
    geom_hline(yintercept=0, linetype="dashed") +
 labs(x="", y="", color="",
      title="Cumulative returns of Fama-French factors") +
 scale_x_datetime(date_breaks="5 years", date_labels
                    ="%Y")
)

st.pyplot(cum_returns_factors_figure.draw())

st.write(r'''
Interestingly, except the rmw (robust minus weak) factor,
         all factors have close to zero or negative cumulative returns. It has 
         been a bad decade for the Fama-French factors.
         ''')

st.write(r'''
         We now turn to the TRI data. We have downloaded the TRI data as 
         a supplementary material from Apel et al. (2023). We 
         use `read_excel()` from pandas to read the data. We select the 
         innovation in the TRI.
         ''')

st.code(r'''
tri = (pd.read_excel('data/tri.xlsx', sheet_name='monthly')
      .rename(str.lower, axis="columns")
      .get(['date','tri_innovation_monthly'])
)
        ''')
tri = (pd.read_excel('data/tri.xlsx', sheet_name='monthly')
      .rename(str.lower, axis="columns")
      .get(['date','tri_innovation_monthly'])
)

st.write(r'''
We plot the cumulative innovation to see if there exists any 
         patterns in the transition risk sentiment.
         ''')


st.code(r'''
cum_tri = (
    tri
    .set_index("date")
    .apply(
        lambda x: x.cumsum()
    )
    .reset_index()
)

cum_tri_figure = (
    ggplot(cum_tri, aes(y="tri_innovation_monthly", x="date")) +
    geom_line() +
    geom_hline(yintercept=0, linetype="dashed") +
    labs(x="", y="",
         title="Cumulative TRI innovations") +
    scale_x_datetime(date_breaks="5 years", date_labels="%Y")
)

cum_tri_figure.draw()
        ''')      

cum_tri = (
    tri
    .set_index("date")
    .apply(
        # cumulative sum of the innovation
        lambda x: x.cumsum()
    )
    .reset_index()
)

cum_tri_figure = (
    ggplot(cum_tri, aes(y="tri_innovation_monthly", x="date")) +
    geom_line() +
    geom_hline(yintercept=0, linetype="dashed") +
    # add an horizontal line in 2007 and another one in 2015
    geom_vline(xintercept=pd.to_datetime("2004-09-01"), linetype="dashed") +
    geom_vline(xintercept=pd.to_datetime("2009-12-01"), linetype="dashed") +
    geom_vline(xintercept=pd.to_datetime("2015-12-01"), linetype="dashed") +
    geom_vline(xintercept=pd.to_datetime("2017-06-01"), linetype="dashed") +
    labs(x="", y="",
         title="Cumulative TRI innovations") +
    scale_x_datetime(date_breaks="2 years", date_labels="%Y")
)

st.pyplot(cum_tri_figure.draw())

st.write(r'''
         We see interesting patterns corresponding to:
         - The intention of Russia to join the Kyoto Protocol in 2004: we see a substantial increase in the TRI for the subsequent years.
         - The 2009 United Nations Climate Change Conference, commonly known as COP15 in Copenhagen in December 2009 is followed by a substantial decrease in unexpected changes in transition concenrs.
         - The Paris Agreement in 2015: we see a substantial increase in the TRI for the subsequent years.
        - The US withdrawal from the Paris Agreement in 2017: we see a substantial decrease in the TRI for the subsequent years.
         ''')


st.subheader('Multivariate Time Series Regression')

st.write(r'''
We are going to use `statsmodels` (Seabold and Perktold, 2010) 
         to estimate the multivariate time series regression model.
         We estimate the same model than Apel et al. (2023) but
         without the momentum factor (and with monthly instead of weekly returns). 

The estimation procedure is based on a rolling-window estimation, where we can use 
different window lenghts. Python provides a simple solution to estimate 
regression models with the function `sm.ols()` from the `statsmodels` package.
         
The function requires a formula as input that is specified in a compact symbolic form. 
An expression of the form `y ~ x1 + x2 + ...` is interpreted as a specification that the
response variable `y` is linearly dependent on the variables `x1`, `x2`, etc.
                  ''')

st.code(r'''
import statsmodels.formula.api as smf

data_for_reg = (
    active_returns
    .merge(factors_ff5_monthly, on="date", how = "inner")
    .merge(tri, on="date", how = "inner")
)

icln = data_for_reg.query('symbol == "ICLN"')

model_beta = (
    smf.ols("active_ret ~ mkt_excess + smb + hml + rmw + cma + tri_innovation_monthly", data=icln)
    .fit()
)

model_beta.summary()
        ''')
import statsmodels.formula.api as smf

data_for_reg = (
    active_returns
    .merge(factors_ff5_monthly, on="date", how = "inner")
    .merge(tri, on="date", how = "inner")
)

icln = data_for_reg.query('symbol == "ICLN"')

model_beta = (
    smf.ols("active_ret ~ mkt_excess + smb + hml + rmw + cma + tri_innovation_monthly", data=icln)
    .fit()
)

st.write(model_beta.summary())

st.write(r'''
`sm.ols()` returns a `RegressionResults` object that 
         contains the estimated coefficients,
            standard errors, t-values, and p-values.
         
The model’s R-squared of 0.263 suggests that about 26.3% of the variation in 
         ICLN’s active returns is explained by the included factors. 
         The intercept, $\hat{\alpha}$, 
         is not statistically significant (p = 0.390), 
         indicating it has little influence when 
         other variables are considered. Among the factors:

- $\hat{\beta}_{1}$ (mkt_excess) is significant (p = 0.001) with a positive effect (0.613), indicating that higher market excess returns are associated with an increase in ICLN’s active returns.
- $\hat{\beta}_{2}$ (smb) is not significant (p = 0.512), suggesting the size factor has minimal impact.
- $\hat{\beta}_{3}$ (hml) is also not significant (p = 0.860), showing no meaningful relationship between the value factor and ICLN’s active returns.
- $\hat{\beta}_{4}$ (rmw) lacks significance (p = 0.574), implying the profitability factor does not explain variation in ICLN’s returns.
- $\hat{\beta}_{5}$ (cma) is marginally significant (p = 0.072) and negative (-0.9168), indicating that conservative investment strategies may be slightly negatively associated with ICLN’s active returns.
- $\hat{\beta}_{6}$ (tri_innovation_monthly) is marginally significant (p = 0.064) and positive (33.722), suggesting that innovations in the transition risk index may positively impact ICLN’s returns, although this result is not conclusively significant.
         ''')

st.write(r'''
We now scale the estimation of the model to all ETFs in the dataset, 
         and performing rolling-window estimation.
         As in Apel et al. (2023), we use a window length of 60 months.
         We require a minimimum of 48 months of data to estimate the model.
The following function implements the regression.
                     ''')

st.code(r'''
from statsmodels.regression.rolling import RollingOLS

window_size = 60
min_obs = 48
# Function to estimate rolling t-statistics
def roll_pvalue_estimation(data, window_size, min_obs):
    data = data.sort_values("date")

    # Fit Rolling OLS and extract t-stats for tri_innovation_monthly
    result = pd.Series((RollingOLS.from_formula(
        formula="active_ret ~ mkt_excess + smb + hml + rmw + cma + tri_innovation_monthly",
        data=data,
        window=window_size,
        min_nobs=min_obs,
        missing="drop")
        .fit()
         # Get t-statistics instead of beta
    ).pvalues[:, -1])

    result.index = data.index
    return result

# Calculate rolling p-values
rolling_pvalues = (
    data_for_reg
    .groupby("symbol")
    .apply(lambda x: x.assign(
        pvalue=roll_pvalue_estimation(x, window_size, min_obs)
    ))
    .reset_index(drop=True)
    .dropna()
)
        ''')        

from statsmodels.regression.rolling import RollingOLS

window_size = 60
min_obs = 48
# Function to estimate rolling t-statistics
def roll_pvalue_estimation(data, window_size, min_obs):
    data = data.sort_values("date")

    # Fit Rolling OLS and extract t-stats for tri_innovation_monthly
    result = pd.Series((RollingOLS.from_formula(
        formula="active_ret ~ mkt_excess + smb + hml + rmw + cma + tri_innovation_monthly",
        data=data,
        window=window_size,
        min_nobs=min_obs,
        missing="drop")
        .fit()
         # Get t-statistics instead of beta
    ).pvalues[:, -1])

    result.index = data.index
    return result

# Calculate rolling p-values
rolling_pvalues = (
    data_for_reg
    .groupby("symbol")
    .apply(lambda x: x.assign(
        pvalue=roll_pvalue_estimation(x, window_size, min_obs)
    ))
    .reset_index(drop=True)
    .dropna()
)


st.write(r'''
         In the above function, we retrieve the p-values of the
            regression coefficient of the TRI innovation.
         We do it in the spirit of Apel et al. (2023) with a window size of 60 months.
         We want to know, among our ETFs, which ones has the most significant relationship with the TRI innovation.
         ''')

st.write(r'''
         We can visualize the rolling p-values of the regression coefficient of the TRI innovation.
         We have added dashed lines for the 10%, 5%, and 1% significance levels.
            ''')

st.code(r'''
# Plot the rolling t-statistics with dashed lines for significance levels
figures_pvalues = (
    ggplot(rolling_pvalues, aes(x="date", y="pvalue", color="symbol")) +
    geom_line() +
    geom_hline(yintercept=0.1, linetype="dashed", color="red") +
    geom_hline(yintercept=0.05, linetype="dashed", color="red") +
    geom_hline(yintercept=0.01, linetype="dashed", color="red") +
    labs(x="Date", y="p-value", color="Index",
         title="Rolling 5-Year Regression p-values for TRI Innovation Coefficient") +
    scale_x_datetime(date_breaks="1 year", date_labels="%Y")
)

figures_pvalues.draw()
        ''')

# Plot the rolling t-statistics with dashed lines for significance levels
figures_pvalues = (
    ggplot(rolling_pvalues, aes(x="date", y="pvalue", color="symbol")) +
    geom_line() +
    geom_hline(yintercept=0.1, linetype="dashed", color="red") +
    geom_hline(yintercept=0.05, linetype="dashed", color="red") +
    geom_hline(yintercept=0.01, linetype="dashed", color="red") +
    labs(x="Date", y="p-value", color="Index",
         title="Rolling 5-Year Regression p-values for TRI Innovation Coefficient") +
    scale_x_datetime(date_breaks="1 year", date_labels="%Y")
)

st.pyplot(figures_pvalues.draw())


st.write(r'''
This chart shows the rolling 5-year regression t-values.

T-values measure the statistical significance of 
         the relationship between the TRI 
         (Transition Risk Innovation) coefficient 
         and the active returns of each ETF.
Positive t-values suggest a positive relationship 
         between the TRI coefficient and the ETF returns, 
         while negative t-values indicate an inverse relationship.

The red dashed lines represent different significance levels 
         for the t-values:
±1.28 for 10% significance.
±1.645 for 5% significance.
±2.33 for 1% significance.
When t-values cross above these thresholds, it implies 
         a statistically significant relationship 
         between TRI and the ETF’s returns at that level. 
         For instance:
- T-values above +1.645 or below -1.645 suggest 
         significance at the 5% level.
- T-values above +2.33 or below -2.33 indicate 
         strong significance at the 1% level.

Our results show:
- FAN (in red): Shows a strong positive relationship with TRI initially, with t-values above 3, suggesting a highly significant effect. However, this relationship declines over time and eventually falls below the 1% significance line.
- ICLN (in green): Initially significant at the 5% level, ICLN’s relationship with TRI gradually becomes less significant but remains close to the 1.28 threshold.
- PBW (in pink): Exhibits relatively high t-values early on, but the significance decreases over time, staying near the 10% threshold.
- QCLN (in blue): Shows a less consistent relationship with TRI, with t-values generally remaining below the significance thresholds.
- TAN (in purple): Similar to QCLN, TAN shows low t-values throughout the period, indicating a weak or non-significant relationship with TRI.
Overall Trends:

There is a general downward trend in t-values for most ETFs, indicating that the strength of the relationship between TRI and active returns has weakened over time.
By the end of the period (around 2020–2021), most ETFs have t-values below the 5% significance threshold, suggesting that TRI’s impact on these ETFs has become less statistically significant.

As green investment portfolios, ETFs like FAN and ICLN showed a stronger and more significant relationship with TRI earlier in the period. However, this effect seems to diminish over time.
The reduced significance of TRI on these ETFs by 2021 suggests that the sensitivity of these ETFs to transition risk may have lessened.
            ''')