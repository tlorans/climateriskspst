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



etfs = {
    "IWRD.L": "iShares MSCI World UCITS ETF",
    "ICLN": "iShares Global Clean Energy ETF",
    "QCLN": "First Trust NASDAQ Clean Edge Green Energy Index Fund",
    "PBW": "Invesco WilderHill Clean Energy ETF",
    "TAN": "Invesco Solar ETF",
    "FAN": "First Trust Global Wind Energy ETF"
}

list_tickers = list(etfs.values())[1:]

name_etf = st.sidebar.selectbox('Select ETF', list_tickers)

# find the corresponding key
ticker = [key for key, value in etfs.items() if value == name_etf][0]

prices = (yf.download(
    tickers=ticker, 
    progress=False
  )
  .reset_index()
  .assign(name = lambda x: name_etf)
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
        title="Adjusted Prices",
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
    .get(["name","date","ret"])
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
       title="Distribution of daily ETF returns") +
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


st.subheader('Resampling and Active Returns')

st.write(r'''
We now want to prepare the data for the multivariate time series regression, 
as in Apel et al. (2023). 
We first need to resample the daily returns to weekly returns.
We also need to calculate the active returns of the ETF, 
that is, the returns net of the benchmark.
         
We can resample the daily returns to weekly returns
by using the `resample()` method from pandas on the price data.
Note that we have added the MSCi World ETF (IWRD.L) to the list of ETFs,
which we use as a benchmark.         
         ''')

st.code(r'''
list_ETFs = ['IWRD.L','ICLN','QCLN','PBW','TAN','FAN']

# Download and process data
prices_weekly = (
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
        .groupby("symbol")["adjusted"] 
        .resample("W-FRI")
        .last()
    )

returns_weekly = (prices_weekly
                  .reset_index()
    .assign(
        ret = lambda x: x.groupby("symbol")["adjusted"].pct_change()
    )
    .get(["symbol", "date", "ret"])
    .dropna(subset="ret")
)
        ''')        

ETF_plus_bench = [ticker] + ['IWRD.L']


# Download and process data
# Function to download and process data with caching
def get_weekly_prices():
    prices = (
        yf.download(
            tickers=ETF_plus_bench, 
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
        .resample("W-FRI")
        .last()
    )
    return prices

prices_weekly = get_weekly_prices()

returns_weekly = (prices_weekly
                  .reset_index()
    .assign(
        name = lambda x: x["symbol"].map(etfs),
        ret = lambda x: x.groupby("name")["adjusted"].pct_change()
    )
    .get(["name", "date", "ret"])
    .dropna(subset="ret")
)


st.write(r'''
The above code downloads the daily prices for the ETFs and the benchmark.
We then calculate the weekly returns and drop the missing values.
An easy way to calculate the active returns is to 
pivot the weekly returns such that we can 
subtract the benchmark returns column from each ETF column.
We then use the `melt()` method from pandas to transform the
data back to a tidy format.
         ''')

st.code(r'''
active_returns = (
    returns_weekly
    .pivot(index="date", columns="name", values="ret")
    .apply(lambda x: x- x["IWRD.L"], axis=1)
    .dropna()
    .reset_index()
    .melt(id_vars="date", var_name="symbol", value_name="active_ret")
)
        ''')

active_returns = (
    returns_weekly
    .pivot(index="date", columns="name", values="ret")
    .apply(lambda x: x- x["iShares MSCI World UCITS ETF"], axis=1)
    .dropna()
    .reset_index()
    .melt(id_vars="date", var_name="name", value_name="active_ret")
)


st.write(r'''
         It may be interesting to visualize the difference between the 
         absolute returns and the active returns. 
         We make use again of the `pivot()` and 
            `melt()` methods from pandas.         ''')


st.code(r'''
 cum_absolute_returns = (
    returns_weekly
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
    returns_weekly
    .pivot(index="date", columns="name", values="ret")
    .dropna()
    .apply(
        lambda x: (1 + x).cumprod() - 1
    )
    .reset_index()
    .melt(id_vars="date", var_name="name", value_name="cum_ret")
)



cum_absolute_returns_figure = (
    ggplot(cum_absolute_returns, 
         aes(y="cum_ret", x="date", color = "name")) +
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
    .pivot(index="date", columns="name", values="active_ret")
    .apply(
        lambda x: (1 + x).cumprod() - 1
    )
    .reset_index()
    .melt(id_vars="date", var_name="name", value_name = "cum_ret")
)

cum_active_returns_figure = (
    ggplot(cum_active_returns, 
         aes(y="cum_ret", x="date", color="name")) +
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
         We start by downloading Fama-French factors (Fama and French, 1993, 2015).
         We use the `pandas_datareader` package that provides a simple interface to 
         read data from Kenneth French's website.
         We use the `pd.DataReader()` function of the package to download daily Fama-French factors,
            which we then resample to weekly frequency.
         ''')


st.code(r'''
import pandas_datareader as pdr
# min date in active returns df as start date
start_date = active_returns["date"].min()
# max date in active returns df as end date
end_date = active_returns["date"].max()

factors_ff5_dailt_raw = pdr.DataReader(
  name="F-F_Research_Data_5_Factors_2x3",
  data_source="famafrench", 
  start=start_date, 
  end=end_date)[0]

factors_ff5_weekly = (factors_ff5_dailt_raw
  .divide(100)
  .resample('W-FRI')
  .apply(lambda x: (1 + x).prod() - 1)
  .reset_index(names="date")
  .assign(
        date=lambda x: pd.to_datetime(x["date"].astype(str))
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
def get_ff_daily() -> pd.DataFrame:
    return pdr.DataReader(
    name="F-F_Research_Data_5_Factors_2x3_daily",
    data_source="famafrench", 
    start=start_date, 
    end=end_date)[0]


factors_ff5_weekly = (get_ff_daily()
  .divide(100)
  .resample('W-FRI')
  .apply(lambda x: (1 + x).prod() - 1)
  .reset_index(names="date")
  # to date time and end of month
  .assign(
        date=lambda x: pd.to_datetime(x["date"].astype(str))
      )
  .rename(str.lower, axis="columns")
  .rename(columns={"mkt-rf": "mkt_excess"})
)

st.dataframe(factors_ff5_weekly.head().round(3))


st.write(r'''
         We can now visualize the cumulative returns of the Fama-French factors.
         ''')

st.code(r'''
cum_returns_factors = (
    factors_ff5_weekly 
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
    factors_ff5_weekly 
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
tri = (pd.read_excel('data/tri.xlsx', sheet_name='weekly')
      .rename(str.lower, axis="columns")
      .get(['date','tri_innovation_weekly'])
)
        ''')
tri = (pd.read_excel('data/tri.xlsx', sheet_name='weekly')
      .rename(str.lower, axis="columns")
      .get(['date','tri_innovation_weekly'])
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
    ggplot(cum_tri, aes(y="tri_innovation_weekly", x="date")) +
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
    ggplot(cum_tri, aes(y="tri_innovation_weekly", x="date")) +
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
         without the momentum factor. 

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
    .merge(factors_ff5_weekly, on="date", how = "inner")
    .merge(tri, on="date", how = "inner")
)

icln = data_for_reg.query('symbol == "ICLN"')

model_beta = (
    smf.ols("active_ret ~ mkt_excess + smb + hml + rmw + cma + tri_innovation_weekly", data=icln)
    .fit()
)

model_beta.summary()
        ''')
import statsmodels.formula.api as smf

data_for_reg = (
    active_returns.query('name != "iShares MSCI World UCITS ETF"')
    .merge(factors_ff5_weekly, on="date", how = "inner")
    .merge(tri, on="date", how = "inner")
)


model_beta = (
    smf.ols("active_ret ~ mkt_excess + smb + hml + rmw + cma + tri_innovation_weekly", 
            data=data_for_reg)
    .fit()
)

st.write(model_beta.summary())

st.write(r'''
`sm.ols()` returns a `RegressionResults` object that 
         contains the estimated coefficients,
            standard errors, t-values, and p-values.

We are mostly interested by:
- the sign of the coefficient of the TRI innovation, which indicates the direction of the relationship between the TRI innovation and the active returns.
- the p-value of the coefficient of the TRI innovation, which indicates the statistical significance of the relationship between the TRI innovation and the active returns.

The coefficient of the TRI innovation is positive,
incicating that when concerns about transition risks increase, 
         the active returns of the ICLN ETF increase. 
         This is in line with green stocks 
outperforming in Pastor et al. (2021, 2022) when 
         climate concerns increase.

The p-value of the coefficient of the TRI innovation is low for most ETFs. 
This indicates a strong statistical relationship between the TRI innovation and the active returns of the 
green ETF.
''')

st.write(r'''
We now scale the estimation of the model performing rolling-window estimation.
         As in Apel et al. (2023), we use a window length of 5 years.
The following function implements the regression.
                     ''')

st.code(r'''
from statsmodels.regression.rolling import RollingOLS

window_size = 5 * 52
min_obs =  5 * 50
# Function to estimate rolling t-statistics
def roll_pvalue_estimation(data, window_size, min_obs):
    data = data.sort_values("date")

    # Fit Rolling OLS and extract t-stats for tri_innovation_weekly
    result = pd.Series((RollingOLS.from_formula(
        formula="active_ret ~ mkt_excess + smb + hml + rmw + cma + tri_innovation_weekly",
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

window_size = 5 * 52
min_obs = 5 * 50
# Function to estimate rolling t-statistics
def roll_pvalue_estimation(data, window_size, min_obs):
    data = data.sort_values("date")

    # Fit Rolling OLS and extract t-stats for tri_innovation_weekly
    result = pd.Series((RollingOLS.from_formula(
        formula="active_ret ~ mkt_excess + smb + hml + rmw + cma + tri_innovation_weekly",
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
    data_for_reg.query('name != "iShares MSCI World UCITS ETF"')
    .groupby("name")
    .apply(lambda x: x.assign(
        pvalue=roll_pvalue_estimation(x, window_size, min_obs)
    ))
    .reset_index(drop=True)
    .dropna()
)


st.write(r'''
         In the above function, we retrieve the p-values of the
            regression coefficient of the TRI innovation.
         We want to know, among our ETFs, which ones has the most significant relationship with the TRI innovation,
         and if this relationship has changed over time.
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
    geom_vline(xintercept=pd.to_datetime("2015-12-01"), linetype="dashed") +
    geom_vline(xintercept=pd.to_datetime("2017-06-01"), linetype="dashed") +
    labs(x="Date", y="p-value", color="Index",
         title="Rolling 5-Year Regression p-values for TRI Innovation Coefficient") +
    scale_x_datetime(date_breaks="1 year", date_labels="%Y")
)

figures_pvalues.draw()
        ''')

# Plot the rolling t-statistics with dashed lines for significance levels
figures_pvalues = (
    ggplot(rolling_pvalues, aes(x="date", y="pvalue")) +
    geom_line() +
    geom_hline(yintercept=0.1, linetype="dashed", color="red") +
    geom_hline(yintercept=0.05, linetype="dashed", color="red") +
    geom_hline(yintercept=0.01, linetype="dashed", color="red") +
    geom_vline(xintercept=pd.to_datetime("2015-12-01"), linetype="dashed") +
    geom_vline(xintercept=pd.to_datetime("2017-06-01"), linetype="dashed") +
    labs(x="Date", y="p-value", color="ETF",
         title="Rolling 5-Year Regression p-values for TRI Innovation Coefficient") +
    scale_x_datetime(date_breaks="1 year", date_labels="%Y")
)

st.pyplot(figures_pvalues.draw())
st.write(r'''
This chart shows the rolling 5-year regression p-values for the TRI Innovation Coefficient.

P-values measure the statistical significance of the relationship between the TRI (Transition Risk Innovation) coefficient and the active returns of each ETF. Lower p-values indicate a more statistically significant relationship, while higher p-values suggest weaker evidence of a relationship.

The red dashed lines represent different significance levels for the p-values:
- 0.1 for 10% significance.
- 0.05 for 5% significance.
- 0.01 for 1% significance.

We have also added vertical dashed lines 
            for the Paris Agreement in 2015 and the US withdrawal from the Paris Agreement in 2017.
Interestingly, the p-values of the TRI innovation coefficient
loss significance after the US withdrawal from the Paris Agreement in 2017 for most 
ETFs, indicating a weaker relationship between the TRI innovation and the active returns.
It may be interpreted as a loss of interest in green investments after the US withdrawal from the Paris Agreement.
''')


st.subheader('Conclusion')

st.write(r'''
Following Pastor et al. (2021, 2022) findings, we were looking for 
an investable portfolio such as ETFs that could serve as a proxy for the green factor.
To do so, we have analyzed the relationship between unexpected changes in transition concerns
and the active returns of green ETFs, in a similar vein than Apel et al. (2023).
We have found that the relationship between the TRI innovation and the active returns of the green ETFs
is statistically significant and positive. 
Therefore, the green ETFs are an investable proxy for the green factor.
''')

st.subheader('Exercice')

st.write(r'''

We haven't completed the analysis of the ETFs yet. In the spirit 
of Apel et al. (2023), we also want to compare decarbonized ETFs
to the pure-play indices ETFs in terms of the relationship between the TRI innovation and the active returns.

You can complete the following tasks:
1. Find ETFs tracking the Decarbonized indices. Download the daily prices and calculate the weekly returns.
2. Estimate the active returns of the Decarbonized ETFs.
3. Estimate the relationship between the TRI innovation and the active returns of the Decarbonized ETFs.
4. Make a bar plot of the p-values of the TRI innovation coefficient for the Decarbonized ETFs.
5. Compute the average of p-values for the Decarbonized ETFs and compare it to the average of p-values for the pure-play indices ETFs.
''')