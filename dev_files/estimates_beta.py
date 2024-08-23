import pandas as pd 
import numpy as np
import yfinance as yf 

### Download S&P 500 constituents
# URL to the Wikipedia page
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Use pandas to read tables from the URL
sp500_constituents = (pd.read_html(url)[0]
                      .get("Symbol")
                      .tolist()
)


### Download stock prices
ret_monthly = (
    yf.download(
        tickers = sp500_constituents[:5],
        start = '2009-12-31',
        end = '2023-12-31',
        interval = '1mo',
        progress = True
    )
    .stack()
    .reset_index(level = 1, drop = False)
    .reset_index()
    .assign(R = lambda df: df.groupby("Ticker")["Adj Close"].pct_change())
    .dropna()
    .assign(Date = lambda x: pd.to_datetime(x['Date']) + pd.offsets.MonthEnd(0))
    .get(["Date", "Ticker", "R"])
)


print((ret_monthly
       .groupby(['Ticker'])['R'].describe()
       .round(3)
       )
)

### Download Fama-French factors
import pandas_datareader as pdr 
factors_ff3_monthly = (pdr.DataReader(
                        name = 'F-F_Research_Data_5_Factors_2x3',
                        data_source = 'famafrench',
                        start = '2010-01-01',
                        end = '2023-12-31',
                    )[0]
                    .divide(100)
                    .reset_index(names = 'Date')
                    .assign(Date = lambda x: pd.to_datetime(x['Date'].astype(str)) + pd.offsets.MonthEnd(0))
                    .rename(columns = {'Mkt-RF':'Rm_minus_Rf',
                                       'RF':'Rf'})
                    .get(['Date', 'Rm_minus_Rf', 'Rf'])
)

print(factors_ff3_monthly['Rm_minus_Rf']
      .groupby(factors_ff3_monthly['Date'].dt.year)
      .describe()
        .round(3)
)

## Plot Cumulative Market Return
from plotnine import *
from mizani.breaks import date_breaks
from mizani.formatters import date_format

plot_market = (
    ggplot(factors_ff3_monthly.assign(Cumulative_Market_Return = lambda x: (1 + x['Rm_minus_Rf']).cumprod()),
           aes(x = 'Date', y = 'Cumulative_Market_Return')) +
    geom_line() +
    scale_x_datetime(breaks = date_breaks('2 years'), labels = date_format('%Y')) +
    labs(x = "Date", y = "Market Return")
)

plot_market.show() 


### Estimate beta
import statsmodels.formula.api as smf
from regtabletotext import prettify_result

monthly_data = (
    ret_monthly
    .merge(factors_ff3_monthly, on = 'Date', how = 'left')
    .assign(
        R_minus_Rf = lambda x: x['R'] - x['Rf'],
    )
    .get(['Date', 'Ticker', 'R_minus_Rf', 'Rm_minus_Rf'])
)


model_beta = (
    smf.ols("R_minus_Rf ~ Rm_minus_Rf",
    data = monthly_data.query("Ticker == 'ABBV'")
    )
    .fit()
)

print(prettify_result(model_beta))

# # ### Rolling Window
# # from statsmodels.regression.rolling import RollingOLS

# # window_size = 60
# # min_obs = 48 

# # def roll_capm_estimation(data, window_size, min_obs):

# #     data = data.sort_values('Date')

# #     result = (
# #         RollingOLS.from_formula(
# #             formula = "R ~ Rm",
# #             data = data,
# #             window = window_size,
# #             min_nobs = min_obs,
# #             missing = 'drop'   
# #         )
# #         .fit()
# #         .params.get('Rm')
# #     )
# #     result.index = data.index

# #     return result

# # rolling_beta = (
# #     monthly_data
# #     .groupby('Ticker')
# #     .apply(lambda x: x.assign(beta = roll_capm_estimation(x, window_size, min_obs)))
# #     .reset_index(drop = True)
# #     .dropna()
# # )

# # print(rolling_beta.head())

# # from plotnine import *
# # from mizani.breaks import date_breaks
# # from mizani.formatters import percent_format, date_format

# # plot_beta = (
# #     ggplot(rolling_beta,
# #            aes(x = 'Date', y = 'beta', color = 'Ticker')) +
# #     geom_line() +
# #     scale_x_datetime(breaks = date_breaks('2 years'), labels = date_format('%Y')) +
# #     labs(x = "Date", y = "Beta")
# # )

# # plot_beta.show()