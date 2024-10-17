import pandas as pd
import numpy as np
from plotnine import * 
from mizani.formatters import date_format, percent_format
from mizani.breaks import date_breaks

data = (pd.read_csv('data/data.csv', sep=';', decimal=',')
        .assign(
            date = lambda x: pd.to_datetime(x['date'], format='%d/%m/%Y')
        )
)

print(data.head())
print(data.columns)


portfolios = (data
  .groupby("date")
  .apply(lambda x: (x.assign(
      portfolio=pd.qcut(
        x["Mkt_Cap_12M_Usd"], q=[0, 0.3, 0.7, 1], labels=["small","medium", "big"]))
    )
  )
  .reset_index(drop=True)
  .groupby(["portfolio","date"])
  .apply(lambda x: np.average(x["R1M_Usd"], weights=x["Mkt_Cap_12M_Usd"]))
  .reset_index(name="ret")
)

print(portfolios.head()) 

portfolios_longshort = (portfolios
  .pivot_table(index="date", columns="portfolio", values="ret")
  .assign(long_short=lambda x: x["small"]-x["big"])
)

print(portfolios_longshort.head())

cum_returns = portfolios_longshort['long_short'].add(1).cumprod().subtract(1)

plot = (ggplot(cum_returns.reset_index(), aes(x='date', y='long_short')) +
    geom_line() +
    labs(title="Cumulative Returns of Long-Short Portfolio",
         x="Date", y="Cumulative Return") +
    scale_x_datetime(breaks=date_breaks('5 years'), labels=date_format('%Y'))
    # scale_y_continuous(labels=percent_format())
)

plot.show()            