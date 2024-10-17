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


size = (data
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
    .pivot_table(index="date", columns="portfolio", values="ret")
  .assign(size=lambda x: x["small"]-x["big"])
  .get(["size"])
)

value = (data
         .assign(
             booktomarket = lambda x: x["Bv"]/x["Mkt_Cap_12M_Usd"]
         )
 .groupby("date")
  .apply(lambda x: (x.assign(
      portfolio=pd.qcut(
        x["booktomarket"], q=[0, 0.3, 0.7, 1], labels=["low","neutral", "high"]))
    )
  )
  .reset_index(drop=True)
  .groupby(["portfolio","date"])
  .apply(lambda x: np.average(x["R1M_Usd"], weights=x["Mkt_Cap_12M_Usd"]))
  .reset_index(name="ret")
    .pivot_table(index="date", columns="portfolio", values="ret")
  .assign(value=lambda x: x["high"]-x["low"])
  .get(["value"])
)


portfolios = pd.concat([size, value], axis=1)

cum_returns = portfolios.add(1).cumprod() - 1
cum_returns_long = cum_returns.reset_index().melt(id_vars='date', value_vars=['size', 'value'], 
                                                  var_name='factor', value_name='cum_return')

# Plot with different colors for each factor
plot = (
    ggplot(cum_returns_long, aes(x='date', y='cum_return', color='factor'))  # Add color by 'factor'
    + geom_line()  # Use one geom_line to plot both factors
    + labs(title="Size and Value Sorted Portfolios",
           x="Date", y="Cumulative Return")
    + scale_x_datetime(breaks=date_breaks('5 years'), labels=date_format('%Y'))
    + theme(axis_text_x=element_text(rotation=45, hjust=1))
)

plot.show()