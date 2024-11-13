import streamlit as st
from plotnine import *
from mizani.formatters import date_format
from mizani.breaks import date_breaks
from mizani.formatters import percent_format
import numpy as np
import pandas as pd
import yfinance as yf
from PIL import Image


st.title('Long-Short Strategy')

st.write(r'''
We have an investable proxy for the green factor, and 
a model that seems to be able to infer the transition regimes.
More precisely, we have a signal indicating a bull or bear market for 
our green ETF. 
We therefore have all the ingredients to implement an investment strategy.
We are going to implement a simple long-short strategy, conditional on the
signal we have.
''')


st.subheader('Backtesting Protocol')

st.write(r'''
We will define a buffer period and a backtest period for our strategy backtesting:

1. **Buffer Period**: In our case, we 
         use a four-year buffer period to train the Jump Model. 
We make the (strong) hypothesis that the four-year period provides enough 
historical data to reliably detect bull and bear regimes.

2. **Backtest Period**: After the buffer period, 
         we enter the backtest phase, 
         which lasts for one year. 
         During this period, the inferred bull 
         and bear regimes are used to actively 
         adjust the portfolio based on the signals. 
         The strategy dynamically 
         changes positions as new signals are generated, 
         allowing us to evaluate its real-time 
         decision-making effectiveness.

3. **Rolling Forward**: After each backtest year, 
         we roll the window forward by one year. 
         This means that the model is retrained 
         using the most recent four years of data, 
         and the inferred regimes are then used for 
         the next one-year backtest period. 
''')

list_ETFs = ['IWRD.L','ICLN']

# Download and process data
# Function to download and process data with caching
@st.cache_data
def get_daily_prices():
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
    )
    return prices


returns_daily = (get_daily_prices()
                  .reset_index()
    .assign(
        ret = lambda x: x.groupby("symbol")["adjusted"].pct_change()
    )
    .get(["symbol", "date", "ret"])
    .dropna(subset="ret")
)

active_returns = (
    returns_daily
    .pivot(index="date", columns="symbol", values="ret")
    .apply(lambda x: x- x["IWRD.L"], axis=1)
    .dropna()
    .reset_index()
    .melt(id_vars="date", var_name="symbol", value_name="active_ret")
)

ret_ser = (
    active_returns
    .pivot(index="date", columns="symbol", values="active_ret")
    .dropna()
    .ICLN
)


def compute_ewm_DD(ret_ser: pd.Series, hl: float) -> pd.Series:
    """
    Compute the exponentially weighted moving downside deviation (DD) for a return series.

    The downside deviation is calculated as the square root of the exponentially 
    weighted second moment of negative returns.

    Parameters
    ----------
    ret_ser : pd.Series
        The input return series.

    hl : float
        The halflife parameter for the exponentially weighted moving average.

    Returns
    -------
    pd.Series
        The exponentially weighted moving downside deviation for the return series.
    """
    ret_ser_neg: pd.Series = np.minimum(ret_ser, 0.)
    sq_mean = ret_ser_neg.pow(2).ewm(halflife=hl).mean()
    return np.sqrt(sq_mean)

def feature_engineer(ret_ser: pd.Series, ver: str = "v0") -> pd.DataFrame:
    """
    Engineer a set of features based on a return series.

    This function customizes the feature set according to the specified version string.

    Parameters
    ----------
    ret_ser : pd.Series
        The input return series for feature engineering.

    ver : str
        The version of feature engineering to apply. Only supports "v0".
    
    Returns
    -------
    pd.DataFrame
        The engineered feature set.
    """
    if ver == "v0":
        feat_dict = {}
        hls = [5, 20, 60]
        for hl in hls:
            # Feature 1: EWM-ret
            feat_dict[f"ret_{hl}"] = ret_ser.ewm(halflife=hl).mean()
            # Feature 2: log(EWM-DD)
            DD = compute_ewm_DD(ret_ser, hl)
            feat_dict[f"DD-log_{hl}"] = np.log(DD)
            # Feature 3: EWM-Sortino-ratio = EWM-ret/EWM-DD 
            feat_dict[f"sortino_{hl}"] = feat_dict[f"ret_{hl}"].div(DD)
        return pd.DataFrame(feat_dict)

    # try out your favorite feature sets
    else:
        raise NotImplementedError()
    
X = (ret_ser
     .pipe(feature_engineer, ver="v0")
     # replace inf values with NaN and drop them
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
)

st.write(r'''
To implement this, we first need to turn our code in the previous 
section into a function that trains the Jump Model on the training period 
and predicts the regimes on the test period.
         ''')

st.code(r'''
from jumpmodels.preprocess import StandardScalerPD, DataClipperStd
from jumpmodels.jump import JumpModel                 # class of JM & CJM

def train_then_predict(X:pd.DataFrame, 
                        ret_ser:pd.Series,
                       train_start:str, 
                       test_start:str, 
                       test_end:str,
                       jump_penalty:float):
    X_train, X_test = X[train_start:test_start], X[test_start:test_end]

    clipper = DataClipperStd(mul=3.) # clip the data at 3 std. dev.
    scalar = StandardScalerPD() # standardize the data
    # fit on training data
    X_train_processed = (X_train 
                        .pipe(clipper.fit_transform)
                        .pipe(scalar.fit_transform)
                            )
    # transform the test data
    X_test_processed = (
        X_test
        .pipe(clipper.transform)
        .pipe(scalar.transform)
    )
    jm = JumpModel(n_components=2, jump_penalty=jump_penalty, cont=False, )
    jm.fit(X_train_processed, ret_ser, sort_by="cumret")
    return jm.predict(X_test_processed)
        ''')

from jumpmodels.preprocess import StandardScalerPD, DataClipperStd
from jumpmodels.jump import JumpModel                 # class of JM & CJM

def train_then_predict(X:pd.DataFrame, 
                        ret_ser:pd.Series,
                       train_start:str, 
                       test_start:str, 
                       test_end:str,
                       jump_penalty:float):
    X_train, X_test = X[train_start:test_start], X[test_start:test_end]

    clipper = DataClipperStd(mul=3.) # clip the data at 3 std. dev.
    scalar = StandardScalerPD() # standardize the data
    # fit on training data
    X_train_processed = (X_train 
                        .pipe(clipper.fit_transform)
                        .pipe(scalar.fit_transform)
                            )
    # transform the test data
    X_test_processed = (
        X_test
        .pipe(clipper.transform)
        .pipe(scalar.transform)
    )
    jm = JumpModel(n_components=2, jump_penalty=jump_penalty, cont=False, )
    jm.fit(X_train_processed, ret_ser, sort_by="cumret")
    return jm.predict(X_test_processed)

st.write(r'''
We then can implement the rolling backtest protocol with 
a simple while loop that iterates over the training and testing periods.
         ''')

st.code(r'''
import datetime as dt

# Define start and end dates for your rolling backtest
start_date = "2010-01-01"
end_date = "2025-01-01"

# Convert to datetime
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Initialize variables
train_period = pd.DateOffset(years=4)
test_period = pd.DateOffset(years=1)
current_train_start = start_date
current_test_start = current_train_start + train_period
current_test_end = current_test_start + test_period

results = []

# Rolling forward until we reach the end date
while current_test_end <= end_date:
    # Convert dates to strings for indexing
    train_start_str = current_train_start.strftime('%Y-%m-%d')
    test_start_str = current_test_start.strftime('%Y-%m-%d')
    test_end_str = current_test_end.strftime('%Y-%m-%d')

    st.write(f"Training from {train_start_str} to {test_start_str}, then testing from {test_start_str} to {test_end_str}")

    # Train the model and predict
    try:
        predictions = train_then_predict(X, ret_ser, train_start_str, test_start_str, test_end_str, jump_penalty=30)
        results.append(predictions)
    except Exception as e:
        st.write(f"An error occurred during training and prediction: {e}")

    # Move the window forward by 1 year
    current_train_start += test_period
    current_test_start = current_train_start + train_period
    current_test_end = current_test_start + test_period

# Concatenate all the results into a single DataFrame
final_results = pd.concat(results).reset_index().rename(columns={'index': 'date', 0: 'regime'})

# Plotting the predictions
final_results['date'] = pd.to_datetime(final_results['date'])

        ''')

import datetime as dt

# Define start and end dates for your rolling backtest
start_date = "2010-01-01"
end_date = "2025-01-01"

# Convert to datetime
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Initialize variables
train_period = pd.DateOffset(years=4)
test_period = pd.DateOffset(years=1)
current_train_start = start_date
current_test_start = current_train_start + train_period
current_test_end = current_test_start + test_period

results = []

# Rolling forward until we reach the end date
while current_test_end <= end_date:
    # Convert dates to strings for indexing
    train_start_str = current_train_start.strftime('%Y-%m-%d')
    test_start_str = current_test_start.strftime('%Y-%m-%d')
    test_end_str = current_test_end.strftime('%Y-%m-%d')

    st.write(f"Training from {train_start_str} to {test_start_str}, then testing from {test_start_str} to {test_end_str}")

    # Train the model and predict
    try:
        predictions = train_then_predict(X, ret_ser, train_start_str, test_start_str, test_end_str, jump_penalty=30)
        results.append(predictions)
    except Exception as e:
        st.write(f"An error occurred during training and prediction: {e}")

    # Move the window forward by 1 year
    current_train_start += test_period
    current_test_start = current_train_start + train_period
    current_test_end = current_test_start + test_period

# Concatenate all the results into a single DataFrame
final_results = pd.concat(results).reset_index().rename(columns={'index': 'date', 0: 'regime'})

# Plotting the predictions
final_results['date'] = pd.to_datetime(final_results['date'])


st.write(r'''
Similarly than in the previous section, we can plot the results of the backtest.
         ''')

st.code(r'''
# Calculate the 252-day rolling average of returns
ret_ser_rolling = ret_ser.rolling(window=126).mean().reset_index()
ret_ser_rolling.columns = ["date", "rolling_avg_return"]
# restrict the rolling average
start_date_for_rolling = final_results['date'].min()
ret_ser_rolling = ret_ser_rolling.query('date >= @start_date_for_rolling')

# Calculate ymin and ymax based on rolling average returns data
ymin = ret_ser_rolling["rolling_avg_return"].min()
ymax = ret_ser_rolling["rolling_avg_return"].max()
std = ret_ser_rolling["rolling_avg_return"].std()

# Prepare the regimes DataFrame to get the start and end dates of each regime period
regimes = (
    final_results
    .assign(
        label=lambda x: x["regime"].map({0: "Bull", 1: "Bear"})
    )
)

# Define start and end dates for each regime type
regime_highlights = (
    regimes.groupby((regimes['label'] != regimes['label'].shift()).cumsum())
    .agg(start_date=('date', 'first'), end_date=('date', 'last'), label=('label', 'first'))
    .assign(ymin=ymin - std, ymax=ymax + std)  # Set ymin and ymax dynamically
)

# Plot regimes with rolling average line
p = (
    ggplot() +
    # Regime shaded areas using geom_rect with dynamic ymin and ymax
    geom_rect(regime_highlights, aes(
        xmin='start_date', xmax='end_date', ymin='ymin', ymax='ymax', fill='label'
    ), alpha=0.3) +
    # Rolling average line plot
    geom_line(ret_ser_rolling, aes(x='date', y='rolling_avg_return')) +
    geom_hline(yintercept=0, linetype="dashed") + 
    labs(y="Rolling Avg Return", x="") +
    scale_fill_manual(values={"Bull": "green", "Bear": "red"}) +  # Green for Bull, Red for Bear
    scale_x_datetime(breaks=date_breaks("1 year"), labels=date_format("%Y")) +
    theme(
        axis_text_x=element_text(angle=45, hjust=1),
        legend_position="none"  # Hide legend if it distracts from the main plot
    )
)

ggplot.draw(p)
        ''')

# Calculate the 252-day rolling average of returns
ret_ser_rolling = ret_ser.rolling(window=126).mean().reset_index()
ret_ser_rolling.columns = ["date", "rolling_avg_return"]
# restrict the rolling average
start_date_for_rolling = final_results['date'].min()
ret_ser_rolling = ret_ser_rolling.query('date >= @start_date_for_rolling')

# Calculate ymin and ymax based on rolling average returns data
ymin = ret_ser_rolling["rolling_avg_return"].min()
ymax = ret_ser_rolling["rolling_avg_return"].max()
std = ret_ser_rolling["rolling_avg_return"].std()

# Prepare the regimes DataFrame to get the start and end dates of each regime period
regimes = (
    final_results
    .assign(
        label=lambda x: x["regime"].map({0: "Bull", 1: "Bear"})
    )
)

# Define start and end dates for each regime type
regime_highlights = (
    regimes.groupby((regimes['label'] != regimes['label'].shift()).cumsum())
    .agg(start_date=('date', 'first'), end_date=('date', 'last'), label=('label', 'first'))
    .assign(ymin=ymin - std, ymax=ymax + std)  # Set ymin and ymax dynamically
)

# Plot regimes with rolling average line
p = (
    ggplot() +
    # Regime shaded areas using geom_rect with dynamic ymin and ymax
    geom_rect(regime_highlights, aes(
        xmin='start_date', xmax='end_date', ymin='ymin', ymax='ymax', fill='label'
    ), alpha=0.3) +
    # Rolling average line plot
    geom_line(ret_ser_rolling, aes(x='date', y='rolling_avg_return')) +
    geom_hline(yintercept=0, linetype="dashed") + 
    labs(y="Rolling Avg Return", x="") +
    scale_fill_manual(values={"Bull": "green", "Bear": "red"}) +  # Green for Bull, Red for Bear
    scale_x_datetime(breaks=date_breaks("1 year"), labels=date_format("%Y")) +
    theme(
        axis_text_x=element_text(angle=45, hjust=1),
        legend_position="none"  # Hide legend if it distracts from the main plot
    )
)

st.pyplot(ggplot.draw(p))

st.write(r'''
It looks quite good! Training the model over 4 years
seem to be sufficient to capture the bull and bear regimes in the online phase.
         ''')

st.subheader('Turning Signals into Positions')  

st.write(r'''
We now have a signal indicating the bull and bear regimes for the 
         green factor. 
We can use this signal to construct a long-short strategy:
         ''')

st.latex(r'''
  \begin{equation}
    r_{ls,t} =
    \begin{cases}
      r_{green, t},  & \text{if in bull regime in } t-1 \\
      -r_{green, t}, & \text{if in bear regime in } t-1
    \end{cases}
  \end{equation}
''')

st.code(r'''
# long the etf when the model predicts a bull regime, short when it predicts a bear regime
long_short = (
    final_results
    .merge(
        ret_ser.reset_index(name="ICLN"),
        on="date",
        how = 'inner'
    )
    # we need to shift the signal (regime) by one day to avoid look-ahead bias
    .assign(
        regime=lambda x: x["regime"].shift(1)
    )
    .assign(
        # - clean energy when regime == 1 (bear), clean energy when regime == 0 (bull)
        long_short = lambda x: np.where(x["regime"] == 1, -x["ICLN"], x["ICLN"])
    )
)
        ''')
        

# long the etf when the model predicts a bull regime, short when it predicts a bear regime
long_short = (
    final_results
    .merge(
        ret_ser.reset_index(name="ICLN"),
        on="date",
        how = 'inner'
    )
    # we need to shift the signal (regime) by one day to avoid look-ahead bias
    .assign(
        regime=lambda x: x["regime"].shift(1)
    )
    .assign(
        # - clean energy when regime == 1 (bear), clean energy when regime == 0 (bull)
        long_short = lambda x: np.where(x["regime"] == 1, -x["ICLN"], x["ICLN"])
    )
)

st.write(r'''
         The code above creates a DataFrame 
         with the long-short strategy returns, 
         which are based on the signal from the Jump Model.
         We take care 
         to shift the signal by one day to avoid look-ahead bias.
         Similarly to the previous section, we can plot the
            performance of the long-short strategy.
         ''')

st.code(r'''

# rolling return of the long_short
rolling_ls = (
    long_short
    .get(["date", "long_short"])
    .set_index("date")
    .rolling(window=126).mean().reset_index()
    .dropna()
    .rename(columns={"long_short": "rolling_avg_return"})
)

# Calculate ymin and ymax based on rolling average returns data
ymin = rolling_ls["rolling_avg_return"].min()
ymax = rolling_ls["rolling_avg_return"].max()
std = rolling_ls["rolling_avg_return"].std()

# Define start and end dates for each regime type
regime_highlights = (
    regimes.groupby((regimes['label'] != regimes['label'].shift()).cumsum())
    .agg(start_date=('date', 'first'), end_date=('date', 'last'), label=('label', 'first'))
    .assign(ymin=ymin - std, ymax=ymax + std)  # Set ymin and ymax dynamically
)

# Plot regimes with rolling average line
p_signal = (
    ggplot() +
    # Regime shaded areas using geom_rect with dynamic ymin and ymax
    geom_rect(regime_highlights, aes(
        xmin='start_date', xmax='end_date', ymin='ymin', ymax='ymax', fill='label'
    ), alpha=0.3) +
    # Rolling average line plot
    geom_line(rolling_ls, aes(x='date', y='rolling_avg_return')) +
    geom_hline(yintercept=0, linetype="dashed") + 
    labs(y="Rolling Avg Return", x="") +
    scale_fill_manual(values={"Bull": "green", "Bear": "red"}) +  # Green for Bull, Red for Bear
    scale_x_datetime(breaks=date_breaks("1 year"), labels=date_format("%Y")) +
    theme(
        axis_text_x=element_text(angle=45, hjust=1),
        legend_position="none"  # Hide legend if it distracts from the main plot
    )
)

ggplot.draw(p_signal)
        ''')       

# rolling return of the long_short
rolling_ls = (
    long_short
    .get(["date", "long_short"])
    .set_index("date")
    .rolling(window=126).mean().reset_index()
    .dropna()
    .rename(columns={"long_short": "rolling_avg_return"})
)

# Calculate ymin and ymax based on rolling average returns data
ymin = rolling_ls["rolling_avg_return"].min()
ymax = rolling_ls["rolling_avg_return"].max()
std = rolling_ls["rolling_avg_return"].std()

# Define start and end dates for each regime type
regime_highlights = (
    regimes.groupby((regimes['label'] != regimes['label'].shift()).cumsum())
    .agg(start_date=('date', 'first'), end_date=('date', 'last'), label=('label', 'first'))
    .assign(ymin=ymin - std, ymax=ymax + std)  # Set ymin and ymax dynamically
)

# Plot regimes with rolling average line
p_signal = (
    ggplot() +
    # Regime shaded areas using geom_rect with dynamic ymin and ymax
    geom_rect(regime_highlights, aes(
        xmin='start_date', xmax='end_date', ymin='ymin', ymax='ymax', fill='label'
    ), alpha=0.3) +
    # Rolling average line plot
    geom_line(rolling_ls, aes(x='date', y='rolling_avg_return')) +
    geom_hline(yintercept=0, linetype="dashed") + 
    labs(y="Rolling Avg Return", x="") +
    scale_fill_manual(values={"Bull": "green", "Bear": "red"}) +  # Green for Bull, Red for Bear
    scale_x_datetime(breaks=date_breaks("1 year"), labels=date_format("%Y")) +
    theme(
        axis_text_x=element_text(angle=45, hjust=1),
        legend_position="none"  # Hide legend if it distracts from the main plot
    )
)

st.pyplot(ggplot.draw(p_signal))

st.write(r'''
         The rolling average returns of the long-short strategy
         are mostly positive across the bull and bear regimes.
            The strategy seems to be able to capture the
            regime changes effectively. We can also see if the strategy
         was profitable over the period by plotting the cumulative returns.
            ''')

st.code(r'''
cum_ret = (
    long_short
    .get(["date", "long_short"])
    .set_index("date")
    .assign(
        long_short=lambda x: (1 + x["long_short"]).cumprod() - 1
    )
    .reset_index()
)

plot_cum_ret = (
    ggplot(cum_ret) +
    geom_line(aes(x="date", y="long_short")) +
    labs(y="Cumulative Return", x="") +
    scale_x_datetime(breaks=date_breaks("1 year"), labels=date_format("%Y")) +
    theme(axis_text_x=element_text(angle=45, hjust=1))
)

ggplot.draw(plot_cum_ret)
        ''')

cum_ret = (
    long_short
    .get(["date", "long_short"])
    .set_index("date")
    .assign(
        long_short=lambda x: (1 + x["long_short"]).cumprod() - 1
    )
    .reset_index()
)

plot_cum_ret = (
    ggplot(cum_ret) +
    geom_line(aes(x="date", y="long_short")) +
    labs(y="Cumulative Return", x="") +
    scale_x_datetime(breaks=date_breaks("1 year"), labels=date_format("%Y")) +
    theme(axis_text_x=element_text(angle=45, hjust=1))
)

st.pyplot(ggplot.draw(plot_cum_ret))


st.subheader('Performance Metrics')

st.write(r'''
            We can evaluate the performance of the long-short strategy
            by calculating the average return, the standard deviation of the return,
            and the Sharpe ratio.
         
            The Sharpe ratio is a measure of risk-adjusted return,
            which is calculated as the ratio of the average return to the standard deviation.
            A higher Sharpe ratio indicates a better risk-adjusted return.
            ''')

st.code(r'''
def evaluate_performance(ret_strats,
                       length_year=252):
    """Calculate portfolio evaluation measures."""

    evaluation_stats = (ret_strats
        .groupby("strat")["ret"]
        .aggregate([
          ("Average return", lambda x: np.mean(length_year*x)*100),
          ("SD return", lambda x: np.std(x)*np.sqrt(length_year)*100),
          ("Sharpe ratio", lambda x: (np.mean(x)/np.std(x)* 
                                        np.sqrt(length_year)))
        ])
    )
        
    evaluation_stats = (evaluation_stats
      .transpose()
      .rename_axis(columns=None)
    )

    return evaluation_stats
        ''')

def evaluate_performance(ret_strats,
                       length_year=252):
    """Calculate portfolio evaluation measures."""

    evaluation_stats = (ret_strats
        .groupby("strat")["ret"]
        .aggregate([
          ("Average return", lambda x: np.mean(length_year*x)*100),
          ("SD return", lambda x: np.std(x)*np.sqrt(length_year)*100),
          ("Sharpe ratio", lambda x: (np.mean(x)/np.std(x)* 
                                        np.sqrt(length_year)))
        ])
    )
        
    evaluation_stats = (evaluation_stats
      .transpose()
      .rename_axis(columns=None)
    )

    return evaluation_stats

st.write(r'''The code above implements the function `evaluate_performance` that calculates our performance metrics.''')


st.write(r'''We add the market return to the DataFrame and evaluate the performance of the long-short strategy,
        the benchmark and the initial green ETF.''')

st.code(r'''
market = (returns_daily
          .query("symbol == 'IWRD.L'")
          .set_index("date")
          .ret
          .reset_index(name = "IWRD.L")
)

ret_strats = (
    long_short 
    .drop(columns = ['regime'])
    .merge(market, on = 'date', how = 'inner')
    .melt(id_vars = ['date'], var_name = 'strat', value_name = 'ret')
)


perf = evaluate_performance(ret_strats)
        ''')

market = (returns_daily
          .query("symbol == 'IWRD.L'")
          .set_index("date")
          .ret
          .reset_index(name = "IWRD.L")
)

ret_strats = (
    long_short 
    .drop(columns = ['regime'])
    .merge(market, on = 'date', how = 'inner')
    .melt(id_vars = ['date'], var_name = 'strat', value_name = 'ret')
)


perf = evaluate_performance(ret_strats)
perf

st.write(r'''
Our long-short strategy significantly outperforms the benchmark and the initial green ETF.
         ''')

st.subheader('Conclusion')

st.write(r'''
In this section, we implemented a long-short strategy based on the Jump Model signal.
We trained the model over a four-year period and then used the inferred regimes to construct the strategy.
The strategy dynamically changes positions based on the signal, allowing us to evaluate its performance in real-time.
We found that the long-short strategy significantly outperformed the benchmark and the initial green ETF.
         This demonstrates that investors can benefit from the green factor outperformance (underperformance) regimes
            by actively adjusting their portfolio positions.
            ''')



st.subheader('Exercices')

st.write(r'''
1. **Backtesting Protocol**: Implement the backtesting protocol with a different buffer period and backtest period.
2. **Performance Metrics**: Implement a function that calculates the maximum drawdown of a return series.
''')
