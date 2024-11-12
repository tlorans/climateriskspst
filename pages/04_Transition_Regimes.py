import streamlit as st
from plotnine import *
from mizani.formatters import date_format
from mizani.breaks import date_breaks
from mizani.formatters import percent_format
import numpy as np
import pandas as pd
import yfinance as yf

st.title('Transition Regimes')



st.subheader('Green Stocks Bull and Bear Markets')

st.code('''

list_ETFs = ['IWRD.L','ICLN']

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


returns_daily = (prices
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

# 252 day moving average

# Compute the 252-day moving average of the active returns
returns_daily_ma = (
    active_returns
    .assign(
        active_ret_ma=lambda x: x.groupby("symbol")["active_ret"].transform(
            lambda x: x.rolling(window=252).mean()
        )
    )
    .dropna(subset=["active_ret_ma"])
    .pivot(index="date", columns="symbol", values="active_ret_ma")
    .get(["ICLN"])
)

# plot the the moving average, add a horizontal line at 0
plot_ma = (
    ggplot(returns_daily_ma.reset_index(), aes(x="date", y="ICLN"))
    + geom_line()
    + geom_hline(yintercept=0, linetype="dashed")
    + scale_x_datetime(breaks=date_breaks("1 year"), labels=date_format("%Y"))
    + labs(
        x="",
        y="126-day Moving Average",
        title=""
    )
    + theme(axis_text_x=element_text(angle=45, hjust=1))
)

ggplot.draw(plot_ma)
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

# 252 day moving average

# Compute the 252-day moving average of the active returns
returns_daily_ma = (
    active_returns
    .assign(
        active_ret_ma=lambda x: x.groupby("symbol")["active_ret"].transform(
            lambda x: x.rolling(window=252).mean()
        )
    )
    .dropna(subset=["active_ret_ma"])
    .pivot(index="date", columns="symbol", values="active_ret_ma")
    .get(["ICLN"])
)

# plot the the moving average, add a horizontal line at 0
plot_ma = (
    ggplot(returns_daily_ma.reset_index(), aes(x="date", y="ICLN"))
    + geom_line()
    + geom_hline(yintercept=0, linetype="dashed")
    + scale_x_datetime(breaks=date_breaks("1 year"), labels=date_format("%Y"))
    + labs(
        x="",
        y="126-day Moving Average",
        title=""
    )
    + theme(axis_text_x=element_text(angle=45, hjust=1))
)

st.pyplot(ggplot.draw(plot_ma))


st.subheader('Features Engineering')

ret_ser = (
    active_returns
    .pivot(index="date", columns="symbol", values="active_ret")
    .dropna()
    .ICLN
)

st.write(ret_ser.head())

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
    

# Apply the feature engineering function to the return series to generate the feature set
X = (ret_ser
     .pipe(feature_engineer, ver="v0")
     # replace inf values with NaN and drop them
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
)

st.write(X.head())


# Train vs. Test Split
train_start, test_start = "2009-09-29", "2022-1-1"
X_train, X_test = X[:test_start], X[test_start:]

# print time split
train_start, train_end = X_train.index[[0, -1]]
test_start, test_end = X_test.index[[0, -1]]

st.write("Training starts at:", train_start, "and ends at:", train_end)
st.write("Testing starts at:", test_start, "and ends at:", test_end)

# Preprocessing
from jumpmodels.preprocess import StandardScalerPD, DataClipperStd

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

st.write(X_train_processed.head())

st.subheader('Jump Model')

from jumpmodels.jump import JumpModel                 # class of JM & CJM

# set the jump penalty
jump_penalty=50.
# initlalize the JM instance
jm = JumpModel(n_components=2, jump_penalty=jump_penalty, cont=False, )

# call .fit()
jm.fit(X_train_processed, ret_ser, sort_by="cumret")


st.write("Scaled Cluster Centroids:", pd.DataFrame(jm.centers_, index=["Bull", "Bear"], columns=X_train.columns))

from plotnine import *
import pandas as pd
from mizani.formatters import date_format
from mizani.breaks import date_breaks

# Calculate the 252-day rolling average of returns
ret_ser_rolling = ret_ser.rolling(window=126).mean().reset_index()
ret_ser_rolling.columns = ["date", "rolling_avg_return"]
# restrict the rolling average to the train period
ret_ser_rolling = ret_ser_rolling.query(f'date < "{test_start}"')

# Calculate ymin and ymax based on rolling average returns data
ymin = ret_ser_rolling["rolling_avg_return"].min()
ymax = ret_ser_rolling["rolling_avg_return"].max()
std = ret_ser_rolling["rolling_avg_return"].std()

# Prepare the regimes DataFrame to get the start and end dates of each regime period
regimes = (
    jm.labels_
    .reset_index(name="regime")
    .assign(
        date=lambda x: pd.to_datetime(x["date"]),
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
