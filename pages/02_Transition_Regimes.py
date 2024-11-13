import streamlit as st
from plotnine import *
from mizani.formatters import date_format
from mizani.breaks import date_breaks
from mizani.formatters import percent_format
import numpy as np
import pandas as pd
import yfinance as yf

st.title('Transition Regimes')

st.write(r'''
While the literature is inconclusive about 
         the transition risks pricing,
         theoretical and empirical evidences from Pastor et al. (2021, 2022)
         suggest that there is period of green (brown) stocks outperformance (underperformance), 
         conditional on unexpected changes in climate concerns. 
         While individual unexpected changes in climate concerns are by definition unpredictable,
         analysing the TRI pattern in the previous section have shown trends in 
         the transition concerns.

In this section, we explore the opportunities presented by 
this cyclicality through the lens of regime analysis. 
Regime switching has gained popularity primarly due to its interpretability, as identified
regimes can be associated with real-world events. In finance, 
a regime may be defined by relative homogeneous market behavior, while a regime shift 
signals an abrupt change in market dynamics.
This framework aligns well with the green factor cyclicality, with expected outperformance 
in periods of increasing climate concerns and underperformance in periods of decreasing climate concerns.

While unsupervised learning methods such as the k-means clustering algorithm has been used to identify market regimes (DiCiurcio et al., 2024),
Jump Models (thereafter JM) are particularly well-suited to time series data 
         due to their incorporation of temporal information through an explicit jump penalty for each transition.
         This approach acknowledges the persistence in financial regimes and enhances the interpretability of 
         the identified regimes, as we do not expect frequent regime shifts.

In a similar vein as the work of Shu and Mulvey (2024) with traditional factors,
we will explore the regime analysis of the green factor, using the ICLN ETF as a proxy,
         and a Jump Model to identify the bull and bear markets of the green factor. 
         ''')

st.subheader('Green Stocks Bull and Bear Markets')

st.write(r'''
A first stage in our analysis 
         is to see if the green factor exhibits regime-like behavior 
         in terms of its returns.
            We will use the ICLN ETF active returns as a proxy for the green factor.
         ''')

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

# 126 day moving average

# Compute the 126-day moving average of the active returns
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

# 126 day moving average

# Compute the 126-day moving average of the active returns
returns_daily_ma = (
    active_returns
    .assign(
        active_ret_ma=lambda x: x.groupby("symbol")["active_ret"].transform(
            lambda x: x.rolling(window=126).mean()
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
    + geom_hline(yintercept=0, linetype="dashed") +
    scale_x_datetime(breaks=date_breaks("1 year"), labels=date_format("%Y"))
    + labs(
        x="",
        y="126-day Moving Average",
        title=""
    )
    + theme(axis_text_x=element_text(angle=45, hjust=1))
)

st.pyplot(ggplot.draw(plot_ma))

st.write(r'''
The code above proceeds similarly to the previous section,
            downloading the daily prices of the ICLN ETF and calculating the active returns.
         We then compute the 126-day moving average of the active returns,
            which is plotted above. The moving average is used to smooth out the active returns
            and highlight the medium-term trend in the green factor.
         We observe upward and downward trends in the moving average. This 
         is our target for regime analysis: to identify the periods of outperformance (bull markets)
            and underperformance (bear markets) of the green factor.
         ''')

st.subheader('Features Engineering')

st.write(r'''
To perform regime analysis, we are going to use the 
         `jumpmodels` package (Shu and Mulvey, 2024), which provides a
            framework for regime analysis based on jump models.

To apply the jump model, we need to engineer a set of features based 
on the active returns. We first create 
a `pd.Series` of the active returns of the ICLN ETF, which will be used as the input to the feature engineering function.
         ''')

st.code(r'''
ret_ser = (
    active_returns
    .pivot(index="date", columns="symbol", values="active_ret")
    .dropna()
    .ICLN
)
         ''')
ret_ser = (
    active_returns
    .pivot(index="date", columns="symbol", values="active_ret")
    .dropna()
    .ICLN
)

st.write(r'''
We use the features provided in the example application of the `jumpmodels` package.
The authors provide a feature engineering function that generates a set of features based on the input return series.
The function generates three sets of features:
1. The exponentially weighted moving average (EWM) of the return series with halflives of 5, 20, and 60 days.
2. The logarithm of the exponentially weighted moving downside deviation (DD) of the return series with the same halflives.
3. The EWM Sortino ratio, which is the ratio of the EWM return to the EWM DD.
         ''')

st.code(r'''
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
        ''')
        


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
    

st.code(r'''
X = (ret_ser
     .pipe(feature_engineer, ver="v0")
     # replace inf values with NaN and drop them
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
)
        ''')
# Apply the feature engineering function to the return series to generate the feature set
X = (ret_ser
     .pipe(feature_engineer, ver="v0")
     # replace inf values with NaN and drop them
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
)

st.write(X.head())

st.write(r'''
         The code above applies the feature engineering function to the active return series of the ICLN ETF.
The function returns a DataFrame of engineered features, which will be used as the input to the Jump Model.
         ''')         

st.code(r'''
# Train vs. Test Split
train_start, test_start = "2009-09-29", "2022-1-1"
X_train, X_test = X[:test_start], X[test_start:]

# print time split
train_start, train_end = X_train.index[[0, -1]]
test_start, test_end = X_test.index[[0, -1]]
        ''')

# Train vs. Test Split
train_start, test_start = "2009-09-29", "2022-1-1"
X_train, X_test = X[:test_start], X[test_start:]

# print time split
train_start, train_end = X_train.index[[0, -1]]
test_start, test_end = X_test.index[[0, -1]]

st.write(fr'''
The code above splits the engineered features into training and testing sets.
The training set spans from the beginning of the data to the start of the testing period,
while the testing set spans from the start of the testing period to the end of the data.
Training starts at: {train_start} and ends at: {train_end}  ''')


st.write(r'''
Before feeding the data into the Jump Model, we need to preprocess the data.
The `jumpmodels` package provides a set of preprocessing functions to standardize and clip the data.
We use the `StandardScalerPD` and `DataClipperStd` classes to standardize and clip the data at 3 standard deviations.
         ''')

st.code(r'''
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
        ''')
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

st.write(r'''
The code above applies the preprocessing steps to the training and testing sets.   
         ''')

st.subheader('Jump Model')

st.write(r'''
         We can now initialize the Jump Model and fit it to the training data.
            We set the number of components to 2, corresponding to the bull and bear markets of the green factor.
            We also set the jump penalty to 30, which controls the penalty for transitioning between regimes. 
            A higher jump penalty results in fewer transitions between regimes.
            We set `cont=False` to enforce discrete regime transitions.
            Finally, we call the `.fit()` method to fit the Jump Model to the training data.
            ''')

st.code(r'''
from jumpmodels.jump import JumpModel                 # class of JM & CJM

# set the jump penalty
jump_penalty=30.
# initlalize the JM instance
jm = JumpModel(n_components=2, jump_penalty=jump_penalty, cont=False, )

# call .fit()
jm.fit(X_train_processed, ret_ser, sort_by="cumret")
        ''')

from jumpmodels.jump import JumpModel                 # class of JM & CJM

# sidebar to choose the jump penalty, slider from 0 to 100, move 
# from 10 to 10
jump_penalty = st.sidebar.slider("Jump Penalty", 0., 100., 30., 10.)

# initlalize the JM instance
jm = JumpModel(n_components=2, jump_penalty=jump_penalty, cont=False, )

# call .fit()
jm.fit(X_train_processed, ret_ser, sort_by="cumret")


st.write(r'''
While scaled, the cluster centroids provide insights into the characteristics of the bull and bear markets.
The centroids represent the average feature values of each regime.
We can interpret the centroids as the "typical" feature values of the bull and bear markets.
Because the values are scaled, you should focus on the relative differences between the bull and bear markets 
rather than the absolute values.
         ''')

st.code(r'''
scaled_centroids = pd.DataFrame(jm.centers_, index=["Bull", "Bear"], columns=X_train.columns)
''')

scaled_centroids = pd.DataFrame(jm.centers_, index=["Bull", "Bear"], columns=X_train.columns)

scaled_centroids

st.write(r'''The code above extracts the scaled centroids of the bull and bear markets from the Jump Model.
The table displays the average feature values of the bull and bear markets.
            ''')


st.write(r'''
         The Jump Model assigns each data point to a regime. 
         We can extract the regime labels from the Jump Model and visualize the regime transitions.
            The regime labels are stored in the `labels_` attribute of the Jump Model.
            ''')

st.code(r'''
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

ggplot.draw(p)
        ''')

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

st.write(r'''
         The code above plots the regime transitions of the green factor,
            with the rolling average of the active returns overlaid.
            The shaded areas represent the bull and bear markets, with green indicating the bull market and red indicating the bear market.
            The rolling average is used to highlight the medium-term trend in the green factor.
            We observe that the Jump Model successfully identifies the bull and bear markets of the green factor.
            ''')

st.subheader('Online Inference')


st.write(r'''
         We now let the Jump Model infer the regimes of the testing data.
         We use the `.predict()` method of the Jump Model to infer the regimes of the testing data.
In the prediction, we use the processed testing data `X_test_processed` as input,
and the Jump Model assigns each data point to a regime, based 
         on the previously learned regime characteristics.
         ''')

st.code(r'''
labels_test_online = jm.predict(X_test_processed)

# plot the online inferred data with rolling average returns of the ETF for the same dates

# Calculate the 252-day rolling average of returns
ret_ser_rolling_test = ret_ser.rolling(window=126).mean().reset_index()
ret_ser_rolling_test.columns = ["date", "rolling_avg_return"]
# restrict the rolling average to the train period
ret_ser_rolling_test = ret_ser_rolling_test.query(f'date >= "{test_start}"')

# Calculate ymin and ymax based on rolling average returns data
ymin = ret_ser_rolling_test["rolling_avg_return"].min()
ymax = ret_ser_rolling_test["rolling_avg_return"].max()
std = ret_ser_rolling_test["rolling_avg_return"].std()

# Prepare the regimes DataFrame to get the start and end dates of each regime period
regimes_test = (
    labels_test_online
    .reset_index(name="regime")
    .assign(
        date=lambda x: pd.to_datetime(x["date"]),
        label=lambda x: x["regime"].map({0: "Bull", 1: "Bear"})
    )
)

# Define start and end dates for each regime type
regime_highlights_test = (
    regimes_test.groupby((regimes_test['label'] != regimes_test['label'].shift()).cumsum())
    .agg(start_date=('date', 'first'), end_date=('date', 'last'), label=('label', 'first'))
    .assign(ymin=ymin - std, ymax=ymax + std)  # Set ymin and ymax dynamically
)

# Plot regimes with rolling average line
p_test = (
    ggplot() +
    # Regime shaded areas using geom_rect with dynamic ymin and ymax
    geom_rect(regime_highlights_test, aes(
        xmin='start_date', xmax='end_date', ymin='ymin', ymax='ymax', fill='label'
    ), alpha=0.3) +
    # Rolling average line plot
    geom_line(ret_ser_rolling_test, aes(x='date', y='rolling_avg_return')) +
    geom_hline(yintercept=0, linetype="dashed") + 
    labs(y="Rolling Avg Return", x="") +
    scale_fill_manual(values={"Bull": "green", "Bear": "red"}) +  # Green for Bull, Red for Bear
    scale_x_datetime(breaks=date_breaks("1 year"), labels=date_format("%Y")) +
    theme(
        axis_text_x=element_text(angle=45, hjust=1),
        legend_position="none"  # Hide legend if it distracts from the main plot
    )
)

ggplot.draw(p_test)
        ''')
labels_test_online = jm.predict(X_test_processed)

# plot the online inferred data with rolling average returns of the ETF for the same dates

# Calculate the 252-day rolling average of returns
ret_ser_rolling_test = ret_ser.rolling(window=126).mean().reset_index()
ret_ser_rolling_test.columns = ["date", "rolling_avg_return"]
# restrict the rolling average to the train period
ret_ser_rolling_test = ret_ser_rolling_test.query(f'date >= "{test_start}"')

# Calculate ymin and ymax based on rolling average returns data
ymin = ret_ser_rolling_test["rolling_avg_return"].min()
ymax = ret_ser_rolling_test["rolling_avg_return"].max()
std = ret_ser_rolling_test["rolling_avg_return"].std()

# Prepare the regimes DataFrame to get the start and end dates of each regime period
regimes_test = (
    labels_test_online
    .reset_index(name="regime")
    .assign(
        date=lambda x: pd.to_datetime(x["date"]),
        label=lambda x: x["regime"].map({0: "Bull", 1: "Bear"})
    )
)

# Define start and end dates for each regime type
regime_highlights_test = (
    regimes_test.groupby((regimes_test['label'] != regimes_test['label'].shift()).cumsum())
    .agg(start_date=('date', 'first'), end_date=('date', 'last'), label=('label', 'first'))
    .assign(ymin=ymin - std, ymax=ymax + std)  # Set ymin and ymax dynamically
)

# Plot regimes with rolling average line
p_test = (
    ggplot() +
    # Regime shaded areas using geom_rect with dynamic ymin and ymax
    geom_rect(regime_highlights_test, aes(
        xmin='start_date', xmax='end_date', ymin='ymin', ymax='ymax', fill='label'
    ), alpha=0.3) +
    # Rolling average line plot
    geom_line(ret_ser_rolling_test, aes(x='date', y='rolling_avg_return')) +
    geom_hline(yintercept=0, linetype="dashed") + 
    labs(y="Rolling Avg Return", x="") +
    scale_fill_manual(values={"Bull": "green", "Bear": "red"}) +  # Green for Bull, Red for Bear
    scale_x_datetime(breaks=date_breaks("1 year"), labels=date_format("%Y")) +
    theme(
        axis_text_x=element_text(angle=45, hjust=1),
        legend_position="none"  # Hide legend if it distracts from the main plot
    )
)

st.pyplot(ggplot.draw(p_test))

st.write(r'''
         The code above plots the regime transitions of the green factor in the testing data,
         We observe that the Jump Model successfully identifies the persistent bull and bear markets of the green factor in the testing data.
            ''')

st.subheader('Conclusion')

st.write(r'''
In this section, we explored the regime analysis of the green factor using the Jump Model.
We engineered a set of features based on the active returns of the ICLN ETF and trained the Jump Model on a subset of the data.
The Jump Model successfully identified the bull and bear markets of the green factor.
We visualized the regime transitions of the green factor in the training and testing data, highlighting the persistent bull and bear markets.
        
Therefore, while individual unexpected changes in climate concerns are unpredictable,
we may still be able to identify trends in the transition concerns and exploit the cyclicality of the green factor,
            as evidenced by the Jump Model's successful identification of the bull and bear markets.
          ''')

st.subheader('Exercices')

st.write(r'''
1. Experiment with different feature engineering functions in the `feature_engineer` function.
2. Try different value for the jump penalty in the Jump Model. How does the penalty affect the regime transitions?
3. Explore the regime transitions of other ETFs using the Jump Model.
         ''')