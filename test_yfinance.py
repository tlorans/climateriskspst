import numpy as np
import pandas as pd
import yfinance as yf
import os
from jumpmodels.utils import check_dir_exist

factors = {"value": "VLUE",
 "size": "SIZE",
 "momentum": "MTUM",
    "quality": "QUAL",
    "low_volatility": "USMV",
 "market":"PBUS",
 }


factors_world = {
    "value": "IVLU",
    "size": "IWSZ.L",
    "momentum": "IMTM",
    "quality": "IWQU.L",
}

pure_play = {
    "clean_energy":"ICLN",
}



TICKER = "IMTM"   # Nasdaq-100 Index

def get_data():
    # download closing prices
    close: pd.Series = yf.download(TICKER)['Close']
    # convert to returns
    ret = close.pct_change()
    # concat as df
    df = pd.DataFrame({"close": close, "ret": ret}, index=close.index.date)
    df.index.name = "date"

    # Set data directory and ensure it exists
    data_dir = "./data/"
    check_dir_exist(data_dir)  # Ensure the directory exists

    # Save data as pickle and CSV in the data directory
    pd.to_pickle(df, os.path.join(data_dir, f"{TICKER}.pkl"))
    np.round(df, 6).to_csv(os.path.join(data_dir, f"{TICKER}.csv"))
    print("Successfully downloaded data for ticker:", TICKER)

get_data()
