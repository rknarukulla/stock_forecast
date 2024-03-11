import pandas as pd
import ta
import yfinance as yf


#


def download_stocks(stock_codes, start_date, end_date):
    # Fetch the stock data
    stock_data = {}
    for code in stock_codes:
        stock_data[code] = yf.download(code, start=start_date, end=end_date)

    return stock_data


def get_tech_features(stock_data_in):
    for code, data in stock_data_in.items():
        if not data.empty:
            # Example: Calculate Moving Average Convergence Divergence (MACD)
            data['MACD'] = ta.trend.MACD(data['Close']).macd()

            # Example: Calculate Relative Strength Index (RSI)
            data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
            print(f"Technical indicators for {code} on {start_date}:")
            print(data[['MACD', 'RSI']])
        else:
            print(f"No data for {code} on {start_date}.")


def get_lag_features(data, feature=None):
    # Calculate lagged features

    lags = [1, 2, 3, 4, 5, 7, 14, 30, 100, 200]

    count = data[feature]

    lag_features = [count.shift(x).rename(f"{feature}_lag_{x}d") for x in lags]
    rolling_mean_features = [count.shift(1).rolling(x).mean().rename(f"{feature}_rolling_mean_{x}d") for x in lags]

    lagged_df = pd.concat([count, *lag_features, *rolling_mean_features], axis="columns")

    return lagged_df


def get_target(data, target_feature=None, future_days=1):
    # Calculate lagged features

    target_col = data[target_feature]

    return pd.DataFrame(target_col.shift(-future_days).rename(f"{target_feature}_next_{future_days}d"))
