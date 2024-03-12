from datetime import timedelta

import pandas as pd
import ta
import yfinance as yf
import xgboost as xgb
from sklearn.metrics import mean_squared_error


#


def download_stocks(stock_codes, start_date, end_date):
    """"""
    # Fetch the stock data
    stock_data = {}
    for code in stock_codes:
        stock_data[code] = yf.download(code, start=start_date, end=end_date)

    return stock_data


def get_tech_features(data):
    # Add all ta features filling nans values
    return ta.add_all_ta_features(
        data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)


def get_lag_features(data, feature=None, lags=None):
    # Calculate lagged features

    if lags is None:
        lags = [1, 2, 3, 4, 5, 7, 14, 30, 100, 200]

    count = data[feature]

    lag_features = [count.shift(x).rename(f"{feature}_lag_{x}d") for x in lags]
    rolling_mean_features = [count.shift(1).rolling(x).mean().rename(f"{feature}_rolling_mean_{x}d") for x in lags]

    lagged_df = pd.concat([count, *lag_features, *rolling_mean_features], axis="columns")

    return lagged_df.drop(feature, axis=1, errors='ignore')


def get_target(data, target_feature=None, future_days=1):
    # Calculate lagged features

    target_col = data[target_feature]
    return target_col.shift(-future_days).rename(f"{target_feature}_next_{future_days}d")


def build_model(X, y):
    xgb_model = xgb.XGBRegressor(objective="reg:linear", random_state=42)
    xgb_model.fit(X, y)
    y_pred = pd.DataFrame(xgb_model.predict(X), index=X.index)
    print(f"mean Square error in train data: {mean_squared_error(y, y_pred)}", )

    return xgb_model


def build_n_forecast(stock_names):
    all_predictions = pd.DataFrame()
    for stock_name in stock_names:
        print("Building model for ", stock_name)
        features_path = f"features_{stock_name}.csv"
        target_path = f"target_{stock_name}.csv"
        test_real_path = f"features_{stock_name}_test.csv"

        X = pd.read_csv(features_path).set_index("Date")
        y = pd.read_csv(target_path).set_index("Date")
        test_real = pd.read_csv(test_real_path).set_index("Date")
        xgb_model = build_model(X, y)

        y_real = pd.DataFrame(xgb_model.predict(test_real), index=test_real.index, columns=["forecast"]).reset_index()
        y_real["Stock"] = stock_name
        y_real["Forecast_period"] = pd.to_datetime(y_real["Date"]) + timedelta(days=1)
        if stock_name == stock_names[0]:
            all_predictions = y_real
        else:
            all_predictions = pd.concat([all_predictions, y_real], axis='index')
    return all_predictions


def generate_all_features(stock_codes, start_date=None, end_date=None, test_days=1, get_test_data=False):
    y_test = pd.DataFrame()
    for stock_name in stock_codes:
        print("generating features for ", stock_name)
        stock_data = download_stocks([stock_name], start_date, end_date)
        data = get_tech_features(stock_data[stock_name])
        lag_features = get_lag_features(data, feature='Close').drop('Close', axis=1, errors='ignore').fillna(0)
        data_features = pd.concat([data, lag_features], axis='columns')
        y = get_target(data_features, target_feature='Close')
        ind_features = data_features.drop(data_features.tail(test_days).index)
        data_future = data_features[data_features.index.isin(data_features.tail(test_days).index)]
        if get_test_data:
            y_test = y.tail(test_days)
        y = y.drop(y.tail(test_days).index)

        features_path = f"features_{stock_name}.csv"
        target_path = f"target_{stock_name}.csv"
        future_data_path = f"features_{stock_name}_test.csv"
        test_data_path = f"target_{stock_name}_test.csv"
        ind_features.to_csv(features_path)
        y.to_csv(target_path)
        data_future.to_csv(future_data_path)
        if get_test_data:
            y_test.to_csv(test_data_path)

        print("file names: ", features_path, target_path, future_data_path)
