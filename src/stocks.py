import datetime
from datetime import timedelta

import pandas as pd
import ta
import xgboost as xgb
import yfinance as yf
from h2o import h2o

PARAMS: dict = {
    "max_depth": 3,
    "eta": 1,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "n_estimators": 100,
    "random_state": 42,
}


#


def download_stocks(
    stock_codes: list[str],
    start_date: datetime.datetime,
    end_date: datetime.datetime,
):
    """"""
    # Fetch the stock data
    stock_data = {}
    for code in stock_codes:
        stock_data[code] = yf.download(code, start=start_date, end=end_date)

    return stock_data


def get_tech_features(data):
    # Add all ta features filling nans values
    return ta.add_all_ta_features(
        data,
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        fillna=True,
    )


def get_lag_features(data, feature=None, lags=None):
    # Calculate lagged features

    if lags is None:
        lags = [1, 2, 3, 4, 5, 7, 14, 30, 100, 200]

    count = data[feature]

    lag_features = [count.shift(x).rename(f"{feature}_lag_{x}d") for x in lags]
    rolling_mean_features = [
        count.shift(1).rolling(x).mean().rename(f"{feature}_rolling_mean_{x}d")
        for x in lags
    ]

    lagged_df = pd.concat(
        [count, *lag_features, *rolling_mean_features], axis="columns"
    )

    return lagged_df.drop(feature, axis=1, errors="ignore")


def get_target(data, target_feature=None, future_days=1):
    # Calculate lagged features

    target_col = data[target_feature]
    return target_col.shift(-future_days).rename(
        f"{target_feature}_next_{future_days}d"
    )


def build_xgboost_model(X, y, validation_days):
    x_train = X.iloc[:-validation_days]
    y_train = y.iloc[:-validation_days]

    x_val = X.iloc[-validation_days:]
    y_val = y.iloc[-validation_days:]

    drain = xgb.DMatrix(x_train, label=y_train)
    deval = xgb.DMatrix(x_val, label=y_val)

    # Build model
    eval_list = [(drain, "train"), (deval, "test")]
    xgb_model = xgb.train(PARAMS, drain, num_boost_round=10, evals=eval_list)

    return xgb_model


def build_model(X, y, validation_days=7, model_name="xgboost"):
    if model_name == "xgboost":
        return build_xgboost_model(X, y, validation_days)
    elif model_name == "h2o_automl":
        return build_h2o_automl(X, y, validation_days)
    else:
        raise Exception("Invalid model_name")


def build_h2o_automl(X, y, validation_days=7, stock_name="stock_name"):
    import h2o
    from h2o.automl import H2OAutoML

    h2o.init()
    data = h2o.H2OFrame(pd.concat([X, y], axis="columns"))
    train = data.head(data.shape[0] - validation_days)
    test = data.tail(validation_days)

    x = list(X.columns)
    y_col = list(y.columns)[0]

    aml = H2OAutoML(
        max_models=5, seed=1, sort_metric="rmse", verbosity="error"
    )  # project_name=stock_name
    aml.train(x=x, y=y_col, training_frame=train, leaderboard_frame=test)

    return aml


def forecast_stocks(x, model, model_name="xgboost"):
    if model_name == "xgboost":
        return model.predict(xgb.DMatrix(x))
    elif model_name == "h2o_automl":
        x_real_h20 = h2o.H2OFrame(x)
        return h2o.as_list(model.leader.predict(x_real_h20))

    else:
        raise Exception("Invalid model")


def build_n_forecast(
    stock_names,
    date_column="Date",
    future_days=1,
    validation_days=7,
    model_name="xgboost",
):
    all_predictions = pd.DataFrame()
    for stock_name in stock_names:
        print("Building model for ", stock_name)
        features_path = f"features_{stock_name}.csv"
        target_path = f"target_{stock_name}.csv"
        test_real_path = f"features_{stock_name}_test.csv"

        X = pd.read_csv(features_path).set_index(date_column)
        y = pd.read_csv(target_path).set_index(date_column)

        test_real = pd.read_csv(test_real_path).set_index(date_column)
        model = build_model(X, y, validation_days, model_name=model_name)

        forecast = forecast_stocks(test_real, model, model_name=model_name)

        if not isinstance(forecast, pd.DataFrame):
            y_real = pd.DataFrame(
                forecast, index=test_real.index, columns=["forecast"]
            ).reset_index()
        else:
            y_real = forecast
            y_real.columns = ["forecast"]
            y_real[date_column] = test_real.index

        y_real["Stock"] = stock_name
        y_real["Forecast_period"] = pd.to_datetime(
            y_real[date_column]
        ) + timedelta(days=future_days)
        if stock_name == stock_names[0]:
            all_predictions = y_real
        else:
            all_predictions = pd.concat([all_predictions, y_real], axis="index")
    return all_predictions


def generate_all_features(
    stock_codes,
    start_date=None,
    end_date=None,
    test_days=1,
    get_test_data=False,
):
    y_test = pd.DataFrame()
    for stock_name in stock_codes:
        print("generating features for ", stock_name)
        stock_data = download_stocks([stock_name], start_date, end_date)
        data = get_tech_features(stock_data[stock_name])
        lag_features = (
            get_lag_features(data, feature="Close")
            .drop("Close", axis=1, errors="ignore")
            .fillna(0)
        )
        data_features = pd.concat([data, lag_features], axis="columns")
        y = get_target(data_features, target_feature="Close")
        ind_features = data_features.drop(data_features.tail(test_days).index)
        data_future = data_features[
            data_features.index.isin(data_features.tail(test_days).index)
        ]
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
