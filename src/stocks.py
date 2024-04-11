"""utilities for stock analysis and forecasting"""

import datetime as dt
from datetime import timedelta
from typing import Any
import h2o
import pandas as pd
import ta
import xgboost as xgb
import yfinance as yf
from h2o.automl import H2OAutoML
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from icecream import ic

PARAMS: dict[str, Any] = {
    "max_depth": 3,
    "eta": 1,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "n_estimators": 100,
    "random_state": 42,
}


def download_stocks(
    stock_codes: list[str],
    start_date: dt.datetime,
    end_date: dt.datetime,
) -> dict[str, pd.DataFrame]:
    """Download stock data from yahoo finance.
    Args:
        stock_codes (list[str]): list of stock codes to download.
        start_date (datetime.datetime): Start date of the data.
        end_date (datetime.datetime): End date of the data.

    Returns:
        pd.DataFrame: Dataframe containing the stock data.

    """
    # Fetch the stock data
    stock_data: dict[str, pd.DataFrame] = {}
    for code in stock_codes:
        ic("downloading stock data for", code)
        stock_data[code] = yf.download(code, start=start_date, end=end_date)
        file_path = f"{code}.csv"
        stock_data[code].to_csv(file_path)
    return stock_data


def get_tech_features(data: pd.DataFrame) -> pd.DataFrame:
    """Get technical indicators for the stock data.
    Args:
        data (pd.DataFrame): Dataframe containing the stock data.

    Returns:
        pd.DataFrame: Dataframe containing the technical indicators.
    """
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


def get_lag_features(
    data: pd.DataFrame, feature: str = None, lags: list[int] = None
) -> pd.DataFrame:
    """Get lagged features for the stock data.
    Args:
        data (pd.DataFrame): Dataframe containing the stock data.
        feature(str): Feature to lag.
        lags(list[int]): Lags to calculate.

    Returns:
        pd.DataFrame: Dataframe containing the lagged features.

    """
    # Calculate lagged features

    if lags is None:
        lags: list[int] = [1, 2, 3, 4, 5, 7, 14, 30, 100, 200]

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


def get_target(
    data: pd.DataFrame, target_feature: str = None, future_days: int = 1
) -> pd.Series:
    """Generate target feature for the stock forcast.
    Args:
        data (pd.DataFrame): Dataframe containing the stock data.
        target_feature (str): Target feature to forecast.
        future_days(int): Number of days to forecast.

    Returns:
        pd.Series: Pandas series containing the target feature.

    """
    target_col: pd.Series = data[target_feature]

    return target_col.shift(-future_days).rename(
        f"{target_feature}_next_{future_days}d"
    )


def build_xgboost_model(
    X: pd.DataFrame, y: pd.Series, validation_days: int = 7
) -> xgb.core.Booster:
    """Build xgboost model.
    Args:
        X(pd.DataFrame): Dataframe containing the features.
        y(pd.Series | pd.DataFrame): Target feature.
        validation_days(int): Number of days to use for validation.

    Returns:
        xgb.core.Booster: Xgboost model.
    """
    x_train: pd.DataFrame = X.iloc[:-validation_days]
    y_train: pd.Series | pd.DataFrame = y.iloc[:-validation_days]

    x_val: pd.DataFrame = X.iloc[-validation_days:]
    y_val: pd.DataFrame | pd.Series = y.iloc[-validation_days:]

    drain: xgb.DMatrix = xgb.DMatrix(x_train, label=y_train)
    deval: xgb.DMatrix = xgb.DMatrix(x_val, label=y_val)

    # Build model
    eval_list: list[[xgb.DMatrix, str]] = [(drain, "train"), (deval, "test")]
    xgb_model: xgb.core.Booster = xgb.train(
        PARAMS, drain, num_boost_round=10, evals=eval_list
    )

    return xgb_model


def build_model(
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    validation_days: int = 7,
    model_name: str = "xgboost",
) -> Any:
    """Build model based on the model_name.
    Args:
        X (pd.DaDataFrame): dataframe containing the features.
        y (pd.Series | pd.DataFrame): target feature.
        validation_days (int): number of days to use for validation.
        model_name (str): model name to build relevant model.

    Returns:
        Any: model object.

    """
    if model_name == "xgboost":
        return build_xgboost_model(X, y, validation_days)
    elif model_name == "h2o_automl":
        return build_h2o_automl(X, y, validation_days)
    else:
        raise Exception("Invalid model_name")


def build_h2o_automl(
    X: pd.DataFrame,
    y: pd.Series,
    validation_days: int = 7,
    stock_name: str = None,
) -> Any:
    """Build h2o automl model

    Args:
        X(pd.DataFrame): Dataframe containing the features.
        y(pd.Series): Target feature.
        validation_days(int): Number of days to use for validation.
        stock_name(str): stock name as the model name to identify model.

    Returns:
        Any: H2O automl model object.
    """

    h2o.init()
    data = h2o.H2OFrame(pd.concat([X, y], axis="columns"))
    train = data.head(data.shape[0] - validation_days)
    test = data.tail(validation_days)

    x = list(X.columns)
    y_col = list(y.columns)[0]

    aml = H2OAutoML(
        max_models=5,
        seed=1,
        sort_metric="rmse",
        verbosity="error",
        project_name=stock_name,
    )
    aml.train(x=x, y=y_col, training_frame=train, leaderboard_frame=test)

    return aml


def forecast_stocks(
    features: pd.DataFrame, model: Any, model_name: str = "xgboost"
) -> pd.DataFrame:
    """Forecast stocks using the model
    Args:
        features (pd.DataFrame): Dataframe containing the features.
        model (Any): Model object.
        model_name (str): Model name.

    Returns:
        pd.DataFrame: Dataframe containing the predictions.

    Raises:
        ValueError: If model_name is invalid.
    """

    if model_name == "xgboost":
        return model.predict(xgb.DMatrix(features))
    elif model_name == "h2o_automl":
        x_real_h20 = h2o.H2OFrame(features)
        return h2o.as_list(model.leader.predict(x_real_h20))
    else:
        raise ValueError("Invalid model")


def build_n_forecast(
    stock_names: list[str],
    date_column: str = "Date",
    future_days: int = 1,
    validation_days: int = 7,
    model_name: str = "xgboost",
) -> pd.DataFrame:
    """Build a timeseries model for the stocks and forecast the stocks for the next future_days.

    Args:
        stock_names (list[str]): List of stock names to build the model.
        date_column (str): Date column name.
        future_days (int): Number of days to forecast.
        validation_days (int): Number of days to use for validation.
        model_name (str): Model name.
    Returns:
        pd.DataFrame: Dataframe containing the forecasted values.

    """
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


def get_day_names(
    data: pd.DataFrame, date_column: str = "Date"
) -> pd.DataFrame:
    """get day names as one hot encoded

    Args:
        data (pd.DataFrame): Dataframe containing the features.
        date_column: Date column name.

    Returns:
        pd.DataFrame: Dataframe containing the day names as one hot encoded.

    """
    if is_col_indexed(data, date_column):
        df = data.reset_index()[[date_column]]
    else:
        df = data[[date_column]]
    df["day"] = df[date_column].dt.day_name()
    dummies = pd.get_dummies(df["day"])
    if is_col_indexed(data, date_column):
        dummies.index = data.index
    return dummies


def optimized_exponential_smoothing(
    data: pd.DataFrame,
    date_column: str = "Date",
    target_column: str = "Close",
    periods: int = 1,
    seasonal_periods: int = 12,
    trend: str = "add",
    seasonal: str = "add",
    use_boxcox: bool = True,
) -> tuple[pd.Series, pd.Series]:

    if is_col_indexed(data, date_column):
        ts = data[target_column]
    else:
        ts = data.set_index(date_column)[target_column]

    model = ExponentialSmoothing(
        ts,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        damped_trend=True,
        use_boxcox=use_boxcox,
    )
    fitted_model = model.fit(optimized=True)
    fitted_values = fitted_model.fittedvalues

    # Forecast future values
    forecast = fitted_model.forecast(periods)

    return fitted_values, forecast


def is_col_indexed(data: pd.DataFrame, column_name: str) -> bool:
    """check if a columns is set"""
    return data.index.name == column_name


def get_date_diff(
    data: pd.DataFrame, date_column: str = "Date"
) -> pd.DataFrame:
    """get date difference between the current date and the previous date

    Args:
        data (pd.DateFrame): input data
        date_column (str): date column name.

    Returns:
        pd.DataFrame: Dataframe containing the date difference.
    """
    if is_col_indexed(data, date_column):
        ts = data.reset_index()[[date_column]]
    else:
        ts = data[[date_column]]
    ts[date_column] = pd.to_datetime(ts[date_column])
    ts["diff_previous_days"] = ts[date_column].diff().dt.days
    ts["diff_previous_days"] = ts["diff_previous_days"].fillna(0)
    if is_col_indexed(data, date_column):
        ts = ts.set_index(date_column)
    return ts["diff_previous_days"]


def generate_all_features(
    stock_codes: list[str],
    start_date: dt.datetime,
    end_date: dt.datetime,
    test_days: int = 1,
    get_test_data: bool = False,
) -> None:
    """Generate all features for the stock forcasting and save it to a csv file.

    Args:
        stock_codes (list[str]): List of stock codes.
        start_date (dt.datetime): Start date. Defaults to None.
        end_date (dt.datetime): End date. Defaults to None.
        test_days (int): Number of days to use for testing. Defaults to 1.
        get_test_data (bool, optional): If True, returns the test data. Defaults to False.
    Returns:
        None
    Raises:
        ValueError: If start_date or end_date is not provided.

    """

    if any([start_date, end_date]) is None:
        raise ValueError("Please provide start and end date")

    y_test: pd.DataFrame = pd.DataFrame()
    for stock_name in stock_codes:
        ic("generating features for ", stock_name)
        stock_data = download_stocks([stock_name], start_date, end_date)

        data = get_tech_features(stock_data[stock_name])
        lag_features = (
            get_lag_features(data, feature="Close")
            .drop("Close", axis=1, errors="ignore")
            .fillna(0)
        )

        exp_fitted, exp_forecast = optimized_exponential_smoothing(
            data=data,
            date_column="Date",
            target_column="Close",
            periods=test_days,
            seasonal_periods=12,
        )

        data["exp_fitted"] = (
            pd.concat([exp_fitted, exp_forecast])
            .shift(-test_days)
            .head(-test_days)
        )
        data["diff_previous_days"] = get_date_diff(data, date_column="Date")
        day_features = get_day_names(data, date_column="Date")
        # combine all features
        data_features = pd.concat(
            [data, lag_features, day_features], axis="columns"
        )
        y = get_target(data_features, target_feature="Close")
        ind_features = data_features.drop(data_features.tail(test_days).index)
        data_future = data_features[
            data_features.index.isin(data_features.tail(test_days).index)
        ]
        if get_test_data:
            y_test = pd.DataFrame(y.tail(test_days))
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

        ic("file names: ", features_path, target_path, future_data_path)
