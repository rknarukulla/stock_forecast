"""test cases for stocks.py"""

from datetime import datetime

import pandas as pd

from src.stocks import (
    download_stocks,
    get_tech_features,
    get_target,
    build_model,
    build_xgboost_model,
    build_h2o_automl,
    build_n_forecast,
    generate_all_features,
)


stock_data = download_stocks(
    ["AAPL"], datetime(2021, 1, 1), datetime(2021, 1, 10)
)


def test_download_stocks() -> None:
    """Test download_stocks function.
    Returns:
        None: None
    """

    assert isinstance(stock_data["AAPL"], pd.DataFrame)


# def test_get_tech_features() -> None:
#     """Test get_tech_features function.
#     Returns:
#         None: None
#     """
#
#     data = get_tech_features(stock_data["AAPL"])
#     isinstance(data, pd.DataFrame)
