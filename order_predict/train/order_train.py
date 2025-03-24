import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from order_predict import config
from order_predict.modules import logger
from order_predict.train.regression import RegressionType, build_regression_model


def train_regression_model(df: pd.DataFrame, fields: list[str], reg_type: RegressionType = None):
    """
    Trains a regression model using the provided DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing training data.
    fields : list[str]
        A list of column names to use as features (X).
    reg_type : RegressionType, optional
        The type of regression model to use. If None, the best model is selected
        based on evaluation. Default is None.

    Returns
    -------
    model
        A trained regression model instance.
    """

    x = df[fields]
    y = df[['total_sum', 'total_count']]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=config.ORDER_TRAIN_TEST_SPLIT_SIZE, random_state=42
    )

    if config.ORDER_TRAIN_APPLY_SCALER:
        x_train, x_test = __apply_scaler(x_train, x_test)

    if reg_type:
        rmse = __compute_rmse_for_regression_type(reg_type, x_train, x_test, y_train, y_test)
    else:
        reg_type, rmse = __find_best_regression_type_with_rmse(x_train, x_test, y_train, y_test)

    logger.info('chosen regression model: %s and rmse: %.2f', reg_type.value, rmse)

    model = build_regression_model(reg_type)

    model.fit(x_train, y_train)

    return model


def __apply_scaler(x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies standard scaling to training and test datasets using `StandardScaler`.
    """

    ss = StandardScaler()
    return ss.fit_transform(x_train), ss.transform(x_test)


def __find_best_regression_type_with_rmse(x_train: pd.DataFrame, x_test: pd.DataFrame,
                                          y_train: pd.DataFrame, y_test: pd.DataFrame) -> tuple[RegressionType, float]:
    """
    Finds the best regression type with least RMSE.
    """

    results = []

    for reg_type in RegressionType:
        rmse = __compute_rmse_for_regression_type(reg_type, x_train, x_test, y_train, y_test)

        results.append((reg_type, rmse))

    results_df = pd.DataFrame([(reg_type.value, rmse) for reg_type, rmse in results], columns=['Model', 'RMSE'])

    logger.info('every regression model performance: \n%s', results_df)

    best_reg_type, best_rmse = min(results, key=lambda item: item[1])

    return best_reg_type, best_rmse


def __compute_rmse_for_regression_type(reg_type,
                                       x_train: pd.DataFrame, x_test: pd.DataFrame,
                                       y_train: pd.DataFrame, y_test: pd.DataFrame) -> float:
    """
    Fits a regression model and computes RMSE.
    """

    model = build_regression_model(reg_type)

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    # r2 = r2_score(y_test, y_pred)

    return round(rmse, 2)
