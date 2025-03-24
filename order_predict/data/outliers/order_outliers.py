from typing import Any

import pandas as pd

from order_predict.modules import logger, OutlierCorrectionStrategy


def correct_outliers(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """
    Corrects outliers in the given DataFrame based on the specified strategy (CAP or FILTER).

    Parameters
    ----------
    df : pd.DataFrame
        Data frame to process. accepted columns=['total_sum', 'total_count']
    strategy : str
        Outlier correction strategy

    Returns
    -------
    pd.DataFrame
        Data frame, returned columns=['total_sum', 'total_count']
    """

    logger.debug('before outlier corrections: \n%s', df)

    fields = ['total_sum', 'total_count']

    match OutlierCorrectionStrategy(strategy):
        case OutlierCorrectionStrategy.CAP:
            df = __cap_outlier_sum_and_count(df, fields)
        case OutlierCorrectionStrategy.FILTER:
            df = __filter_outlier_sum_and_count(df, fields)

    logger.debug('after outlier corrections: \n%s', df)

    return df


def __cap_outlier_sum_and_count(df: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    logger.info('capping outliers from dataframe...')

    original_values = df[fields]

    df[fields] = df[fields].apply(__clip_outliers)

    outlier_mask = (df[fields] != original_values).any(axis=1)

    logger.info('capped outliers: \n%s', df[outlier_mask])

    return df


def __clip_outliers(series: pd.Series) -> pd.Series:
    lower_bound, upper_bound = __find_lower_and_upper_bound(series)

    return series.clip(lower=lower_bound, upper=upper_bound)


def __filter_outlier_sum_and_count(df: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    logger.info('filtering outliers from dataframe...')

    inlier_mask = df[fields].apply(__find_outliers).all(axis=1)

    logger.info('filtered outliers: \n%s', df[~inlier_mask])

    return df[inlier_mask]


def __find_outliers(series: pd.Series) -> Any:
    lower_bound, upper_bound = __find_lower_and_upper_bound(series)

    return (series >= lower_bound) & (series <= upper_bound)


def __find_lower_and_upper_bound(series: pd.Series) -> tuple[float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    # Define upper and lower bounds for outliers using IQR method
    lower_bound = max(q1 - 1.5 * iqr, 0)
    upper_bound = q3 + 1.5 * iqr

    logger.info('for %s column => lower_bound: %f and upper_bound: %f', series.name, lower_bound, upper_bound)

    return lower_bound, upper_bound
