import pandas as pd
import pandas.core.arrays as pdarrays


def extract_fields_from_date(df: pd.DataFrame, date_field: str, fields: list[str]) -> pd.DataFrame:
    """
    Extracts specific datetime-related fields from a given date column in a Pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame to process. accepted columns=[date_field]
    date_field : str
        A date column from which datetime attributes are extracted
    fields : list[str]
        A list of datetime attributes to extract

    Returns
    -------
    pd.DataFrame
        Data frame, returned columns=[date_field, *fields]
    """

    date_dt = df[date_field].dt

    for field in fields:
        df[field] = __extract_field_from_pandas_dtarray(date_dt, field)

    return df


def __extract_field_from_pandas_dtarray(dt: pdarrays.DatetimeArray, field: str) -> pd.Series:
    """
    Parameters
    ----------
    dt : pdarrays.DatetimeArray
        Datetime Array to process
    field : str
        datetime attribute to extract

    Returns
    -------
    pd.Series
        Pandas Series of extracted field
    """

    match field:
        case 'day_of_week':
            extracted = dt.dayofweek  # Monday=0, Sunday=6
        case 'month':
            extracted = dt.month  # 1 to 12
        case 'day_of_month':
            extracted = dt.day  # 1 to 31
        case 'week_of_year':
            extracted = dt.isocalendar().week  # 1 to 52
        case 'holiday':
            extracted = dt.dayofweek.isin([5, 6])  # True for Sat/Sun
        case _:
            raise ValueError(f'unsupported field: {field}')

    return extracted
