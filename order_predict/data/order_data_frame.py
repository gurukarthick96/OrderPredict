import pandas as pd

from order_predict import config
from order_predict.data.extractors import extract_fields_from_date
from order_predict.data.outliers import correct_outliers
from order_predict.data.prerequisites import add_missing_date
from order_predict.db import OrderDomain


def aggregate_by_date(fields_to_extract: list[str], fields_to_train: list[str]) -> pd.DataFrame:
    """
    Aggregates order data by date, calculating total sum and count, and extracts additional date-related fields.

    Parameters
    ----------
    fields_to_extract : list[str]
        A list of datetime attributes to extract from the 'created_date' column.
        Example: ['day_of_week', 'month', 'day_of_month', 'week_of_year', 'holiday'].
    fields_to_train : list[str]
        A list of column names representing target variables for training.
        Example: ['total_sum', 'total_count'].

    Returns
    -------
    pd.DataFrame
        A DataFrame with aggregated order data, containing the following columns:
        - 'created_date': The date of aggregation.
        - 'total_sum': The sum of order totals for each date.
        - 'total_count': The count of orders for each date.
        - Additional extracted date-related fields from the `fields` list.
    """

    df = __get_orders_as_data_frame()

    df['created_at'] = pd.to_datetime(df['created_at'].dt.date)

    df = df.groupby('created_at').aggregate({'total': ['sum', 'count']}).reset_index()

    df.columns = ['created_date', 'total_sum', 'total_count']

    if config.ORDER_DF_ADD_MISSING_DATE:
        df = add_missing_date(df)

    df = correct_outliers(df, fields_to_train, config.ORDER_DF_OUTLIER_CORRECTION_STRATEGY)

    df = extract_fields_from_date(df, 'created_date', fields_to_extract)

    df = df[['created_date', *fields_to_extract, *fields_to_train]]

    return df


def __get_orders_as_data_frame() -> pd.DataFrame:
    """
    Get All Orders from DB and form a DataFrame out of it.

    Returns
    -------
    pd.DataFrame
        Data frame, returned columns=['created_at', 'total']
    """

    order_dict_list = OrderDomain.get_all_as_dict()

    df = pd.DataFrame(order_dict_list)

    df = df.drop(columns=['_id'])

    return df
