import pandas as pd

from order_predict import config
from order_predict.data.extractors import extract_fields_from_date
from order_predict.data.outliers import correct_outliers
from order_predict.data.prerequisites import add_missing_date
from order_predict.db import OrderDomain


def aggregate_by_date() -> pd.DataFrame:
    """
    Returns
    -------
    pd.DataFrame
        Data frame, returned columns=['created_date', 'total_sum', 'total_count', *config.ORDER_DF_DATE_FIELDS_TO_EXTRACT]
    """

    df = __get_orders_as_data_frame()

    df['created_at'] = pd.to_datetime(df['created_at'].dt.date)

    df = df.groupby('created_at').aggregate({'total': ['sum', 'count']}).reset_index()

    df.columns = ['created_date', 'total_sum', 'total_count']

    if config.ORDER_DF_ADD_MISSING_DATE:
        df = add_missing_date(df)

    df = correct_outliers(df, config.ORDER_DF_OUTLIER_CORRECTION_STRATEGY)

    df = extract_fields_from_date(df, config.ORDER_DF_DATE_FIELDS_TO_EXTRACT)

    return df


def __get_orders_as_data_frame() -> pd.DataFrame:
    """
    Returns
    -------
    pd.DataFrame
        Data frame, returned columns=['created_at', 'total']
    """

    order_dict_list = OrderDomain.get_all_as_dict()

    df = pd.DataFrame(order_dict_list)

    df = df.drop(columns=['_id'])

    return df
