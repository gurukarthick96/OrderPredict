import pandas as pd


def add_missing_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Data frame to process. accepted columns=['created_date']

    Returns
    -------
    pd.DataFrame
        Data frame, returned columns=['created_date']
    """

    start_date, end_date = df['created_date'].aggregate(['min', 'max'])

    complete_dates = pd.DataFrame({'created_date': pd.date_range(start=start_date, end=end_date, freq='D')})

    return complete_dates.merge(df, on='created_date', how='left').fillna(0)
