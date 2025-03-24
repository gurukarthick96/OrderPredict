import pandas as pd

from order_predict.data.extractors import extract_fields_from_date


def predict_sales_and_count(model, fields: list[str], date_range: tuple[str, str]) -> pd.DataFrame:
    """
    Predicts total sales and order count for a given date range using a trained model.

    Parameters
    ----------
    model :
        A trained predictive model with a `.predict()` method that accepts a DataFrame.
    fields : list[str]
        A list of column names to use as features (X).
    date_range : tuple[str, str]
        A tuple containing the start and end dates as strings in the format 'YYYY-MM-DD'.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame with the following columns:
        - `future_date`: The future date.
        - `day_of_week`, `month`, `day_of_month`, `week_of_year`, `holiday`: Extracted date features.
        - `predicted_sum`: Predicted total sales for the date.
        - `predicted_count`: Predicted number of orders for the date.
    """

    future_dates = pd.date_range(start=date_range[0], end=date_range[1])

    future_df = pd.DataFrame({'future_date': future_dates})
    future_df = extract_fields_from_date(future_df, 'future_date', fields)

    predictions = model.predict(future_df[fields])
    future_df[['predicted_sum', 'predicted_count']] = pd.DataFrame(
        predictions, columns=['predicted_sum', 'predicted_count']
    ).clip(lower=0).round().astype(int)

    future_df = future_df[['future_date', *fields, 'predicted_sum', 'predicted_count']]

    return future_df
