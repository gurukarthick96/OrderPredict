from order_predict import config
from order_predict.data import order_df
from order_predict.db import ensure_indexes
from order_predict.modules import logger
from order_predict.predict import order_predict
from order_predict.train import order_train
from order_predict.train.regression import RegressionType


def get_active_fields(fields: dict[str, int]) -> list[str]:
    return [key for key, value in fields.items() if value]


def train_and_predict_order():
    logger.info('starting...')

    fields_to_extract = get_active_fields(config.ORDER_DF_DATE_FIELDS_TO_EXTRACT)
    fields_to_train = get_active_fields(config.ORDER_DF_FIELDS_TO_TRAIN)
    fields_to_predict = get_active_fields(config.ORDER_DF_FIELDS_TO_PREDICT)

    reg_type = RegressionType.lookup_by_name(config.ORDER_TRAIN_REGRESSION_TYPE)

    df = order_df.aggregate_by_date(fields_to_extract, fields_to_train)

    model = order_train.train_regression_model(df, fields_to_extract, fields_to_train, reg_type)

    future_df = order_predict.predict_sales_and_count(model, fields_to_extract, fields_to_predict,
                                                      ('2025-03-11', '2025-04-11'))

    logger.info('Original Data Frame: \n%s', df)

    logger.info('Future Predictions: \n%s', future_df)


if __name__ == '__main__':
    ensure_indexes()
    train_and_predict_order()
