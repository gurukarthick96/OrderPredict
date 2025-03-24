from order_predict import config
from order_predict.data import order_df
from order_predict.modules import logger
from order_predict.predict import order_predict
from order_predict.train import order_train
from order_predict.train.regression import RegressionType


def train_and_predict_order():
    logger.info('starting...')

    fields = config.ORDER_DF_DATE_FIELDS_TO_EXTRACT

    reg_type = RegressionType.lookup_by_name(config.ORDER_TRAIN_REGRESSION_TYPE)

    df = order_df.aggregate_by_date(fields)

    model = order_train.train_regression_model(df, fields, reg_type)

    future_df = order_predict.predict_sales_and_count(model, fields, ('2025-03-11', '2025-04-11'))

    logger.info('Original Data Frame: \n%s', df)

    logger.info('Future Predictions: \n%s', future_df)


if __name__ == '__main__':
    train_and_predict_order()
