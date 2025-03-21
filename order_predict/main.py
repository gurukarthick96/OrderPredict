from order_predict.data import order_df
from order_predict.modules import logger

logger.info('starting...')

df = order_df.aggregate_by_date()

logger.info('head: \n%s', df)
