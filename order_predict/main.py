from order_predict.db import OrderDomain
from order_predict.modules import logger

logger.info('order count: %d', len(OrderDomain.get_all()))
