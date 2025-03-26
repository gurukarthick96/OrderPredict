from pymongo import MongoClient, ASCENDING

from order_predict import config
from order_predict.modules import logger

logger.info('connecting to database...')
client = MongoClient(config.DATABASE_URL)

order_db = client[config.DATABASE_NAME]

order_collection = order_db[config.ORDER_COLLECTION_NAME]


def ensure_indexes():
    logger.info('ensuring indexes...')
    order_collection.create_index([('created_at', ASCENDING)])
