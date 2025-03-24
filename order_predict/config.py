from order_predict.env import get_env

ENVIRONMENT = get_env('ENVIRONMENT')

SERVICE_NAME = get_env('SERVICE_NAME', 'OrderPredict-App')
BASE_PATH = get_env('BASE_PATH', '/order-predict-app')

SERVER_HOST = get_env('SERVER_HOST')
SERVER_PORT = get_env('SERVER_PORT', required_type=int)
SERVER_RELOAD = get_env('SERVER_RELOAD', False, required_type=bool)
SERVER_MAX_WORKERS = get_env('SERVER_MAX_WORKERS', 1, required_type=int)

LOGGING_LEVEL = get_env('LOGGING_LEVEL', 'INFO')
LOGGING_FORMAT = '%(asctime)s %(levelname)s %(name)s %(filename)s:%(lineno)d -- %(message)s'

DATABASE_URL = get_env('DATABASE_URL')
DATABASE_NAME = get_env('DATABASE_NAME')
ORDER_COLLECTION_NAME = get_env('ORDER_COLLECTION_NAME', 'AI_Order')

EXECUTOR_MAX_WORKERS = get_env('EXECUTOR_MAX_WORKERS', 5, required_type=int)

ORDER_DF_ADD_MISSING_DATE = True
ORDER_DF_DATE_FIELDS_TO_EXTRACT = ['day_of_week', 'holiday']
# ORDER_DF_DATE_FIELDS_TO_EXTRACT = ['day_of_week', 'month', 'day_of_month', 'week_of_year', 'holiday']
# ORDER_DF_OUTLIER_CORRECTION_STRATEGY = 'CAP'
ORDER_DF_OUTLIER_CORRECTION_STRATEGY = 'FILTER'

ORDER_TRAIN_TEST_SPLIT_SIZE = 0.1
ORDER_TRAIN_APPLY_SCALER = False
ORDER_TRAIN_REGRESSION_TYPE = None
# ORDER_TRAIN_REGRESSION_TYPE = 'LINEAR'
