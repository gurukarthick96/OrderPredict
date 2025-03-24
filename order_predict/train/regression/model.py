from enum import Enum

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    LinearRegression,
    ElasticNet,
    SGDRegressor,
    Lasso,
    Ridge,
    BayesianRidge,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


class RegressionType(Enum):
    LINEAR = 'Linear Regression'
    RANDOM_FOREST = 'Random Forest'
    DECISION_TREE = 'Decision Tree'
    GRADIENT_BOOSTING = 'Gradient Boosting'
    SVR = 'Support Vector Regressor'
    KNN = 'K-Nearest Neighbors'
    ELASTIC_NET = 'ElasticNet'
    ADA_BOOST = 'AdaBoost'
    XGBOOST = 'XGBoost'
    SGDR = 'SGDRegressor'
    LASSO = 'Lasso'
    RIDGE = 'Ridge'
    BAYESIAN_RIDGE = 'BayesianRidge'
    KERNEL_RIDGE = 'KernelRidge'

    @staticmethod
    def lookup_by_name(name: str) -> 'RegressionType':
        return RegressionType.__members__.get(name)


def build_regression_model(reg_type):
    match reg_type:
        case RegressionType.LINEAR:
            return LinearRegression()
        case RegressionType.RANDOM_FOREST:
            return RandomForestRegressor(n_estimators=100, random_state=42)
        case RegressionType.DECISION_TREE:
            return DecisionTreeRegressor(random_state=42)
        case RegressionType.GRADIENT_BOOSTING:
            return MultiOutputRegressor(GradientBoostingRegressor(random_state=42))
        case RegressionType.SVR:
            return MultiOutputRegressor(SVR(kernel='rbf'))  # Radial basis function kernel
        case RegressionType.KNN:
            return KNeighborsRegressor(n_neighbors=5)
        case RegressionType.ELASTIC_NET:
            return ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        case RegressionType.ADA_BOOST:
            return MultiOutputRegressor(AdaBoostRegressor(n_estimators=100, random_state=42))
        case RegressionType.XGBOOST:
            return XGBRegressor(n_estimators=100, random_state=42)
        case RegressionType.SGDR:
            return MultiOutputRegressor(SGDRegressor(max_iter=1000, tol=1e-3, random_state=42))
        case RegressionType.LASSO:
            return Lasso(random_state=42)
        case RegressionType.RIDGE:
            return Ridge(random_state=42)
        case RegressionType.BAYESIAN_RIDGE:
            return MultiOutputRegressor(BayesianRidge())
        case RegressionType.KERNEL_RIDGE:
            return KernelRidge()
    pass
