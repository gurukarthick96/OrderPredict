from datetime import datetime

from pymongo import ASCENDING

from order_predict.db.database import order_collection


class OrderDomain:
    id: str
    total: float
    created_at: datetime

    def __init__(self, order_id: str, total: float, created_at: datetime):
        self.id = order_id
        self.total = total
        self.created_at = created_at

    @staticmethod
    def get_all() -> list['OrderDomain']:
        order_docs = order_collection.find({}).sort('created_at', ASCENDING)
        return [OrderDomain.from_dict(order_doc) for order_doc in order_docs]

    @classmethod
    def from_dict(cls, order_doc: dict) -> 'OrderDomain':
        return cls(order_id=order_doc['_id'], total=order_doc['total'], created_at=order_doc['created_at'])
