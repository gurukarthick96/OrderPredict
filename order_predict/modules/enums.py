from enum import Enum


class OutlierCorrectionStrategy(Enum):
    CAP = 'CAP'  # Cap outliers at lower/upper bounds
    FILTER = 'FILTER'  # Remove outliers
    NONE = 'NONE'  # No action
