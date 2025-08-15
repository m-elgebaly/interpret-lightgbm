"""Top-level package for interpret_lightgbm.

Public API exposes progressive LightGBM models and interpretation utilities.
"""
from .model import ProgressiveLGBMRegressor, ProgressiveLGBMClassifier
from .interpret import (
    interpret_prediction
)


__all__ = [
    "ProgressiveLGBMRegressor",
    "ProgressiveLGBMClassifier",
    "interpret_prediction",

]
__version__ = "0.1.0"
