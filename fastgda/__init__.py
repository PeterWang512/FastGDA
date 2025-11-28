"""
FastGDA: Fast Gradient-based Data Attribution
A library for efficient data attribution in large-scale datasets.
"""

__version__ = "0.2.0"

from .models import FCModel, DualMLPModel
from .dataset import RankDatasetCOCO
from .utils import get_calibrated_feats, get_rank

__all__ = [
    'FCModel',
    'DualMLPModel', 
    'RankDatasetCOCO',
    'get_calibrated_feats',
    'get_rank',
]



