"""Prediction writer module for saving model predictions as H5 files.

This module provides classes for writing neural network predictions back to H5 format,
matching the structure created by preprocessing. Supports both grid and point cloud formats.
"""

from magnet_pinn.prediction.writer import PredictionWriter
from magnet_pinn.prediction.grid_writer import GridPredictionWriter
from magnet_pinn.prediction.point_writer import PointPredictionWriter

__all__ = ["PredictionWriter", "GridPredictionWriter", "PointPredictionWriter"]
