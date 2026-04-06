"""Grid prediction writer for 3D voxel grid data."""

from pathlib import Path
from typing import Dict

import h5py
import numpy as np

from magnet_pinn.prediction.writer import PredictionWriter
from magnet_pinn.preprocessing.preprocessing import (
    FEATURES_OUT_KEY,
    SUBJECT_OUT_KEY,
    COORDINATES_OUT_KEY,
    PROCESSED_SIMULATIONS_DIR_PATH,
)


class GridPredictionWriter(PredictionWriter):
    """Prediction writer for grid-based (voxelized) data.

    Handles 3D voxel grid data with shape (feature, x, y, z).
    Preserves grid-specific metadata like voxel_size and spatial extents.
    """

    def _get_source_h5_path(self, simulation_name: str) -> Path:
        """Get path to source H5 file for grid simulation.

        Parameters
        ----------
        simulation_name : str
            Name of the simulation

        Returns
        -------
        Path
            Path to source H5 file

        Raises
        ------
        FileNotFoundError
            If source H5 file does not exist
        """
        path = self.source_data_dir / PROCESSED_SIMULATIONS_DIR_PATH / f"{simulation_name}.h5"
        if not path.exists():
            raise FileNotFoundError(f"Source H5 file not found: {path}")
        return path

    def _load_passthrough_data(self, source_path: Path) -> Dict[str, np.ndarray]:
        """Load passthrough datasets from source H5 file.

        Loads grid data with shape (feature, x, y, z) for input,
        (s, x, y, z) for subject, and (3, x, y, z) for positions.

        Parameters
        ----------
        source_path : Path
            Path to source H5 file

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with keys: 'input', 'subject', 'positions'
        """
        with h5py.File(source_path, "r") as f:
            return {
                "input": f[FEATURES_OUT_KEY][:],
                "subject": f[SUBJECT_OUT_KEY][:],
                "positions": f[COORDINATES_OUT_KEY][:],
            }
