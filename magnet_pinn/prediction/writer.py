"""Base prediction writer class for saving model predictions as H5 files."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any, Optional
import logging

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import einops

from magnet_pinn.utils import StandardNormalizer
from magnet_pinn.preprocessing.preprocessing import (
    FEATURES_OUT_KEY,
    E_FIELD_OUT_KEY,
    H_FIELD_OUT_KEY,
    SUBJECT_OUT_KEY,
    COORDINATES_OUT_KEY,
    DTYPE_OUT_KEY,
    TRUNCATION_COEFFICIENTS_OUT_KEY,
    PROCESSED_SIMULATIONS_DIR_PATH,
)

logger = logging.getLogger(__name__)


def to_structured_dtype(field: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Convert field array to structured dtype with 're'/'im' fields.

    Args:
        field: Shape (coils, xyz, reim, spatial_dims...) where dim 2 is [real, imag]
        dtype: Target dtype (e.g., np.float32)

    Returns:
        Structured array shape (coils, xyz, spatial_dims...) with dtype=[('re', dtype), ('im', dtype)]
    """
    coils, xyz = field.shape[:2]
    spatial_shape = field.shape[3:]

    # Create structured array
    structured = np.empty(
        (coils, xyz, *spatial_shape),
        dtype=[("re", dtype), ("im", dtype)]
    )

    # Assign real and imaginary parts
    structured["re"] = field[:, :, 0, ...]
    structured["im"] = field[:, :, 1, ...]

    return structured


class PredictionWriter(ABC):
    """Abstract base class for writing model predictions to H5 files.

    This class provides shared logic for accumulating coil predictions, managing H5 I/O,
    and orchestrating the write process. Subclasses handle grid vs point cloud specifics.

    Parameters
    ----------
    output_dir : Union[str, Path]
        Directory where prediction H5 files will be saved
    source_data_dir : Union[str, Path]
        Directory containing source H5 files for metadata/passthrough data
    target_normalizer : StandardNormalizer
        Normalizer for denormalizing model output predictions
    model : torch.nn.Module
        Trained neural network model
    input_normalizer : Optional[StandardNormalizer]
        Optional normalizer for model inputs
    device : str
        Torch device for inference (default: "cpu")
    field_dtype : np.dtype
        Dtype for field storage (default: np.float32)
    num_coils : int
        Expected number of coils (default: 8)
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        source_data_dir: Union[str, Path],
        target_normalizer: StandardNormalizer,
        model: torch.nn.Module,
        input_normalizer: Optional[StandardNormalizer] = None,
        device: str = "cpu",
        field_dtype: np.dtype = np.float32,
        num_coils: int = 8,
    ):
        """Initialize prediction writer."""
        self.output_dir = Path(output_dir)
        self.source_data_dir = Path(source_data_dir)
        self.target_normalizer = target_normalizer
        self.input_normalizer = input_normalizer
        self.model = model
        self.device = device
        self.field_dtype = np.dtype(field_dtype)
        self.num_coils = num_coils

        # Internal state
        self._coil_accumulator: Dict[str, Dict[str, Any]] = {}
        self._metadata_cache: Dict[Path, Dict[str, Any]] = {}

        # Validation
        self._validate_initialization()

        # Setup
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.eval()
        self.model.to(self.device)

    def _validate_initialization(self) -> None:
        """Validate initialization parameters."""
        if not self.source_data_dir.exists():
            raise FileNotFoundError(f"Source data directory does not exist: {self.source_data_dir}")

        sim_dir = self.source_data_dir / PROCESSED_SIMULATIONS_DIR_PATH
        if not sim_dir.exists():
            raise FileNotFoundError(f"Simulations directory does not exist: {sim_dir}")

        if not isinstance(self.model, torch.nn.Module):
            raise TypeError(f"Model must be torch.nn.Module, got {type(self.model)}")

        # Validate normalizer has required methods (duck typing for testing)
        if not hasattr(self.target_normalizer, '__call__') or not hasattr(self.target_normalizer, 'inverse'):
            raise TypeError(f"Target normalizer must have __call__ and inverse methods")

    def write_predictions(self, dataloader: DataLoader) -> Dict[str, Path]:
        """Write predictions for all simulations in dataloader.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader with CoilEnumeratorPhaseShift transform and batch_size=1

        Returns
        -------
        Dict[str, Path]
            Mapping of simulation names to output H5 file paths
        """
        written_files = {}

        try:
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Writing predictions"):
                    # Validate batch size
                    batch_size = batch['input'].shape[0]
                    if batch_size != 1:
                        raise ValueError(f"Batch size must be 1 for coil enumeration, got {batch_size}")

                    # Run inference
                    try:
                        predictions = self._run_inference(batch)
                    except Exception as e:
                        sim_name = batch['simulation'][0]
                        logger.error(f"Inference failed for {sim_name}: {e}")
                        raise

                    # Accumulate prediction
                    self._accumulate_prediction(batch, predictions)

                    # Check for complete simulations and write
                    for sim_name in list(self._coil_accumulator.keys()):
                        if self._is_simulation_complete(sim_name):
                            try:
                                output_path = self.write_simulation(sim_name)
                                written_files[sim_name] = output_path
                                logger.info(f"Wrote predictions for {sim_name} to {output_path}")
                            except Exception as e:
                                logger.error(f"Failed to write {sim_name}: {e}")
                                raise

        finally:
            # Cleanup on error
            self._coil_accumulator.clear()
            self._metadata_cache.clear()

        return written_files

    def write_simulation(self, simulation_name: str) -> Path:
        """Write accumulated predictions for one simulation to H5 file.

        Parameters
        ----------
        simulation_name : str
            Name of the simulation

        Returns
        -------
        Path
            Path to created H5 file
        """
        # Validate coil count
        n_coils = len(self._coil_accumulator[simulation_name]["predictions"])
        if n_coils != self.num_coils:
            raise ValueError(
                f"Incomplete predictions for {simulation_name}: "
                f"expected {self.num_coils} coils, got {n_coils}"
            )

        # Load source data
        source_path = self._get_source_h5_path(simulation_name)
        passthrough_data = self._load_passthrough_data(source_path)
        metadata = self._load_metadata(source_path)

        # Transform predictions to field format
        efield, hfield = self._transform_predictions_to_fields(
            self._coil_accumulator[simulation_name]["predictions"]
        )

        # Prepare output data
        output_data = {
            FEATURES_OUT_KEY: passthrough_data["input"],
            E_FIELD_OUT_KEY: efield,
            H_FIELD_OUT_KEY: hfield,
            SUBJECT_OUT_KEY: passthrough_data["subject"],
            COORDINATES_OUT_KEY: passthrough_data["positions"],
        }

        # Write H5
        output_path = self.output_dir / f"{simulation_name}.h5"
        self._write_h5_file(output_path, output_data, metadata)

        # Clean up accumulator for this simulation
        del self._coil_accumulator[simulation_name]

        return output_path

    def _run_inference(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run model inference on batch.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch dictionary with 'input' and 'coils' keys

        Returns
        -------
        torch.Tensor
            Model predictions, shape (batch, channels, spatial_dims...)
        """
        # Extract inputs (matching training pattern from example_unet3d.py)
        properties = batch['input'].to(self.device)
        phase = batch['coils'].to(self.device)
        x = torch.cat([properties, phase], dim=1)

        # Apply input normalization if provided
        if self.input_normalizer is not None:
            x = self.input_normalizer(x)

        # Forward pass
        predictions = self.model(x)

        # Validate output shape
        if predictions.ndim < 3:
            raise ValueError(f"Model output must have at least 3 dimensions, got shape {predictions.shape}")

        expected_channels = 12  # 6 E-field + 6 H-field
        if predictions.shape[1] != expected_channels:
            logger.warning(
                f"Expected {expected_channels} output channels, got {predictions.shape[1]}. "
                "Ensure model outputs both E and H fields."
            )

        return predictions

    def _accumulate_prediction(self, batch: Dict[str, torch.Tensor], predictions: torch.Tensor) -> None:
        """Accumulate prediction for one coil.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch dictionary containing 'simulation' key
        predictions : torch.Tensor
            Model predictions for this coil
        """
        # Denormalize
        predictions = self.target_normalizer.inverse(predictions)

        # Extract simulation name
        sim_name = batch['simulation'][0]  # Batch size is 1

        # Initialize accumulator for new simulation
        if sim_name not in self._coil_accumulator:
            self._coil_accumulator[sim_name] = {
                "predictions": [],
            }

        # Append prediction (detach and move to CPU to free GPU memory)
        pred_cpu = predictions[0].cpu().numpy()  # Remove batch dim
        self._coil_accumulator[sim_name]["predictions"].append(pred_cpu)

    def _is_simulation_complete(self, simulation_name: str) -> bool:
        """Check if all coils have been accumulated for a simulation.

        Parameters
        ----------
        simulation_name : str
            Name of the simulation

        Returns
        -------
        bool
            True if num_coils predictions accumulated
        """
        return len(self._coil_accumulator[simulation_name]["predictions"]) == self.num_coils

    def _transform_predictions_to_fields(
        self, predictions: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform list of coil predictions to E/H field format.

        Transforms predictions from model output format to H5 storage format:
        1. Stack 8 coil predictions
        2. Split into E-field and H-field
        3. Reshape to separate real/imag components
        4. Convert to structured dtype

        Parameters
        ----------
        predictions : List[np.ndarray]
            List of 8 predictions, each shape (12, spatial_dims...)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            E-field and H-field arrays with structured dtype [('re', dtype), ('im', dtype)]
            Shape: (8, 3, spatial_dims...)
        """
        # Validate shapes match
        shapes = [p.shape for p in predictions]
        if not all(s == shapes[0] for s in shapes):
            raise ValueError(f"Shape mismatch across coil predictions: {shapes}")

        # Step 1: Stack coils
        stacked = np.stack(predictions, axis=0)  # (8, 12, ...)

        # Step 2: Split E-field and H-field
        # First 6 channels: E-field (3 components × 2 real/imag)
        # Last 6 channels: H-field (3 components × 2 real/imag)
        e_flat = stacked[:, :6, ...]   # (8, 6, ...)
        h_flat = stacked[:, 6:, ...]   # (8, 6, ...)

        # Step 3: Reshape to (coils, components, real/imag, spatial)
        # Order: [Ex_re, Ex_im, Ey_re, Ey_im, Ez_re, Ez_im] → (3, 2, ...)
        e_reshaped = einops.rearrange(e_flat, "coils (xyz reim) ... -> coils xyz reim ...", xyz=3, reim=2)
        h_reshaped = einops.rearrange(h_flat, "coils (xyz reim) ... -> coils xyz reim ...", xyz=3, reim=2)

        # Step 4: Convert to structured dtype
        efield = to_structured_dtype(e_reshaped, self.field_dtype)
        hfield = to_structured_dtype(h_reshaped, self.field_dtype)

        return efield, hfield

    def _load_metadata(self, source_path: Path) -> Dict[str, Any]:
        """Load and cache metadata from source H5 file.

        Parameters
        ----------
        source_path : Path
            Path to source H5 file

        Returns
        -------
        Dict[str, Any]
            Metadata attributes from source file
        """
        if source_path not in self._metadata_cache:
            with h5py.File(source_path, "r") as f:
                metadata = dict(f.attrs)
            self._metadata_cache[source_path] = metadata
        return self._metadata_cache[source_path]

    def _write_h5_file(
        self, output_path: Path, data: Dict[str, np.ndarray], metadata: Dict[str, Any]
    ) -> None:
        """Write prediction data to H5 file.

        Parameters
        ----------
        output_path : Path
            Path to output H5 file
        data : Dict[str, np.ndarray]
            Dictionary of datasets to write
        metadata : Dict[str, Any]
            Metadata attributes to write
        """
        try:
            with h5py.File(output_path, "w") as f:
                # Write datasets
                for key, array in data.items():
                    f.create_dataset(key, data=array)

                # Write metadata attributes
                f.attrs[DTYPE_OUT_KEY] = self.field_dtype.name
                for key, value in metadata.items():
                    if key != DTYPE_OUT_KEY:  # Skip to avoid conflict
                        f.attrs[key] = value
        except OSError as e:
            if "No space left" in str(e):
                logger.error(f"Disk full, cannot write {output_path}")
            raise

    @abstractmethod
    def _get_source_h5_path(self, simulation_name: str) -> Path:
        """Get path to source H5 file for a simulation.

        Parameters
        ----------
        simulation_name : str
            Name of the simulation

        Returns
        -------
        Path
            Path to source H5 file
        """
        pass

    @abstractmethod
    def _load_passthrough_data(self, source_path: Path) -> Dict[str, np.ndarray]:
        """Load passthrough datasets from source H5 file.

        Loads data that is copied unchanged: input, subject, positions.

        Parameters
        ----------
        source_path : Path
            Path to source H5 file

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with keys: 'input', 'subject', 'positions'
        """
        pass
