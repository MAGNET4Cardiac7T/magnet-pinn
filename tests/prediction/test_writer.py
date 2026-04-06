"""Unit tests for prediction writer."""

import pytest
import numpy as np
import torch
from pathlib import Path

from magnet_pinn.prediction.writer import to_structured_dtype, PredictionWriter
from magnet_pinn.prediction.grid_writer import GridPredictionWriter


class TestToStructuredDtype:
    """Tests for to_structured_dtype helper function."""

    def test_basic_conversion(self):
        """Test basic conversion to structured dtype."""
        # Input: (8, 3, 2, 10, 10, 10)
        field = np.random.randn(8, 3, 2, 10, 10, 10).astype(np.float32)

        structured = to_structured_dtype(field, np.float32)

        assert structured.shape == (8, 3, 10, 10, 10)
        assert structured.dtype.names == ("re", "im")
        assert np.array_equal(structured["re"], field[:, :, 0, ...])
        assert np.array_equal(structured["im"], field[:, :, 1, ...])

    def test_different_dtypes(self):
        """Test conversion with different dtypes."""
        field = np.random.randn(8, 3, 2, 5, 5, 5).astype(np.float64)

        structured = to_structured_dtype(field, np.float32)

        assert structured.dtype["re"] == np.float32
        assert structured.dtype["im"] == np.float32

    def test_point_cloud_shape(self):
        """Test conversion with point cloud shape."""
        # Input: (8, 3, 2, 1000)
        field = np.random.randn(8, 3, 2, 1000).astype(np.float32)

        structured = to_structured_dtype(field, np.float32)

        assert structured.shape == (8, 3, 1000)
        assert structured.dtype.names == ("re", "im")


class TestGridPredictionWriter:
    """Tests for GridPredictionWriter class."""

    def test_initialization(self, tmp_path, mock_source_dir, mock_model, mock_normalizer):
        """Test writer initialization."""
        output_dir = tmp_path / "predictions"

        writer = GridPredictionWriter(
            output_dir=output_dir,
            source_data_dir=mock_source_dir,
            target_normalizer=mock_normalizer,
            model=mock_model,
            device="cpu",
        )

        assert writer.output_dir.exists()
        assert writer.num_coils == 8
        assert writer.field_dtype == np.float32

    def test_initialization_missing_source_dir(self, tmp_path, mock_model, mock_normalizer):
        """Test initialization with missing source directory."""
        output_dir = tmp_path / "predictions"
        nonexistent = tmp_path / "nonexistent"

        with pytest.raises(FileNotFoundError, match="Source data directory does not exist"):
            GridPredictionWriter(
                output_dir=output_dir,
                source_data_dir=nonexistent,
                target_normalizer=mock_normalizer,
                model=mock_model,
            )

    def test_transform_predictions_to_fields(
        self, tmp_path, mock_source_dir, mock_model, mock_normalizer, mock_coil_predictions
    ):
        """Test full transformation pipeline."""
        output_dir = tmp_path / "predictions"

        writer = GridPredictionWriter(
            output_dir=output_dir,
            source_data_dir=mock_source_dir,
            target_normalizer=mock_normalizer,
            model=mock_model,
        )

        efield, hfield = writer._transform_predictions_to_fields(mock_coil_predictions)

        # Check shapes
        assert efield.shape == (8, 3, 100, 100, 100)
        assert hfield.shape == (8, 3, 100, 100, 100)

        # Check dtypes
        assert efield.dtype.names == ("re", "im")
        assert hfield.dtype.names == ("re", "im")
        assert efield.dtype["re"] == np.float32
        assert hfield.dtype["im"] == np.float32

    def test_transform_shape_mismatch_raises_error(
        self, tmp_path, mock_source_dir, mock_model, mock_normalizer
    ):
        """Test error when coil predictions have different shapes."""
        output_dir = tmp_path / "predictions"

        writer = GridPredictionWriter(
            output_dir=output_dir,
            source_data_dir=mock_source_dir,
            target_normalizer=mock_normalizer,
            model=mock_model,
        )

        # Create predictions with mismatched shapes
        predictions = [np.random.randn(12, 100, 100, 100).astype(np.float32) for _ in range(7)]
        predictions.append(np.random.randn(12, 50, 50, 50).astype(np.float32))  # Wrong shape

        with pytest.raises(ValueError, match="Shape mismatch"):
            writer._transform_predictions_to_fields(predictions)

    def test_accumulate_single_coil(
        self, tmp_path, mock_source_dir, mock_model, mock_normalizer, mock_batch
    ):
        """Test accumulating one coil prediction."""
        output_dir = tmp_path / "predictions"

        writer = GridPredictionWriter(
            output_dir=output_dir,
            source_data_dir=mock_source_dir,
            target_normalizer=mock_normalizer,
            model=mock_model,
        )

        predictions = torch.randn(1, 12, 100, 100, 100)

        writer._accumulate_prediction(mock_batch, predictions)

        assert "test_sim" in writer._coil_accumulator
        assert len(writer._coil_accumulator["test_sim"]["predictions"]) == 1

    def test_accumulate_multiple_coils(
        self, tmp_path, mock_source_dir, mock_model, mock_normalizer, mock_batch
    ):
        """Test accumulating 8 coils sequentially."""
        output_dir = tmp_path / "predictions"

        writer = GridPredictionWriter(
            output_dir=output_dir,
            source_data_dir=mock_source_dir,
            target_normalizer=mock_normalizer,
            model=mock_model,
        )

        for i in range(8):
            predictions = torch.randn(1, 12, 100, 100, 100)
            writer._accumulate_prediction(mock_batch, predictions)

        assert len(writer._coil_accumulator["test_sim"]["predictions"]) == 8
        assert writer._is_simulation_complete("test_sim")

    def test_get_source_h5_path(
        self, tmp_path, mock_source_dir, mock_model, mock_normalizer, mock_grid_h5_file
    ):
        """Test getting source H5 path."""
        output_dir = tmp_path / "predictions"

        writer = GridPredictionWriter(
            output_dir=output_dir,
            source_data_dir=mock_source_dir,
            target_normalizer=mock_normalizer,
            model=mock_model,
        )

        path = writer._get_source_h5_path("test_sim")

        assert path.exists()
        assert path.name == "test_sim.h5"

    def test_get_source_h5_path_missing_raises_error(
        self, tmp_path, mock_source_dir, mock_model, mock_normalizer
    ):
        """Test error when source H5 doesn't exist."""
        output_dir = tmp_path / "predictions"

        writer = GridPredictionWriter(
            output_dir=output_dir,
            source_data_dir=mock_source_dir,
            target_normalizer=mock_normalizer,
            model=mock_model,
        )

        with pytest.raises(FileNotFoundError, match="Source H5 file not found"):
            writer._get_source_h5_path("nonexistent_sim")

    def test_load_passthrough_data(
        self, tmp_path, mock_source_dir, mock_model, mock_normalizer, mock_grid_h5_file
    ):
        """Test loading passthrough data from source H5."""
        output_dir = tmp_path / "predictions"

        writer = GridPredictionWriter(
            output_dir=output_dir,
            source_data_dir=mock_source_dir,
            target_normalizer=mock_normalizer,
            model=mock_model,
        )

        passthrough = writer._load_passthrough_data(mock_grid_h5_file)

        assert "input" in passthrough
        assert "subject" in passthrough
        assert "positions" in passthrough
        assert passthrough["input"].shape == (3, 100, 100, 100)
        assert passthrough["subject"].shape == (6, 100, 100, 100)
        assert passthrough["positions"].shape == (3, 100, 100, 100)

    def test_load_metadata(
        self, tmp_path, mock_source_dir, mock_model, mock_normalizer, mock_grid_h5_file
    ):
        """Test loading and caching metadata."""
        output_dir = tmp_path / "predictions"

        writer = GridPredictionWriter(
            output_dir=output_dir,
            source_data_dir=mock_source_dir,
            target_normalizer=mock_normalizer,
            model=mock_model,
        )

        metadata1 = writer._load_metadata(mock_grid_h5_file)
        metadata2 = writer._load_metadata(mock_grid_h5_file)

        assert "voxel_size" in metadata1
        assert metadata1["voxel_size"] == 4
        assert metadata1 is metadata2  # Same object reference (cached)

    def test_run_inference_wrong_batch_size_raises_error(
        self, tmp_path, mock_source_dir, mock_model, mock_normalizer
    ):
        """Test error when batch size is not 1."""
        output_dir = tmp_path / "predictions"

        writer = GridPredictionWriter(
            output_dir=output_dir,
            source_data_dir=mock_source_dir,
            target_normalizer=mock_normalizer,
            model=mock_model,
        )

        # Create batch with size 2
        batch = {
            'simulation': ['test_sim1', 'test_sim2'],
            'input': torch.randn(2, 3, 100, 100, 100),
            'coils': torch.randn(2, 2, 100, 100, 100),
        }

        from torch.utils.data import DataLoader

        # Simulate dataloader iteration
        with pytest.raises(ValueError, match="Batch size must be 1"):
            writer.write_predictions([batch])
