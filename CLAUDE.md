# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MAGNET-PINN is a deep learning package for predicting electromagnetic (EM) fields in 7 Tesla cardiac MRI scanners using Physics-Informed Neural Networks. The package provides tools for:
- Generating synthetic tissue phantoms with realistic geometries
- Preprocessing EM field simulation data
- Training ML models (primarily 3D UNets) to predict E and B fields
- Physics-informed loss functions (divergence constraints, Faraday's law)

## Development Commands

### Dependency Management
This project uses Poetry for dependency management:
```bash
poetry install                    # Install all dependencies
poetry add <package>              # Add a new dependency
poetry add --group dev <package>  # Add a dev dependency
```

### Testing
```bash
poetry run pytest                      # Run all tests
poetry run pytest tests/path/to/test.py  # Run specific test file
poetry run pytest -k test_name         # Run specific test by name
poetry run pytest -v                   # Verbose output
```

### Code Quality
Pre-commit hooks are configured with uv (not poetry):
```bash
pre-commit run --all-files        # Run all linters manually
```

Linters (run via uv in pre-commit):
- **black**: Code formatting (config in pyproject.toml)
- **isort**: Import sorting (config in pyproject.toml)
- **flake8**: Linting (config in pyproject.toml)
- **mypy**: Type checking (config in pyproject.toml)

### Coverage
```bash
poetry run coverage run -m pytest  # Run tests with coverage
poetry run coverage report         # Show coverage report
poetry run coverage html           # Generate HTML coverage report
```

Coverage config is in `.coveragerc`:
- Source: `magnet_pinn`
- Omits: `tests/*`, `*__init__.py`

### Building Documentation
```bash
cd docs
make html                         # Build HTML documentation (uses Sphinx)
```

## Architecture

### Core Modules

#### 1. Data Pipeline (`magnet_pinn/data/`)
Handles loading and transforming preprocessed EM field data.

**Key Classes:**
- `MagnetBaseIterator`: Abstract base class for data iterators (extends `torch.utils.data.IterableDataset`)
- `MagnetGridIterator`: Loads voxelized grid data
- `MagnetPointIterator`: Loads point cloud data
- `DataItem`: Container for a single data sample

**Transforms:**
- `Compose`: Chains multiple transforms
- `Crop`: Spatial cropping
- `PhaseShift`/`CoilEnumeratorPhaseShift`: Apply phase shifts to coil data
- `PointSampling`: Sample points from grid

**Important:** At least one transform must perform phase shifting during data loading.

#### 2. Preprocessing (`magnet_pinn/preprocessing/`)
Converts raw simulation data to training-ready format.

**Key Classes:**
- `Preprocessing` (abstract): Base preprocessing pipeline
- `GridPreprocessing`: Voxelizes meshes and fields onto 3D grids
- `PointPreprocessing`: Extracts point clouds

**Process:**
1. Read antenna geometry and physical properties
2. Read EM field data (E-field, H-field) from simulations
3. Voxelize/sample spatial data
4. Save to HDF5 format with keys: `input`, `efield`, `hfield`, `subject`, `masks`

**CLI Usage:**
```bash
python -m magnet_pinn.preprocessing --help
python -m magnet_pinn.preprocessing grid    # Grid preprocessing
python -m magnet_pinn.preprocessing point   # Point cloud preprocessing
```

#### 3. Generator (`magnet_pinn/generator/`)
Creates synthetic tissue phantoms for simulation inputs.

**Key Components:**
- `Phantom` classes (`Tissue`, `MeshTissue`): High-level phantom generators
- `Sampler` classes (`BlobSampler`, `TubeSampler`): Sample geometric structures
- `Structure` classes (`Blob`, `Tube`, `CustomMeshStructure`): Geometric primitives
- `Transforms`: `ToMesh`, `MeshesCutout`, `MeshesCleaning`
- `PropertySampler`: Samples physical properties (density, conductivity, permittivity)
- `MeshWriter`: Saves generated meshes to STL files

**CLI Usage:**
```bash
python -m magnet_pinn.generator --help
```

#### 4. Models (`magnet_pinn/models/`)
Neural network architectures for EM field prediction.

**Available Models:**
- `UNet3D`: Standard 3D U-Net
- `ResidualUNet3D`: U-Net with residual connections
- `ResidualUNetSE3D`: Residual U-Net with Squeeze-and-Excitation blocks
- `UNet2D`, `ResidualUNet2D`: 2D variants

All models inherit from `AbstractUNet`.

#### 5. Losses (`magnet_pinn/losses/`)
Loss functions including physics-informed constraints.

**Basic Losses:**
- `MSELoss`, `MAELoss`, `HuberLoss`, `LogCoshLoss`

**Physics Losses:**
- `DivergenceLoss`: Enforces ∇·B = 0 (magnetic field divergence-free constraint)
- `FaradaysLawLoss`: Enforces Faraday's law (∇×E = -∂B/∂t)

**Constants:**
- `MRI_FREQUENCY_HZ`: 297.2e6 Hz (7 Tesla)
- `VACUUM_PERMEABILITY`: 4π × 10⁻⁷ H/m

#### 6. Utils (`magnet_pinn/utils/`)
- `StandardNormalizer`, `MinMaxNormalizer`: Data normalization classes
  - `.fit_params()`: Compute normalization parameters
  - `.save_as_json()` / `.load_from_json()`: Persist normalizers
- `worker_init_fn`: Initialize data loader workers properly

### Data Format

**Raw Data Structure:**
```
data/raw/GROUP_NAME/
├── simulations/
│   └── children_X_tubes_Y_id_ZZZZ/
│       ├── fields/ (E and B fields)
│       └── meshes/
└── antenna/
    ├── *.stl (antenna geometry)
    └── materials.txt
```

**Preprocessed Data Structure:**
```
data/processed/VARIANT/
├── simulations/
│   └── *.h5 files with keys:
│       - input: Material properties (density, conductivity, permittivity)
│       - efield: Electric field (complex, 3D)
│       - hfield: Magnetic field (complex, 3D)
│       - subject: Binary mask for subject region
│       - positions: Spatial coordinates (for point clouds)
├── antenna/
│   └── masks.h5: Coil masks
└── normalization/
    ├── input_normalization.json
    └── target_normalization.json
```

**Field Data Format:**
- E and B fields are complex-valued: separate real/imaginary components
- 3 spatial dimensions (x, y, z)
- Multiple field components: shape is typically `(field_component, real_imag, x, y, z)`
- Standard grid size: 100×100×100 points

### Training Workflow

1. **Preprocess data** using `GridPreprocessing` or CLI
2. **Compute normalization** on training set using `StandardNormalizer.fit_params()`
3. **Create iterator** with transforms (including phase shift)
4. **Create DataLoader** with `worker_init_fn`
5. **Train model** with combined physics and data losses:
   ```python
   subject_loss = mse_loss(pred, target, subject_mask)
   space_loss = mse_loss(pred, target, ~subject_mask)
   loss = subject_lambda * subject_loss + space_lambda * space_loss
   ```

See `examples/example_unet3d.py` for complete training example.

## Important Notes

- **Python version**: Requires Python >=3.11, <3.14
- **PyTorch**: Required for models and data loading
- **HDF5**: All processed data stored in `.h5` format
- **Phase shifts**: Critical for data augmentation during training
- **Testing**: Uses pytest with `--import-mode=importlib` (allows same-named test files in different directories)
- **Type hints**: Project uses mypy for type checking
- **Normalization**: Always normalize inputs and targets separately; save normalizers for inference
- **Physics constraints**: Divergence and curl constraints improve physical accuracy but are optional
