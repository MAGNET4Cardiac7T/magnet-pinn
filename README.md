# MAGNET-PINN
<img src="https://magnet4cardiac7t.github.io/assets/img/magnet_logo_venn.svg" width="400em" align="right" />

[![PyPI version](https://badge.fury.io/py/magnet_pinn.svg)](https://badge.fury.io/py/magnet_pinn)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![All Tests](https://github.com/MAGNET4Cardiac7T/magnet-pinn/actions/workflows/test_all.yaml/badge.svg)](https://github.com/MAGNET4Cardiac7T/magnet-pinn/actions/workflows/test_all.yaml)

[comment]: [![Docs](https://github.com/badulion/dynabench/actions/workflows/build_docs.yml/badge.svg)](https://dynabench.github.io)

Heart failure is one of the main causes of death worldwide, and high-resolution imaging plays a very important role in diagnosing it. 
Cardiac MRI at 7 Tesla (ultrahigh-field) provides excellent image quality because of its high signal-to-noise ratio (SNR) and spatial resolution.

However, its wider use is limited by the safety concerns related to the complex distribution of electromagnetic (EM) fields inside the body. 
These field distributions can lead to safety problems, such as localized tissue heating due to the radio frequency (RF) energy absorbed by the body during UHF MRI.

The simulations to accuratly predict how the EM field behaves inside the human body are complex and tedious. 
Therefore, a dataset was developed to immitate MRI images that can be used to train, validate, and test machine learning (ML) models, slashing the time for a good estimate of the EM field.

This package contains functions that can be applied to the dataset to preprocess it and finally use it as input to an ML model.
The package contains an easy-to-use interface to make the data readily available and fit it to the desired needs.

For more details check out the [documentation](https://magnet_pinn.github.io).


## ⚡️ Getting Started
To get started with the package, you can install it via pip:
```shell
pip install magnet_pinn
```

### Downloading Data
Download the datset using the following command:

```python

```

The dataset consists of multiple simulations which are tagged by their construction.
For example there is a simulation that containes two children and 4 tubes and is therefore tagged "children_2_tubes_4_id_xxxx".
A single simulation item contains the E and B-fields that are both nested under fields and the coils nested under the tag coils.
Moreover, both fields have a real and imaginary part and span all three spatial dimensions with $100\times100\times100$ points.
Additionally there exists a mask for the subject, the phases of the coils and another mask for the coils.

In the dataset, the fields are the results of extensive simulations and are used as the target for training a ML model.

#### Examplary slices of the absolute E and B-field:
<p align="center">
  <img src="docs/source/images/slices_e2.png" alt="E-field Slice" width="45%" />
  <img src="docs/source/images/slices_b2.png" alt="B-field Slice" width="45%" />
</p>

The simulation data needs to be placed in the data folder under `data/raw/GROUP_NAME/simulations` and the antenna data under `data/raw/GROUP_NAME/antenna`.

E.g.:
`data/raw/batch_1/simulations/children_0_tubes_0_id_3114`,
`data/raw/batch_1/antenna/Dipole_1.stl`, `data/raw/batch_1/antenna/materials.txt`

## ⚙️ Usage

### Loading & Preprocessing of the Data
Once the dataset is downloaded you can simply load the data in your project and start using it.
Based on your needs decide wether to use the grid layout of the datapoints (simple voxalization) or the pointcloud.
Then start by instantiating a preprocessor or use cli interface which will transform the data to the needed specifications.
Finally, instantiate an iterator to load the data.

### Example: Using the cli Interface for Preprocessing
An easy way to preprocess the data is the cli interface which enables the use directly from the command line.
To use the cli interface execute the following command, which will return instructions on how to use the function.
The processed data will be saved in the default output path `./data/processed`, where it can then be loaded from to be used in i.e. the iterator.
```shell
python -m magnet_pinn.preprocessing --help
```

A basic example of the usage when the data follows the general datastructure (described [here](#downloading-data)) is:
```shell
python -m magnet_pinn.preprocessing grid
```

### Example: Preprocessing and Loading Grid Data
```python
from magnet_pinn.preprocessing.preprocessing import GridPreprocessing
import numpy as np
# The prepocessor subclass for grid data
preprocessor = GridPreprocessing(
    ["data/raw/batches/batch_1", "data/raw/batches/batch_2"],   # simulation files to load
    "data/raw/antenna",                                         # path to the antenna file
    "data/processed/train",                                     # directory to save the processed data
    field_dtype=np.float32,                                     # data type of the field values
    x_min=-240,                                                 # kwargs
    x_max=240,
    y_min=-220,
    y_max=220,
    z_min=-250,
    z_max=250
)
# Process the simulation data and save it in the specified directory
preprocessor.process_simulations()
```
Using the preprocessed data from the previous step we then build an iterator instance where the data is loaded and a list of transforms is applied.

```python
from magnet_pinn.data.grid import MagnetGridIterator
from magnet_pinn.data.transforms import Crop, GridPhaseShift, Compose, DefaultTransform
# Compose a series of transformations to apply to the data
augmentation = Compose(
    [
        Crop(crop_size=(100, 100, 100)),
        GridPhaseShift(num_coils=8)
    ]
)
# Create an iterator for the processed grid data
iterator = MagnetGridIterator(
    "/home/andi/coding/data/magnet/processed/train/grid_voxel_size_4_data_type_float32",
    transforms=augmentation,
    num_samples=1
```
Normalization is an important part of the data pipeline and therefore we need to compute the normalization for the input and target data.
The normalization can be computed using the available example script.
It is useful to set the num_samples of the iterator to more than 1. We recommend using `num_samples=10`.

```python
from magnet_pinn.utils import MinMaxNormalizer, StandardNormalizer
import einops

class Iterator:
    def __init__(self, path, iterator):
        self.path = path
        self.iterator = iterator

    def __len__(self):
        return len(self.iterator)

    def __iter__(self):
        for batch in self.iterator:
            input = np.concatenate([batch['input'], batch['coils']], axis=0)
            target = einops.rearrange(batch['field'], 'he reim xyz ... -> (he reim xyz) ...')
            yield {
                'input': input,
                'target': target,
            }

iterator = Iterator("data/processed/train/grid_voxel_size_4_data_type_float32", iterator)

normalizer = StandardNormalizer()
normalizer.fit_params(iterator, key='input', axis=0)  # use either target or input
normalizer.save_as_json("data/processed/train/grid_voxel_size_4_data_type_float32/normalization/input_normalization.json")
```

### Example: Training a ML model
Once the data is preprocessed and ready, you can use it to train a ML model.
In the following the already instantiated iterator from [Example: Preprocessing and Loading Grid Data](#example-preprocessing-and-loading-grid-data) is used.
The full example can also be found in the `examples/` directory.
```python
import torch
import einops
from magnet_pinn.losses import MSELoss
from magnet_pinn.utils import StandardNormalizer
from magnet_pinn.data.utils import worker_init_fn
from magnet_pinn.models import UNet3D

# Set the base directory where the preprocessed data is stored
BASE_DIR = "data/processed/train/grid_voxel_size_4_data_type_float32"
target_normalizer = StandardNormalizer.load_from_json(f"{BASE_DIR}/normalization/target_normalization.json")
input_normalizer = StandardNormalizer.load_from_json(f"{BASE_DIR}/normalization/input_normalization.json")

# Create a DataLoader for the preprocessed data
train_loader = torch.utils.data.DataLoader(iterator, batch_size=4, num_workers=16, worker_init_fn=worker_init_fn)

# Create the model
model = UNet3D(5, 12, f_maps=32)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = MSELoss()
subject_lambda = 10.0
space_lambda = 0.01

for epoch in range(10):
    model.train()
    for i, batch in enumerate(train_loader):
        properties, phase, field, subject_mask = batch['input'], batch['coils'], batch['field'], batch['subject']
        x = input_normalizer(torch.cat([properties, phase], dim=1))
        y = target_normalizer(einops.rearrange(field, 'b he reim xyz ... -> b (he reim xyz) ...'))
        optimizer.zero_grad()
        y_hat = model(x)
        # calculate loss
        subject_loss = criterion(y_hat, y, subject_mask)
        space_loss = criterion(y_hat, y, ~subject_mask)
        loss = subject_loss*subject_lambda + space_loss*space_lambda

        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")
```

### Example: Generating Object Meshes
In the following it is shown how to generate your own sample data.
The following code snippet generates stl files with the given Tissue data.

```python
from numpy.random import default_rng
from magnet_pinn.generator.io import MeshWriter
from magnet_pinn.generator.phantoms import Tissue
from magnet_pinn.generator.samplers import PropertySampler
from magnet_pinn.generator.transforms import ToMesh, MeshesCutout, MeshesCleaning, Compose

# Step 1/4: Generate Tissue with blobs and tubes inside
tissue = Tissue(
    num_children_blobs=3,
    initial_blob_radius=100,
    initial_blob_center_extent={
        "x": [-5, 5],
        "y": [-5, 5],
        "z": [-50, 50],
    },
    blob_radius_decrease_per_level=0.3,
    num_tubes=10,
    relative_tube_max_radius=0.1,
    relative_tube_min_radius=0.01
)
raw_3d_structures = tissue.generate(seed=42)

# Step 2/4: Define the workflow for processing structures
workflow = Compose([
    ToMesh(),
    MeshesCutout(),
    MeshesCleaning()
])
meshes = workflow(raw_3d_structures)

# Step 3/4: Sample physical properties for the generated meshes
prop_sampler = PropertySampler(
    {
        "density": {
            "min": 400,
            "max": 2000
        },
        "conductivity": {
            "min": 0,
            "max": 2.5
        },
        "permittivity": {
            "min": 1,
            "max": 71
        }
    }
)
prop = prop_sampler.sample_like(meshes, rng=default_rng())

# Step 4/4: Save the generated meshes and properties to files
writer = MeshWriter("./gen_data/raw/tissue_meshes")
writer.write(meshes, prop)
```

## 🤝 How to contribute to *magnet-pinn*
This guide has been largely adapted from [the findiff contribution guide](https://github.com/maroba/findiff/blob/master/CONTRIBUTING.md)


#### **Did you find a bug?** 

* **Ensure the bug was not already reported** by searching on GitHub
  under [Issues](https://github.com/MAGNET4Cardiac7T/magnet-pinn/issues).

* If you're unable to find an open issue addressing the
  problem, [open a new one](https://github.com/MAGNET4Cardiac7T/magnet-pinn/issues/new). Be sure to include a **title and clear
  description**, as much relevant information as possible, and a **code sample** or an **executable test case**
  demonstrating the expected behavior that is not occurring.

#### **Did you write a patch that fixes a bug?**

* Open a new GitHub pull request with the patch.

* Ensure the PR description clearly describes the problem and solution. Include the relevant issue number if applicable.

#### **Do you intend to add a new feature or change an existing one?**

* Suggest your change in the [dynabench discussion forum](https://github.com/MAGNET4Cardiac7T/magnet-pinn/discussions) and
  start writing code.

* Do not open an issue on GitHub until you have collected positive feedback about the change. GitHub issues are
  primarily intended for bug reports and fixes.

#### **Do you have questions about the source code?**

* Ask any question about how to use *dynabench* in
  the [discussion forum](https://github.com/MAGNET4Cardiac7T/magnet-pinn/discussions).

Thank you for your support! :heart:

The *magnet-pinn* Team


## 📄 License
The content of this project itself, including the data and pretrained models, is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International Public License (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/). The underlying source code used to generate the data and train the models is licensed under the [MIT license](LICENSE).