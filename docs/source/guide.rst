===================
User Guide
===================
This guide is an overview of the usage of the magnet-pinn package.
It is intended to help users understand how to load and preporcess simulations of EM-Fields inside a MRI scanner using this package and the published datasets.
The guide contains the following sections

- :ref:`Using Grid Data <grid_data>`
- :ref:`Using Point Cloud Data <point_data>`
- :ref:`Normalization <Normalization>`
- :ref:`Training a PINN <PINN_training>`

.. _install:
---------------------------
:ref:`Installation <start>`
---------------------------

.. _usage:
--------------------
Usage
--------------------
The main purpose of this package is to load simulation files that can be used to e.g. train a PINN model.
There are multiple useful functions that make it easy to load the data and preprocess it.
Finally the data can be saved in a format that is easy to use with a specified model.
The python snippets can also be found in the examples folder of the package.
The Data can be loaded in two formats:

- Grid data: The data is stored in a grid format, where the field values are stored in a grid structure.
- Point cloud data: The data is stored as a point cloud, where each point has a field value associated with it.

The usage is similar for both formats, but there are some differences in the preprocessing steps and the way the data is loaded.
Therefore, we explore both formats separately.

Make sure you have installed the magnet-pinn package as well as the required dependencies.
We recommend using a virtual environment to manage the dependencies of the package.

- `NumPy <https://numpy.org/install/>`_
- `Torch <https://pytorch.org/get-started/locally/>`_

.. _grid_data:
^^^^^^^^^^^^^^^^^^^
Using Grid Data
^^^^^^^^^^^^^^^^^^^
To start off we can load the data from a simulation file and preprocess it.
The following code snippet shows how to load the data from a simulation file and preprocess it:
For further detail see the :ref:`API reference <api>`.

.. code-block:: python

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

The next step is to build an iterator that can be used to load the now processed data.
The following code snippet shows how to build an iterator that can be used to load the processed data:

.. code-block:: python

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
    )

.. _point_data:
^^^^^^^^^^^^^^^^^^^^^^
Using Point Cloud Data
^^^^^^^^^^^^^^^^^^^^^^
To start off we can load the data from a simulation file and preprocess it.
The following code snippet shows how to load the data from a simulation file and preprocess it:
For further detail see the :ref:`API reference <api>`.

.. code-block:: python

    from magnet_pinn.preprocessing.preprocessing import PointPreprocessing
    import numpy as np

    # The prepocessor subclass for point data
    point_preprocessor = PointPreprocessing(
        ["data/raw/batches/batch_1", "data/raw/batches/batch_2"],   # simulation files to load
        "data/raw/antenna",                                         # path to the antenna file
        "data/processed/train",                                     # directory to save the processed data
        field_dtype=np.float32                                      # data type of the field values
    )

    # Process the simulation data and save it in the specified directory
    point_preprocessor.process_simulations()

The next step is to build an iterator that can be used to load the now processed data.
The following code snippet shows how to build an iterator that can be used to load the processed data:

.. code-block:: python

    from magnet_pinn.data.point import MagnetPointIterator
    from magnet_pinn.data.transforms import PointSampling, PointPhaseShift, Compose, PointFeatureRearrange

    # Compose a series of transformations to apply to the data
    augmentation = Compose(
        [
            PointSampling(points_sampled=1000),
            PointPhaseShift(num_coils=8),
            PointFeatureRearrange(num_coils=8)
        ]
    )

    # Create an iterator for the processed point data
    iterator = MagnetPointIterator(
        "data/processed/train/point_data_type_float32",
        transforms=augmentation,
        num_samples=100
    )

.. _Normalization:
^^^^^^^^^^^^^^^^^^^^^^^
Normalization
^^^^^^^^^^^^^^^^^^^^^^^
Normalization is an important step in the preprocessing pipeline to ensure that the data is in a suitable range for training models.
The `StandardNormalizer` class can be used to normalize the data.
The following code snippet shows how to normalize the data:


.. _PINN_training:
-----------------------
Training a ML model
-----------------------
Once the data is preprocessed and ready, you can use it to train a model.
The following code snippet shows how to train a PINN using the preprocessed data:

.. code-block:: python

    import torch
    import einops
    from magnet_pinn.losses import MSELoss
    from magnet_pinn.utils import StandardNormalizer
    from magnet_pinn.data.utils import worker_init_fn
    from magnet_pinn.models import UNet3D

    # Set the base directory where the preprocessed data is stored
    BASE_DIR = "data/processed/train/grid_voxel_size_4_data_type_float32"

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
            x = torch.cat([properties, phase], dim=1)
            y = einops.rearrange(field, 'b he reim xyz ... -> b (he reim xyz) ...')
            optimizer.zero_grad()
            y_hat = model(x)
            # calculate loss
            subject_loss = criterion(y_hat, y, subject_mask)
            space_loss = criterion(y_hat, y, ~subject_mask)
            loss = subject_loss*subject_lambda + space_loss*space_lambda

            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")