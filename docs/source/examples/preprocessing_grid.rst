.. _grid_data:

-----------------------
Using Grid Data
-----------------------
To start off we can load the data from a simulation file and preprocess it.
The following code snippet shows how to load the data from a simulation file and preprocess it:
For further detail see the :ref:`API reference <api_>`.

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
