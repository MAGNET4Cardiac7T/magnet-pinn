.. _point_data:

-----------------------
Using Point Cloud Data
-----------------------
To start off we can load the data from a simulation file and preprocess it.
The following code snippet shows how to load the data from a simulation file and preprocess it:
For further detail see the :ref:`API reference <api_>`.

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
