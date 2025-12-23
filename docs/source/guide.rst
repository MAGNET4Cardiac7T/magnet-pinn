===================
Tutorial
===================
This guide is an overview of the usage of the magnet-pinn package.
It is intended to help users understand how to load and preporcess simulations of EM-Fields inside a MRI scanner using this package and the published datasets.

.. _install:

----------------------------
:ref:`Installation <start_>`
----------------------------
For installation follow the steps indicated in the getting started guide. There you will also find a guide on how to download the dataset.

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

There are two ways to preprocess the data, either using the command line interface (CLI) or using the python functions directly.
The CLI is the easiest way to preprocess the data, as it requires no coding.
The python functions give more flexibility and allow for more advanced preprocessing steps.
The CLI is explained in the following section, while the python functions are explained in the examples section.

Make sure you have installed the magnet-pinn package as well as the required dependencies.
We recommend using a virtual environment to manage the dependencies of the package.

^^^^^^^^^^^^^^^^^^^^^^^
Using the CLI interface
^^^^^^^^^^^^^^^^^^^^^^^
An easy way to preprocess the data is the cli interface which enables the use directly from the command line.
To use the cli interface execute the following command, which will return instructions on how to use the function.
The processed data will be saved in the default output path `./data/processed`, where it can then be loaded from to be used in i.e. the iterator.

.. code-block:: shell

    python -m magnet_pinn.preprocessing --help

A basic example of the usage when the data follows the general datastructure is:

.. code-block:: shell

    python -m magnet_pinn.preprocessing grid


^^^^^^^^^^^^^^^^^^^^^^^^
Using Python functions
^^^^^^^^^^^^^^^^^^^^^^^^

To preprocess the data directly in python we can use the preprocessing functions provided in the package.

**Grid Data**

For grid data we can use the `GridPreprocessing` class.

.. code-block:: python

    from magnet_pinn.preprocessing.preprocessing import GridPreprocessing

    preprocessor = GridPreprocessing(
        simulations_dir_path = ["data/raw/batches/batch_1", "data/raw/batches/batch_2"],
        antenna_dir_path = "data/raw/antenna",
        output_dir_path = "data/processed/train"
    )

**Point Cloud Data**

The same can be done for point cloud data using the `PointCloudPreprocessing` class.

.. code-block:: python

    from magnet_pinn.preprocessing.preprocessing import PointCloudPreprocessing

    preprocessor = PointCloudPreprocessing(
        simulations_dir_path = ["data/raw/batches/batch_1", "data/raw/batches/batch_2"],
        antenna_dir_path = "data/raw/antenna",
        output_dir_path = "data/processed/train"
    )

    preprocessor.process_simulations()

After initializing the preprocessor we can call the `process_simulations` function to preprocess all simulations in the specified directory.

.. code-block:: python

    preprocessor.process_simulations()

**Normalization**

Additionally, we can normalize the data using the `MinMaxNormalizer` or `StandardNormalizer` class.

.. code-block:: python

    from magnet_pinn.utils import MinMaxNormalizer, StandardNormalizer

    normalizer = StandardNormalizer()

A full example can be found in the examples section of the documentation.
