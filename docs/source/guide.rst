===================
User Guide
===================
This guide is an overview of the usage of the magnet-pinn package.
It is intended to help users understand how to load and preporcess simulations of EM-Fields inside a MRI scanner using this package and the published datasets.

.. _install:

----------------------------
:ref:`Installation <start_>`
----------------------------

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

^^^^^^^^^^^^^^^^^^^^^^^
Examples
^^^^^^^^^^^^^^^^^^^^^^^

The examples can be used to fully trace all steps from loading the data to training a model.
The first two examples show how to preprocess the the data, either in grid or point cloud format using python.
The third example shows how to normalize the data, which is an important step before training a model.
The fourth example shows how to train a model using the preprocessed data and including a normalization step.

.. toctree::
    :maxdepth: 1

    Example 1: Preprocessing Grid Data <examples/preprocessing_grid>
    Example 2: Preprocessing Point Cloud Data <examples/preprocessing_point>
    Example 3: Normalization <examples/normalization>
    Example 4: Training a ML model <examples/ml_train>