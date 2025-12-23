"""Utility functions for UNet3D model configuration and dynamic class loading."""

import importlib


def get_class(class_name, modules):
    """
    Dynamically load a class from a list of module paths.

    Parameters
    ----------
    class_name : str
        Name of the class to load
    modules : list of str
        List of module paths to search for the class

    Returns
    -------
    type
        The requested class object

    Raises
    ------
    RuntimeError
        If the class cannot be found in any of the provided modules
    """
    for module in modules:
        m = importlib.import_module(module)
        clazz = getattr(m, class_name, None)
        if clazz is not None:
            return clazz
    raise RuntimeError(f"Unsupported dataset class: {class_name}")


def number_of_features_per_level(init_channel_number, num_levels):
    """
    Calculate the number of feature maps at each level of the UNet encoder.

    Parameters
    ----------
    init_channel_number : int
        Initial number of feature channels
    num_levels : int
        Number of levels in the encoder path

    Returns
    -------
    list of int
        List containing the number of feature maps at each level,
        computed as init_channel_number * 2^k for k in range(num_levels)
    """
    return [init_channel_number * 2**k for k in range(num_levels)]
