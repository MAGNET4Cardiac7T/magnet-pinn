:html_theme.sidebar_secondary.remove: true

.. title:: MAGNET4Cardiac7T

.. image:: images/logo_magnet.svg
     :width: 1800px
     :align: center

.. toctree::
    :maxdepth: 2
    :caption: About
    :hidden:

    Home <self>
    Gallery <gallery>
    Getting Started <start>
    Benchmark Results <results>

.. toctree::
    :maxdepth: 2
    :caption: Docs
    :hidden:

    Tutorial <guide>
    Examples <examples>
    API reference <api>

.. raw:: html

    <div style="margin-bottom: 40px;"></div>

-------------------------------------------------------------------------------------------------------------------------
Patient-specific modeling of electromagnetic fields in ultrahigh-field cardiac MRI using physics-informed neural networks
-------------------------------------------------------------------------------------------------------------------------

**Date**: |today|


.. image:: https://img.shields.io/pypi/v/magnet-pinn.svg
    :target: https://pypi.org/project/magnet-pinn/
    :alt: PyPI

.. image:: https://img.shields.io/pypi/pyversions/magnet-pinn.svg
    :target: https://pypi.org/project/magnet-pinn/
    :alt: Python versions

.. image:: https://github.com/MAGNET4Cardiac7T/magnet-pinn/actions/workflows/test_all.yaml/badge.svg
    :target: https://github.com/MAGNET4Cardiac7T/magnet-pinn/actions/workflows/test_all.yaml
    :alt: All Tests

.. image:: https://img.shields.io/pypi/l/magnet-pinn.svg
    :target: https://pypi.org/project/magnet-pinn/
    :alt: License

.. image:: https://img.shields.io/pypi/dm/magnet-pinn.svg
    :target: https://pypi.org/project/magnet-pinn/
    :alt: Downloads


**Useful links**:
:doc:`Install <start>` |
`Source Repository <https://github.com/MAGNET4Cardiac7T/magnet-pinn>`__ |
`Dataset <https://github.com/MAGNET4Cardiac7T/magnet-pinn>`__


----------------------
Package Overview
----------------------
The **magnet-pinn** package is an open-source package for developing, training and evaluating deep learning models for predicting EM-Fields inside an MRI scanner.
It is built on top of PyTorch and provides a simple and flexible API for preprocessing data, generating new 3D geometries, and training and evaluating ML models.
The dataset includes simulated EM fields for a number of geometries that can be used to train Neural Networks.
The package contains easy to use functions and scripts to process the data and fit them to your individual needs.
Additionally, we supplement examples on how to train a simple UNet model to predict the EM fields inside an MRI scanner.

Important features of the **magnet-pinn** package include:

- Easy-to-use data loading and preprocessing functions for the provided dataset
- Tools for generating new 3D geometries
- predefined ML models
- Evaluation metrics and visualization tools for analyzing model performance
- Comprehensive documentation and examples to help users get started quickly

Check out all :doc:`newly added features <features>`.

----------------------
Project Overview
----------------------
Heart failure is a common disease with high mortality and is one of the most common causes of death.
Magnetic resonance imaging of the heart is an important diagnostic technique for functional diagnosis of heart failure and many other cardiac diseases.
Cardiac ultrahigh-field magnetic resonance imaging (UHF MRI) at a field strength of 7 Tesla promises the highest physical sensitivity, the highest spatial resolution and completely new image contrasts.
However, a major obstacle to widespread application is the complex distribution of of electromagnetic waves in the patient's thorax, which has a very negative impact on image quality and also poses the risk of unwanted overheating of the tissue.

Within the scope of this project, a method for patient-specific analysis of the three-dimensional distribution of the electromagnetic fields in the body will be developed.
So far, the field distribution must be calculated by simulating Maxwell's equations with special electromagnetic field simulators.
This process takes several days and can therefore not be used in clinical routine on individual patients due to time time constraints.
For this reason, Deep Learning (DL) methods will be used, adapted and further developed.
In particular, we want to employ physics-informed neural networks and use physical constraints to compensate for limited amounts of training data.

Within the project, we use a multi-step approach:
We first test different DL methods on simple 3D geometries, such as a sphere in an EM field, for which data can be generated with low computational effort from Maxwell simulations.
We use the pre-trained DL models then to improve them with increasingly complex 3D models until the target structure, the human thorax, can be modeled with sufficient complexity.


----------------------
Publications
----------------------
When you use our package please cite us using the following publications:

- ...

----------------------
Partners
----------------------

This project is a cooperation with the following partners:

- `Data Science Chair (X), University of Würzburg <https://dmir.org/>`_
- `Deutsches Zentrum für Herzinsufizienz (DZHI), Universitätsklinikum Würzburg <https://www.ukw.de/behandlungszentren/dzhi/startseite/>`_
