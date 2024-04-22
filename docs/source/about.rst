==================================
About the MAGNET4Cardiac7T project
==================================

.. image:: images/logo_magnet.svg
     :width: 1800px
     :align: center

----------------------
Motivation
----------------------

Heart failure is a common disease with high mortality and is one of the most common causes of death. Magnetic resonance imaging of the heart is an important diagnostic technique for functional diagnosis of heart failure and many other cardiac diseases. heart diseases. Cardiac ultrahigh-field magnetic resonance imaging (UHF MRI) at a field strength of 7 Tesla promises the highest physical sensitivity, the highest spatial resolution and completely new image contrasts. However, a major obstacle to widespread application is the complex distribution of of electromagnetic waves in the patient's thorax, which has a very negative impact on image quality and also poses the risk of unwanted overheating of the tissue.

----------------------
Goal of the Project
----------------------

Within the scope of this project, a method for patient-specific analysis of the three-dimensional distribution of the electromagnetic fields in the body will be developed. So far, the field distribution must be calculated by simulating Maxwell's equations with special electromagnetic field simulators. This process takes several days and can therefore not be used in clinical routine on individual patients due to time time constraints. For this reason, Deep Learning (DL) methods will be used, adapted and further developed. In particular, we want to employ physics-informed neural networks and use physical constraints to compensate for limited amounts of training data.

----------------------
Implementation
----------------------

Within the project, we use a multi-step approach: we first test different DL methods on simple 3D geometries, such as a sphere in an EM field, for which data can be generated with low computational effort from Maxwell simulations. We use the pre-trained DL models then to improve them with increasingly complex 3D models until the target structure, the human thorax, can be modeled with sufficient complexity.
Partners

This project is done in cooperation with:

`Data Science Chair (X), University of Würzburg <https://dmir.org/>`_

`Deutsches Zentrum für Herzinsufizienz (DZHI), Universitätsklinikum Würzburg <https://www.ukw.de/behandlungszentren/dzhi/startseite/>`_


