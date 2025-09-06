.. _start_:

===================
Getting Started
===================

This section guides you on how to start using the magnet-pinn package.

^^^^^^^^^^^^^^^^^^^^
Install using pip
^^^^^^^^^^^^^^^^^^^^

The easiest way to install DynaBench is to use pip:

.. code-block::

    pip install magnet_pinn

Also when using pip, it’s good practice to use a virtual environment - see `this guide <https://dev.to/bowmanjd/python-tools-for-managing-virtual-environments-3bko#howto>`_ for details on using virtual environments.


.. _download:

^^^^^^^^^^^^^^^^^^^^
Downloading Data
^^^^^^^^^^^^^^^^^^^^
Download the datset using the following command:

.. code-block:: shell
    
    python ...

The dataset consists of multiple simulations which are tagged by their construction.
For example there is a simulation that containes two children and 4 tubes and is therefore tagged "children_2_tubes_4_id_1556".
A single simulation item contains the E and B-fields that are both nested under fields and the coils nested under the tag coils.
Moreover, both fields have a real and imaginary part and span all three spatial dimensions with 100 x 100 x 100 points.
Additionally there exists a mask for the subject, the phases of the coils and another mask for the coils.

Examplary slices of the absolute E and B-field:

.. list-table::
    :widths: 50 50
    :align: center

    * - .. image:: images/slices_e2.png
            :width: 900px
            :align: center
      - .. image:: images/slices_b2.png
            :width: 900px
            :align: center