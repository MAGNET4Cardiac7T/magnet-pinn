:html_theme.sidebar_secondary.remove: true

*************************
magnet-pinn documentation
*************************

**Date**: |today| **Version**: |release|


**Useful links**:
`Install <start>`__ |
`Source Repository <https://github.com/MAGNET4Cardiac7T/magnet-pinn>`__ |
`Dataset <https://github.com/MAGNET4Cardiac7T/magnet-pinn>`__ |

**magnet-pinn** is an open-source package for developing, training and evaluating deep learning models for predicting EM-Fields inside a MRI scanner. 
It is built on top of PyTorch and provides a simple and flexible API for training and evaluating models. The package includes a dataset class for loading simulated EM-FIleds, a data iterator for training models, and a set of pre-trained models.

.. grid:: 1 1 2 2
    :gutter: 2 3 4 4

    .. grid-item-card::
        :img-top: _static/book-solid.svg
        :text-align: center

        **Getting Started**
        ^^^

        The Getting Started guide will help you get started with magnet-pinn. 
        It shows how to download the data use our data iterator.

        +++

        .. button-ref:: start
            :color: secondary
            :click-parent:

            To the guide

    .. grid-item-card::
        :img-top: _static/wrench-solid.svg
        :text-align: center

        **API reference**
        ^^^

        The reference guide contains a description of the magnet-pinn package. 

        +++

        .. button-ref:: api
            :color: secondary
            :click-parent:

            To the API reference

    .. grid-item-card::
        :img-top: _static/chart-line-solid.svg
        :text-align: center

        **Benchmark results**
        ^^^

        The benchmark results show the performance of different models for the task of predicting EM Fields. 

        +++

        .. button-ref:: results
            :color: secondary
            :click-parent:

            To the results

    .. grid-item-card::
        :img-top: _static/file-lines-solid.svg
        :text-align: center

        **Paper List**
        ^^^

        This is a list of papers related to the MAGNET4Cardiac7T Project.

        +++

        .. button-ref:: papers
            :color: secondary
            :click-parent:

            To the paper


.. toctree::
    :maxdepth: 1
    :hidden:

    About <about>
    Getting Started <start>
    User Guide <guide>
    Paper List <papers>
    Benchmark Results <results>
    API reference <api>