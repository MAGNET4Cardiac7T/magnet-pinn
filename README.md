# MAGNET-PINN
<img src="https://magnet4cardiac7t.github.io/assets/img/magnet_logo_venn.svg" width="400em" align="right" />

[![PyPI version](https://badge.fury.io/py/magnet_pinn.svg)](https://badge.fury.io/py/magnet_pinn)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![All Tests](https://github.com/MAGNET4Cardiac7T/magnet-pinn/actions/workflows/test_all.yaml/badge.svg)](https://github.com/MAGNET4Cardiac7T/magnet-pinn/actions/workflows/test_all.yaml)

[comment]: [![Docs](https://github.com/badulion/dynabench/actions/workflows/build_docs.yml/badge.svg)](https://dynabench.github.io)

This is the software package for simulating EM Fields using NNs

## preprocessing
This module is used to preprocess the data. 

### data
The simulation data needs to be placed in the data folder under `data/raw/GROUP_NAME/simulations`.
E.g. `data/raw/batch_1/simulations/children_0_tubes_0_id_3114`, `data/raw/batch_1/simulations/children_0_tubes_1_id_3382`, ...

Additionally, the antennae data needs to be placed in the data folder under `data/raw/GROUP_NAME/antenna`, i.e.:
`data/raw/batch_1/antenna/Dipole_1.stl`, `data/raw/batch_1/antenna/Dipole_2.stl`, ..., `data/raw/batch_1/antenna/materials.txt`

### usage
An example is given in examples/preprocessing.ipynb
