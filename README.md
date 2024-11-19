# magnet
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