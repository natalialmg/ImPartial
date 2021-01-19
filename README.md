# ImPartial:
#### Partial Annotations for MultiplexCell Instance Segmentation

--- Required packages are :

* Tensorflow

* CSBDeep: https://github.com/CSBDeep/CSBDeep/

* Noise2Void: https://github.com/juglab/n2v


This github adapts code from Noise2Void and CARE frameworks: https://csbdeep.bioimagecomputing.com/tools/


--- Preprocessing and training examples:

notebooks/MIBI_1CH/ (MIBI modality 1 channel nuclei)

notebooks/MIBI_2CH/ (MIBI modality 2 channels cytoplasm + nuclei)

--- Datasets :

data/MIBI_1Channel/ (MIBI modality 1 channel nuclei)

data/MIBI_2Channel/ (MIBI modality 2 channels cytoplasm + nuclei)

Each normalized image and corresponding labels are stored in an .npz file.
Each dataset folder contains a file.csv with the specifications, more details in the preprocessing notebook examples.
