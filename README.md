# ImPartial

--- Code for :

## Partial Annotations for MultiplexCell Instance SegmentationNatalia Martinez1*, Guillermo Sapiro1, Allen Tannenbaum2,Travis Hollmann3, and Saad Nadeem


--- Required packages are :

Tensorflow

CSBDeep:
https://github.com/CSBDeep/CSBDeep/

Noise2Void:
https://github.com/juglab/n2v


This github adapts code from Noise2Void and CARE frameworks: https://csbdeep.bioimagecomputing.com/tools/


--- Preprocessing and training examples in :

notebooks/MIBI_1CH/ (MIBI modality 1 channel nuclei)

notebooks/MIBI_2CH/ (MIBI modality 2 channels cytoplasm + nuclei)

--- Data examples in :

data/MIBI_1Channel/ (MIBI modality 1 channel nuclei)

data/MIBI_2Channel/ (MIBI modality 2 channels cytoplasm + nuclei)

Each preprocessed image and corresponding labels are stored in a .npz file.
Each dataset folder contains a file.csv with the specifications, more details in the preprocessing notebook examples.
