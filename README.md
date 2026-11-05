# Introduction
Metal-SigGTR is a framework for identifying protein metal-ion binding sites that utilizes deep sequence features (ESM-2 refined by GatedCNN) and geometrically invariant structural features (pairwise distance matrices derived directly from ESMFold internal tensors). Metal-SigGTR is fast and accurate, achieving AUC above 0.89 across seven metal-ion tasks, with six out of seven exceeding 0.90. The model is validated on multiple independent benchmarks with strong zero-shot generalization.

# System requirement
python  3.7.11  
numpy  1.21.2  
pandas  1.1.3  
torch  1.8.0  
biopython  1.79  
sentencepiece 0.1.96  
transformers 4.15.0
fair-esm 2.0.0

# Install and set up Metal-SigGTR
**1.** Download the [ESMFold](https://github.com/facebookresearch/esm) model and install according to the official tutorials：
```
pip install "fair-esm[esmfold]"
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'

Extract distance matrices and 14D biophysical features via python esm.py 
```  

**2.** Download the ESMFold (esmfold_3B_v1.pt) model weights from https://dl.fbaipublicfiles.com/fair-esm/models/esmfold_3B_v1.pt  and the ESM-2 3B (esm2_t36_3B_UR50D.pt) model weights from https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt , and place them in both directories: ~/.cache/torch/hub/checkpoints/ and esm-main/esm-main/esmfold_weights/checkpoints/. Extracting language model features:
```
python ./esmf.py 
python ./process_Esm.py
```

**3.** Merge features:
```
python ./pad_feature.py 
```
# Run Metal-SigGTR for prediction
Run the following command to predict the binding sites of the sequence in "./datasets/test.fa" on GPU (id=0):
```
CUDA_VISIBLE_DEVICES=0 python ./main.py --task CA --test --seed 2022 --run_id prediction --dataset_path ./datasets/ --feature_path ./feature/
```
# Datasets and models
The datasets used in this study are stored in `./datasets/`  
The associated dataset (DOI/CSTR: ) for this paper is available in the ScienceDB database at https://www.scidb.cn/s/mAJnEz.


