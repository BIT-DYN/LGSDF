# LGSDF
LGSDF: Signed Distance Fields Continual Global Learning Aided by Local Updating
 ## Abstract
 
Implicit reconstruction of ESDF (Euclidean Signed Distance Field) involves training a neural network to regress the signed distance from any point to the nearest obstacle, which has the advantages of lightweight storage and continuous querying. However, existing algorithms usually rely on conflicting raw observations as training data, resulting in poor map performance. In this paper, we propose LGSDF, an ESDF continual Global learning algorithm aided by Local updating. 
In the front-end, anchors are uniformly distributed throughout the scene and incrementally updated based on preprocessed sensor observations, reducing estimation errors caused by limited viewing directions. In the back-end, a randomly initialized implicit ESDF neural network undergoes continuous self-supervised learning, driven by strategically sampled anchors, to produce smooth and continuous maps.
The results on multiple scenes show that LGSDF can construct more accurate SDFs and meshes compared with SOTA ESDF mapping algorithms.
 
<img src="https://github.com/BIT-DYN/LGSDF/blob/main/figs/framework.png" >

## Install
```bash
git clone https://github.com/BIT-DYN/LGSDF
conda env create -f environment.yml
conda activate lgsdf
```

## Download Data

```bash
bash data/download_data.sh
```

## Run

### ReplicaCad
```bash
cd lgsdf/train/
python train.py --config configs/replicaCAD.json
```
### Scannet
```bash
cd lgsdf/train/
python train.py --config configs/scannet.json
```

## Result
<img src="https://github.com/BIT-DYN/LGSDF/blob/main/figs/com.png"  width="50%">
