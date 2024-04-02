# LGSDF
LGSDF: Signed Distance Fields Continual Global Learning Aided by Local Updating
 ## Abstract
 
Implicit reconstruction of ESDF (Euclidean Signed Distance Field) involves training a neural network to regress the signed distance from any point to the nearest obstacle,} which has the advantages of lightweight storage and continuous querying. However, existing algorithms usually rely on conflicting raw observations as training data, resulting in poor map performance. In this paper, we propose **LGSDF**, an algorithm for Local updating and Global learning of Signed Distance Fields for robot mapping. At the front end, axis-aligned grids are dynamically updated by pre-processed sensor observations, where incremental fusion alleviates estimation error caused by limited viewing directions. At the back end, a randomly initialized implicit ESDF neural network performs continual self-supervised learning guided by these grids to generate smooth and continuous maps. The results on multiple scenes show that LGSDF can construct more accurate ESDF maps and meshes compared with SOTA explicit and implicit mapping algorithms. 
 
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
