# LGSDF
LGSDF: Signed Distance Fields Continual Global Learning Aided by Local Updating
 ## Abstract
 Fast incremental reconstruction of Euclidean Signed Distance Field (ESDF) maps is useful for truly autonomous navigating robots exploring unfamiliar environments, where signed distances and gradients of query points to the nearest obstacles are readily available. The current methods of generating ESDF maps can be roughly divided into two strategies. Explicit mapping updates distance locally in discrete voxel grids, resulting in resolution-scale detailed and non-compact reconstructions. Implicit mapping exploits a multilayer perceptron (MLP) to regress the global SDF, but often suffers from training data conflicts and catastrophic forgetting. This paper proposes LGSDF, a novel ESDF map reconstruction algorithm to address the above challenges, which can be divided into front and back ends. On the front end, axis-aligned grids are dynamically updated by pre-processed sensor observations, where context fusion alleviates estimation error caused by limited viewing directions and improves local accuracy. On the back end, a randomly initialized implicit ESDF neural network performs continuous self-supervised learning under the guidance of these grids to generate smooth and continuous maps. The verification results on multiple scenes show that LGSDF can construct more accurate ESDF and mesh compared with state-of-the-art both explicit and implicit mapping algorithms.
 
<img src="https://github.com/BIT-DYN/LGSDF/blob/main/figs/framework.png" >

## Install
```bash
it clone https://github.com/BIT-DYN/LGSDF
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
<img src="https://github.com/BIT-DYN/LGSDF/blob/main/figs/com.png" >
