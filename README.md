
<p align="center">
<h1 align="center"><strong> LGSDF: Signed Distance Fields Continual Global Learning Aided by Local Updating</strong></h1>
</p>



<p align="center">
  <a href="https://lgsdf.github.io/" target='_blank'>
    <img src="https://img.shields.io/badge/Project-👔-green?">
  </a> 
  
  <a href="https://arxiv.org/pdf/2404.05187" target='_blank'>
    <img src="https://img.shields.io/badge/Paper-📖-blue?">
  </a> 

</p>


 ## 🏠  Abstract
 
Implicit reconstruction of ESDF (Euclidean Signed Distance Field) involves training a neural network to regress the signed distance from any point to the nearest obstacle, which has the advantages of lightweight storage and continuous querying. However, existing algorithms usually rely on conflicting raw observations as training data, resulting in poor map performance. In this paper, we propose LGSDF, an ESDF continual Global learning algorithm aided by Local updating. 
In the front-end, anchors are uniformly distributed throughout the scene and incrementally updated based on preprocessed sensor observations, reducing estimation errors caused by limited viewing directions. In the back-end, a randomly initialized implicit ESDF neural network undergoes continuous self-supervised learning, driven by strategically sampled anchors, to produce smooth and continuous maps.
The results on multiple scenes show that LGSDF can construct more accurate SDFs and meshes compared with SOTA ESDF mapping algorithms.
<img src="https://github.com/BIT-DYN/LGSDF/blob/main/figs/framework.png" >

##  🛠 Install
```bash
git clone https://github.com/BIT-DYN/LGSDF
conda env create -f environment.yml
conda activate lgsdf
```

## 📊 ownload Data

```bash
bash data/download_data.sh
```

## 🏃 Run

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
## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@article{lgsdf,
  title={LGSDF: Continual Global Learning of Signed Distance Fields Aided by Local Updating},
  author={Yue, Yufeng and Deng, Yinan and Tang, Yujie and Wang, Jiahui and Yang, Yi},
  journal={arXiv preprint arXiv:2404.05187},
  year={2024}
}
```

## 👏 Acknowledgements
We would like to express our gratitude to the open-source projects and their contributors [iSDF]([https://github.com/kxhit/vMAP](https://github.com/facebookresearch/iSDF)). 
Their valuable work has greatly contributed to the development of our codebase.
