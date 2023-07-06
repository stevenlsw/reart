<!-- PROJECT LOGO -->
<br />
<p align="center">

  <h1 align="center">Building Rearticulable Models for Arbitrary 3D Objects from 4D Point Clouds </h1>

  <p align="center">
    <img src="assets/method.png" alt="Logo" width="80%">
  </p>
  <p align="center">
    CVPR, 2023
    <br />
    <a href="https://stevenlsw.github.io"><strong>Shaowei Liu</strong></a>
    ·
    <a href="https://saurabhg.web.illinois.edu/"><strong>Saurabh Gupta*</strong></a>
    ·
    <a href="https://shenlong.web.illinois.edu/"><strong>Shenlong Wang*</strong></a>
    ·
  </p>

  <p align="center">
    <a href='https://arxiv.org/abs/2306.00979'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='Paper PDF'>
    </a>
    <a href='https://stevenlsw.github.io/reart/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    <a href='https://colab.research.google.com/drive/16-77HTfuniHwyc9XsvvP_7hyuilNWtjd' style='padding-left: 0.5rem;'><img src='https://colab.research.google.com/assets/colab-badge.svg' alt='Google Colab'></a>
    <a href='https://www.youtube.com/watch?v=Bdolp3FTZUc' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Youtube-Video-red?style=flat&logo=youtube&logoColor=red' alt='Youtube Video'>
    </a>
  </p>
</p>
<br />

This repository contains a pytorch implementation for the paper: [Building Rearticulable Models for Arbitrary 3D Objects from 4D Point Clouds](https://arxiv.org/abs/2306.00979). In this paper, we build animatable 3D models from arbitrary articulated object point cloud sequence.<br><br>

## Overview
![overview](assets/overview.png)


## Installation
- Clone this repository: 
    ```Shell
    git clone https://github.com/stevenlsw/reart
    cd reart
    ```
- Install requirements in a virtual environment:
    ```Shell
    sh setup_env.sh
    ```

The code is tested on Python `3.6.13` and Pytorch `1.10.2+cu113`.

## Colab notebook
Run our [Colab notebook](https://colab.research.google.com/drive/16-77HTfuniHwyc9XsvvP_7hyuilNWtjd) for quick start! 


## Demo

`demo_data` folder contains data and pretrained model of Nao robot. We provide two pretrained models, `base-2` is the relaxation model and `kinematic-2` is the projection model. Postfix `2` is the canonical frame index. Canonical frame index is selected by the lowest energy.

### Evaluate and visualization
Canonical frame index `cano_idx` should be consistent with postfix in pretrained model name.

- projection model
  ```Shell
  python run_robot.py --seq_path=demo_data/data/nao --save_root=exp --cano_idx=2 --evaluate --resume=demo_data/pretrained/nao/kinematic-2/model.pth.tar --model=kinematic
  ```

- relaxation model
  ```Shell
  python run_robot.py --seq_path=demo_data/data/nao --save_root=exp --cano_idx=2 --evaluate --resume=demo_data/pretrained/nao/base-2/model.pth.tar --model=base
  ```

After running the command, results are stored in `${save_root}/${robot name}`. `input.gif` visualize the input sequence, `recon.gif` visualize the reconstruction, 
`gt.gif` visualize the GT. `seg.html` visualize the pred segmentation, `structure.html` visualize the infered topology. `result.txt` contains the evaluation result. 

Input| Recon | GT
---|---|---
![](assets/nao_input.gif) | ![](assets/nao_recon.gif) | ![](assets/nao_gt.gif)



## Data and pretrained model

### Download data
Download the data from [here](https://drive.google.com/drive/folders/1P87A4nAktU8qUJICoAvoQCWp90YFY6ZG?usp=sharing) and save as `data` folder.
```
data
├──  robot
│     └── nao   - robot name
│     └── ...       
├──  category_normalize_scale.pkl  - center and scale of each category
├──  real
│     └── toy   - real scan object
│     └── switch  
```

### Download pretrained model
Download pretrained models from [here](https://drive.google.com/drive/folders/1bu2-zg1KVt7nIBG_L9UVjQTwesZqDoPo?usp=sharing) and save as `pretrained` folder.
```
pretrained
├──  robot
│     └── nao   - robot name
│       └── base-{cano_idx}       - pretrained relaxation model			    
│       └── kinematic-{cano_idx}  - pretrained projection model  
├──  real
├──  corr_model.pth.tar  - pretrained correspondence model
```
## Robot Experiment
Take `nao` as an example.
### Train relaxation model
`corr_model.pth.tar` is needed for training. Recommend set `cano_idx` same as our release pretrained model to get the reported performance for each category. 

```Shell
python run_robot.py --seq_path=data/robot/nao --save_root=exp --cano_idx=2 --use_flow_loss --use_nproc --use_assign_loss --downsample 4 --n_iter=15000
```
The relaxation results are stored at `${save_root}/${robot name}/result.pkl` and needed for training projection model.

### Train projection model

Set the relaxation result `base_result_path` as above.

```Shell
python run_robot.py --seq_path=data/robot/nao --save_root=exp --cano_idx=2  --use_flow_loss --use_nproc --use_assign_loss --model=kinematic --base_result_path=exp/nao/result.pkl --assign_iter=0 --downsample=2 --assign_gap=1 --snapshot_gap=10
```

### Evaluate pretrained model

```Shell
python run_robot.py  --seq_path=data/robot/nao --save_root=exp --cano_idx=2 --evaluate --resume=pretrained/robot/nao/kinematic-2/model.pth.tar --model=kinematic
```

See all robots and pretrained models in `pretrained/robot`, Take `spot` as another example, you could get

Input| Recon | GT
---|---|---
![](assets/spot_input.gif) | ![](assets/spot_recon.gif) | ![](assets/spot_gt.gif)

## Real-world experiment
Follow instructions similar to robot. Take `toy` as an example.

### Inference
```Shell
python run_real.py --seq_path=data/real/toy --evaluate --model=kinematic --save_root=exp --cano_idx=0  --resume=pretrained/real/toy/kinematic-0/model.pth.tar
```
### Train relaxation model
```Shell
python run_real.py --seq_path=data/real/toy --save_root=exp --cano_idx=0 --use_flow_loss --use_nproc --use_assign_loss --assign_iter=1000 
```
### Train projection model
```Shell
python run_real.py --seq_path=data/real/toy --cano_idx=0 --save_root=exp --n_iter=200 --use_flow_loss --use_nproc --use_assign_loss --model=kinematic --assign_iter=0 --assign_gap=1 --snapshot_gap=10 --base_result_path=exp/toy/result.pkl  
```

We provide real-scan `toy` and `switch` from `Polycam` app in iPhone. Take `toy` as an example, you could get

Input| Recon
---|---
![](assets/toy_input.gif) | ![](assets/toy_recon.gif)


## Sapien Experiment

### Setup

- Data

  Follow [multibody-sync](https://github.com/huangjh-pub/multibody-sync.git), download [`mbs_sapien.zip`](https://drive.google.com/file/d/1HR2X0DjgXLwp8K5n2nsvfGTcDMSckX5Z) and unzip it as `mbs_sapien` and put under `data` folder. 

- Model

  We use the [pretrained flow model](https://drive.google.com/file/d/1bomD88-6N1iGsTtftfGvAm9JeOw8gKwb) from [multibody-sync](https://github.com/huangjh-pub/multibody-sync.git) in our method for fair comparison. First clone the repo as `msync`.

  ```Shell
  git clone https://github.com/huangjh-pub/multibody-sync.git msync
  ```

  Follow [multibody-sync](https://github.com/huangjh-pub/multibody-sync.git) instruction, download the [trained weights](https://drive.google.com/file/d/1bomD88-6N1iGsTtftfGvAm9JeOw8gKwb/view?usp=sharing) and extract the weights to `msync/ckpt/articulated-full/best.pth.tar`.

### Train relaxation model

Specify `sapien_idx` to select different sapien objects, all experiments use canonical frame 0 `cano_idx=0`.

```Shell
python run_sapien.py --sapien_idx=212 --save_root=exp --n_iter=2000 --cano_idx=0 --use_flow_loss --use_nproc --use_assign_loss
```

The relaxation results are stored at `${save_root}/sapien_{sapien_idx}/result.pkl` and needed for training projection model.

### Train projection model

Set the relaxation result `base_result_path` as above.

```Shell
python run_sapien.py --sapien_idx=212 --save_root=exp --n_iter=200 --cano_idx=0 --model=kinematic --use_flow_loss --use_nproc --use_assign_loss  --assign_iter=0 --assign_gap=1 --snapshot_gap=10 --base_result_path=exp/sapien_212/result.pkl
```

After training, results are stored in `${save_root}/sapien_{sapien_idx}/`. `result.txt` contains the evaluation result. 

Take `sapien_idx=212`as an example, you could get

Input| Recon | GT
---|---|---
![](assets/sapien_212_input.gif) | ![](assets/sapien_212_recon.gif) | ![](assets/sapien_212_gt.gif)


## Citation

If you find our work useful in your research, please cite:

```BiBTeX
@inproceedings{liu2023building,
  title={Building Rearticulable Models for Arbitrary 3D Objects from 4D Point Clouds},
  author={Liu, Shaowei and Gupta, Saurabh and Wang, Shenlong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21138--21147},
  year={2023}
}
```


## Acknowledges
We thank:
* [Watch It Move](https://github.com/NVlabs/watch-it-move.git) for MST implementation
* [multibody-sync](https://github.com/huangjh-pub/multibody-sync.git) for Sapien dataset setup
* [APTED](https://github.com/JoaoFelipe/apted) for tree edit distance measure
* [KNN_CUDA](https://github.com/unlimblue/KNN_CUDA.git) for KNN with CUDA support
