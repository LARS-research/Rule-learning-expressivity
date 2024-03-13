# Understanding Expressivity of Neural KG Reasoning in Rule Structure Learning #

## Synthetic experiments ##

Install the dependencies with conda with the following commands.

```bash
conda create -n el-gnn python=3.10
conda activate el-gnn
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install ray[all]==2.9.3
pip install tqdm
conda install ninja
conda install nvidia/label/cuda-11.7.0::cuda
```

To test EL-GNN, run the following commands.

```bash
cd synthetic

# C3
python main.py --dataset data/C3

# C4
python main.py --dataset data/C4

# I1
python main.py --dataset data/I1

# I2
python main.py --dataset data/I2 --dropout 0.2 --weight_decay 0.0001 --lr 0.01

# T
python main.py --dataset data/T --weight_decay 0.001 --act relu --lr 0.01 --sum max

# U
python main.py --dataset data/U
```

Hyperparameter optimization of EL-GNN can be conducted with `synthetic/ray_hpo.py`.


## Real experiments ##

This code is re-implemented from [NBFNet](https://github.com/DeepGraphLearning/NBFNet).


The code works with Python 3.7/3.8 and PyTorch version >= 1.8.0. Install the dependencies with conda with the following commands.

```bash
conda create -n el-gnn python=3.10
conda install torchdrug -c milagraph -c conda-forge -c pytorch -c pyg
conda install ninja
conda install ogb easydict pyyaml -c conda-forge
```

Run the following commands to test EL-GNN on real datasets.

```bash
python script/run.py -c config/family.yaml --gpus [0]
python script/run.py -c config/fb15k237.yaml --gpus [0]
python script/run.py -c config/kinship.yaml --gpus [0]
python script/run.py -c config/umls.yaml --gpus [0]
python script/run.py -c config/wn18rr.yaml --gpus [0]
```

`degree` in `config/*.yaml` is the hyperparameter denoting the degree threshold for EL-GNN.


Citation
--------

If you find our code useful in your research, please cite the following paper.

```bibtex
@inproceedings{
qiu2024understanding,
title={Understanding Expressivity of Neural KG Reasoning from Rule Structure Learning},
author={Haiquan Qiu, Yongqi Zhang, Yong Li and Quanming Yao},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=43cYe4oogi}
}
```
