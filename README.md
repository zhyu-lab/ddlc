# DDLC

A diffusion-based disentanglement learning framework for clustering single-cell multi-omics data

## Requirements

* Python 3.9+.

# Installation

## Clone repository

First, download DDLC from github and change to the directory:

```bash
git clone https://github.com/zhyu-lab/ddlc
cd ddlc
```

## Create conda environment (optional)

Create a new environment named "sctca":

```bash
conda create --name ddlc python=3.9
```

Then activate it:

```bash
conda activate ddlc
```

## Install requirements

```bash
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install -r requirements.txt
```

# Usage

## Step 1: Prepare the input data in h5ad format.

We use RNA and ATAC (or ADT) data stored in .h5ad files as input.

## Step 2: Run DDLC

The “train.py” Python script is used to train the model and obtain the clustering results.

The arguments to run “train.py” are as follows:

| Parameter      | Description                                                   | Possible values                   |
| -------------  | ------------------------------------------------------------- | --------------------------------- |
| --rna_path     | input file containing RNA data                                | Ex: /path/to/RNA.h5ad             |
| --atac_path    | input file containing ATAC(ADT)data                           | Ex: /path/to/ATAC(ADT).h5ad       |
| --result_dir   | a directory to save results                                   | Ex: /path/to/results              |
| --train_epochs | number of epoches to train the DDLC                           | Ex: epochs=300  default:500       |
| --neighbors    | the k nearest neighbors of each cell                          | Ex: neighbors=20  default:10      |
| --lr           | learning rate                                                 | Ex: lr=1e-5  default:1e-3         |
| --lambda1      | weight for diffusion-based reconstruction loss                | Ex: lambda1=1  default:1          |
| --lambda2      | weight for semantic alignment loss                            | Ex: lambda2=1  default:0.05       |
| --lambda3      | weight for modality disentanglement loss                      | Ex: lambda3=1  default:1          |
| --seed         | random seed (for reproduction of the results)                 | Ex: seed=1  default:0             |

Example:

```bash
python train.py --rna_path ./datasets/10X_PBMC/10x-Multiome-Pbmc10k-RNA.h5ad  --atac_path ./datasets/10X_PBMC/10x-Multiome-Pbmc10k-ATAC.h5ad  --train_epochs 500  --lr 1e-3  --seed 0  --result_dir ./results/10X_PBMC
```

# Contact

If you have any questions, please contact lfr_nxu@163.com.