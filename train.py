import os
import argparse
import numpy as np
import random
import torch
from torch.optim import Adam
import warnings

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.cluster import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    homogeneity_score
)

import scanpy as sc
import anndata as ad

from diffusion import logger
from diffusion.resample import create_named_schedule_sampler
from utils.script_util import (
    diffusion_defaults,
    create_dae,
    create_diffusion,
    args_to_dict,
    add_dict_to_argparser
)
from utils.training import train_dae
from utils.graph_function import create_edge_index

from utils.loader import (
    load_RNA_data,
    load_ATAC_data,
    normalize_rna,
    normalize_atac
)
from utils.clustering import leiden_clustering


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    warnings.filterwarnings("ignore")
    parser = create_argparser()
    args = parser.parse_args()
    setup_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.set_device(args.gpu)

    result_dir = args.result_dir
    os.makedirs(result_dir, exist_ok=True)
    logger.configure(dir=result_dir)

    logger.log("loading dataset...")

    rna_data = load_RNA_data(args.rna_path)
    atac_data = load_ATAC_data(args.atac_path)
    logger.log("preprocess RNA and ATAC data...")
    rna_data = normalize_rna(rna_data,
                             size_factors=True,
                             normalize_input=True,
                             logtrans_input=True,
                             use_hvg=True,
                             n_top_genes=args.n_hvgs,
                             n_comps=args.n_comps
                             )
    atac_data = normalize_atac(atac_data,
                               size_factors=True,
                               normalize_input=True,
                               logtrans_input=True,
                               use_hvg=True,
                               n_top_genes=args.n_hvgs,
                               n_comps=args.n_comps
                               )

    labels_t = LabelEncoder().fit_transform(rna_data.obs['cell_type'])

    rna_train = torch.tensor(rna_data.obsm['X_pca'], dtype=torch.float32).to(device)
    atac_train = torch.tensor(atac_data.obsm['X_pca'], dtype=torch.float32).to(device)
    rna_hvg = torch.tensor(rna_data.obsm['hvg_data'], dtype=torch.float32).to(device)
    atac_hvg = torch.tensor(atac_data.obsm['hvg_data'], dtype=torch.float32).to(device)

    e_r, A_r = create_edge_index(rna_data.obsm['X_pca'], args.neighbors)
    e_a, A_a = create_edge_index(atac_data.obsm['X_pca'], args.neighbors)

    A = A_r + A_a
    index = np.nonzero(A)
    e_index = np.concatenate((np.expand_dims(index[0], axis=0), np.expand_dims(index[1], axis=0)), axis=0)
    e_weights = np.float32(A[index])
    e = torch.from_numpy(e_index).to(device)
    e_weights = torch.from_numpy(e_weights).to(device)

    dae = create_dae(args.n_hvgs, args.n_comps, args.emb_size, args.hidden_dims)
    diffusion_x = create_diffusion(
        **args_to_dict(args, diffusion_defaults().keys())
    )
    diffusion_y = create_diffusion(
        **args_to_dict(args, diffusion_defaults().keys())
    )
    dae.to(device)

    optimizer_dae = Adam(dae.parameters(), lr=args.lr)
    schedule_sampler_x = create_named_schedule_sampler(args.schedule_sampler, diffusion_x)
    schedule_sampler_y = create_named_schedule_sampler(args.schedule_sampler, diffusion_y)

    logger.log("training the model...")
    train_dae(dae,
              diffusion_x, diffusion_y,
              rna_train, atac_train,
              rna_hvg, atac_hvg,
              labels_t,
              e, e_weights,
              optimizer_dae,
              args.lr,
              args.lambda1,
              args.lambda2,
              args.lambda3,
              result_dir,
              args.train_epochs,
              schedule_sampler_x,
              schedule_sampler_y
              )

    dae.eval()
    with torch.no_grad():
        z_x_s, z_y_s, z_c, _, _, _, _ = dae(rna_train, atac_train, e, e_weights)

    z = z_c.detach().cpu().numpy()
    predict_labels, best_metrics = leiden_clustering(z, labels_t)

    label_file = result_dir + '/labels.txt'
    file_o = open(label_file, 'w')
    np.savetxt(file_o, np.c_[np.reshape(predict_labels, (1, len(predict_labels)))], fmt='%s', delimiter=',')
    file_o.close()

    latent_file = result_dir + '/embeddings.txt'
    file_o = open(latent_file, 'w')
    np.savetxt(file_o, np.c_[z], fmt='%.3f', delimiter=',')
    file_o.close()

    logger.log("complete.")


def create_argparser():
    defaults = dict(
        # model paras
        emb_size=64,
        hidden_dims=[512, 512, 256, 256],
        schedule_sampler="uniform",
        lr=1e-3,
        train_epochs=500,
        neighbors=10,
        n_hvgs=3000,
        n_comps=256,
        lambda1=1,
        lambda2=0.05,
        lambda3=1,
        gpu=0,
        seed=0,

        # input and output paras
        rna_path='./datasets/10X_PBMC/10x-Multiome-Pbmc10k-RNA.h5ad',
        atac_path='./datasets/10X_PBMC/10x-Multiome-Pbmc10k-ATAC.h5ad',
        result_dir='./results/10X_PBMC',
    )
    defaults.update(diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
