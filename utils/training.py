import sys

import anndata as ad
import numpy as np
import scanpy as sc
import torch
import torch.nn as nn

from .losses import zero_loss, unbalanced_ot, distance_matrix

sys.path.append("..")
from diffusion.resample import UniformSampler
from diffusion import logger


def train_dae(dae,
              diffusion_x, diffusion_y,
              rna_train, atac_train,
              rna_hvg, atac_hvg,
              labels,
              e, ew,
              optimizer,
              lr,
              lambda1,
              lambda2,
              lambda3,
              model_dir,
              train_epoch=100,
              schedule_sampler_x=None,
              schedule_sampler_y=None):
    rna_diff_losses = []
    atac_diff_losses = []
    zero_losses = []
    r_rec_losses = []
    a_rec_losses = []
    train_losses = []

    schedule_sampler_x = schedule_sampler_x or UniformSampler(diffusion_x)
    schedule_sampler_y = schedule_sampler_y or UniformSampler(diffusion_y)

    lr_s = lr

    loss_fn_recon = nn.MSELoss()
    for epoch in range(train_epoch):
        dae.train()
        optimizer.zero_grad()
        z_x_s, z_y_s, z_c, z_y_x, z_x_y, x_hat, y_hat = dae(rna_train, atac_train, e, ew)

        # diffusion loss
        t_x, weights_x = schedule_sampler_x.sample(rna_train.shape[0], rna_train.device)
        model_kwargs = {'y': torch.cat([z_x_s, z_c], dim=1)}
        loss_diffx = diffusion_x.training_losses(dae.unet_x, rna_train, t_x, model_kwargs)
        loss_diffx = (loss_diffx["loss"] * weights_x).mean()

        t_y, weights_y = schedule_sampler_y.sample(atac_train.shape[0], atac_train.device)
        model_kwargs = {'y': torch.cat([z_y_s, z_c], dim=1)}
        loss_diffy = diffusion_y.training_losses(dae.unet_y, atac_train, t_y, model_kwargs)
        loss_diffy = (loss_diffy["loss"] * weights_y).mean()

        loss_diff = loss_diffx + loss_diffy

        # reconstruction loss
        loss_r_rec = loss_fn_recon(x_hat, rna_hvg)
        loss_a_rec = loss_fn_recon(y_hat, atac_hvg)

        loss_rec = loss_r_rec + loss_a_rec

        # zero loss
        loss_zero = zero_loss(z_y_x) + zero_loss(z_x_y)

        loss = lambda1 * loss_diff + lambda2 * loss_rec + lambda3 * loss_zero

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        rna_diff_losses.append(loss_diffx.item())
        atac_diff_losses.append(loss_diffy.item())
        zero_losses.append(loss_zero.item())
        r_rec_losses.append(loss_r_rec.item())
        a_rec_losses.append(loss_a_rec.item())

        frac_done = (epoch + 1) / train_epoch
        lr = lr_s - (lr_s - lr_s / 10) * frac_done
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        logger.log(f'Epoch [{epoch + 1}/{train_epoch}]:')
        logger.log(
            f'Train Loss: {train_losses[-1]:.4f}, RNA D Loss: {rna_diff_losses[-1]:.4f}, ATAC D Loss: {atac_diff_losses[-1]:.4f}, Zero Loss: {zero_losses[-1]:.4f}, R REC Loss: {r_rec_losses[-1]:.4f}, A REC Loss: {a_rec_losses[-1]:.4f}')

    torch.save(dae.state_dict(), model_dir + '/dae.pt')

    logger.log(f'Training model complete.')

    return train_losses
