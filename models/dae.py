# a diffusion autoencoder
# --------------------------------------------------------

import torch
import torch.nn as nn

from .autoencoder import DualAutoEncoder
from .unet import Linear_UNet


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()


class DAE(nn.Module):
    """
    diffusion autoencoder
    """
    def __init__(
        self,
        data_dim,
        input_dim,
        emb_size=64,
        hidden_dims=[512, 256, 128, 64]
    ):
        super().__init__()

        self.encoder = DualAutoEncoder(input_dim=input_dim, output_dim=data_dim, emb_size=emb_size)
        self.unet_x = Linear_UNet(input_dim=input_dim, cond_dim=emb_size*2, hidden_dims=hidden_dims)
        self.unet_y = Linear_UNet(input_dim=input_dim, cond_dim=emb_size*2, hidden_dims=hidden_dims)

        # initialize_weights(self)

    def forward(self, x, y, e, ew):
        return self.encoder(x, y, e, ew)

