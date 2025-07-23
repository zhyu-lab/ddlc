import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import TAGConv


#  Position-wise Feed-Forward Networks
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model):
        super(PoswiseFeedForwardNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(x + self.mlp(x))


# Scaled Dot-Product Attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        d = Q.size(2)
        scores = torch.matmul(Q, K.transpose(1, 2)) / np.sqrt(d)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context


# Multi-Head Cross Attention Transformer Block
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.W_Q1 = nn.Linear(d_model, d_model, bias=False)
        self.W_K1 = nn.Linear(d_model, d_model, bias=False)
        self.W_V1 = nn.Linear(d_model, d_model, bias=False)
        self.W_Q2 = nn.Linear(d_model, d_model, bias=False)
        self.W_K2 = nn.Linear(d_model, d_model, bias=False)
        self.W_V2 = nn.Linear(d_model, d_model, bias=False)
        self.fc1 = nn.Linear(d_model, d_model, bias=False)
        self.fc2 = nn.Linear(d_model, d_model, bias=False)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn1 = PoswiseFeedForwardNet(d_model)
        self.ffn2 = PoswiseFeedForwardNet(d_model)

    def forward(self, input1, input2):
        Q1 = self.W_Q1(input1).view(input1.size(0), self.n_heads, -1).transpose(0, 1)
        K1 = self.W_K1(input1).view(input1.size(0), self.n_heads, -1).transpose(0, 1)
        V1 = self.W_V1(input1).view(input1.size(0), self.n_heads, -1).transpose(0, 1)

        Q2 = self.W_Q2(input2).view(input2.size(0), self.n_heads, -1).transpose(0, 1)
        K2 = self.W_K2(input2).view(input2.size(0), self.n_heads, -1).transpose(0, 1)
        V2 = self.W_V2(input2).view(input2.size(0), self.n_heads, -1).transpose(0, 1)

        context1 = ScaledDotProductAttention()(Q1, K2, V2)
        output1 = context1.transpose(0, 1).reshape(input1.size(0), -1)
        output1 = self.norm1(input1 + self.fc1(output1))

        context2 = ScaledDotProductAttention()(Q2, K1, V1)
        output2 = context2.transpose(0, 1).reshape(input2.size(0), -1)
        output2 = self.norm2(input2 + self.fc2(output2))

        return output1, output2


class CAAFF(nn.Module):
    def __init__(self, input_dim=128, n_heads=8, num_layers=1):
        super(CAAFF, self).__init__()

        self.attention_layers = nn.ModuleList([MultiHeadCrossAttention(input_dim, n_heads) for _ in range(num_layers)])
        self.fc = nn.Linear(input_dim, input_dim)
        self.weights = nn.Linear(input_dim * 2, 1)

    def forward(self, x, y):
        for layer in self.attention_layers:
            x, y = layer(x, y)
        x = self.fc(x)
        y = self.fc(y)

        tmp = torch.cat([x, y], dim=1)
        w = torch.sigmoid(self.weights(tmp))
        z = x + w * y

        return z


class DualAutoEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, emb_size=128):
        super(DualAutoEncoder, self).__init__()

        self.rna_encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, emb_size),
        )
        self.atac_encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, emb_size),
        )
        self.common_encoder = TAGConv(input_dim, emb_size, K=3)
        self.rna_decoder = nn.Sequential(
            nn.Linear(emb_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_dim)
        )
        self.atac_decoder = nn.Sequential(
            nn.Linear(emb_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_dim)
        )
        self.caaff = CAAFF(input_dim=emb_size)

    def forward(self, x1, x2, e, ew):
        # omics-specific information
        z_x_s = self.rna_encoder(x1)
        z_y_s = self.atac_encoder(x2)

        # cross-omics information
        z_y_x = self.rna_encoder(x2)
        z_x_y = self.atac_encoder(x1)

        # common information
        z_x_c = self.common_encoder(x1, e, ew)
        z_y_c = self.common_encoder(x2, e, ew)

        z_c = self.caaff(z_x_c, z_y_c)

        x_hat = self.rna_decoder(z_c)
        y_hat = self.atac_decoder(z_c)

        return z_x_s, z_y_s, z_c, z_y_x, z_x_y, x_hat, y_hat
