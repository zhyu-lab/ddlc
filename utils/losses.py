import sys
import numpy as np
import math
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_selection import mutual_info_regression


def ce_loss(masks_prob, true_masks, eps=sys.float_info.epsilon):
    loss = true_masks * torch.log(masks_prob + eps) + (1 - true_masks) * torch.log(1 - masks_prob + eps)
    loss = -torch.mean(loss)
    return loss


# joint probability
def compute_joint(x, y):
    """Compute the joint probability matrix P"""

    bn, k = x.size()
    assert (y.size(0) == bn and y.size(1) == k)

    p_i_j = x.unsqueeze(2) * y.unsqueeze(1)
    p_i_j = p_i_j.sum(dim=0)
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j


def mi_loss(x, y, eps=sys.float_info.epsilon):
    """mutual information"""
    bn, k = x.size()
    p_i_j = compute_joint(x, y)

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

    p_i_j = torch.where(p_i_j < eps, torch.tensor([eps], device=p_i_j.device), p_i_j)
    p_j = torch.where(p_j < eps, torch.tensor([eps], device=p_j.device), p_j)
    p_i = torch.where(p_i < eps, torch.tensor([eps], device=p_i.device), p_i)

    loss = - p_i_j * (torch.log(p_j) + torch.log(p_i) - torch.log(p_i_j))

    loss = loss.sum()

    return loss


def cosine_similarity_loss(x, y):
    cos = nn.CosineSimilarity(dim=1)
    outputs = cos(x, y) + 1
    loss = outputs.sum()
    return loss


def distribution_loss(Q, P, eps=sys.float_info.epsilon):
    loss = F.kl_div(Q.log(), P, reduction='batchmean')
    return loss


def zero_loss(x):
    x = torch.abs(x)
    x = torch.sum(x, dim=1)
    return torch.mean(x)


def info_nce_loss(features, batch_size, temperature=0.07):
    n_views = features.size(0) // batch_size
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(features.device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

    logits = logits / temperature

    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, labels)

    return loss


def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)


def binary_cross_entropy(x_pred, x):
    # mask = torch.sign(x)
    return - torch.sum(x * torch.log(x_pred + 1e-8) + (1 - x) * torch.log(1 - x_pred + 1e-8), dim=1)


def reconstruction_loss(decoded, x):
    loss_func = torch.nn.MSELoss()
    loss_rec = loss_func(decoded, x)
    return loss_rec


def zinb_loss(x, mu, theta, pi, scale_factor=1.0, eps=1e-8):
    # x = x.float()
    scale_factor = scale_factor[:, None]
    mu = mu * scale_factor

    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))

    softplus_pi = F.softplus(-pi)

    log_theta_eps = torch.log(theta + eps)

    log_theta_mu_eps = torch.log(theta + mu + eps)

    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = (-softplus_pi + pi_theta_log
                     + x * (torch.log(mu + eps) - log_theta_mu_eps)
                     + torch.lgamma(x + theta)
                     - torch.lgamma(theta)
                     - torch.lgamma(x + 1))

    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero
    result = - torch.sum(res, dim=1)
    result = _nan2inf(result)
    result = torch.mean(result)

    return result


def log_truncated_normal(x, mu, sigma, eps=1e-8):
    # sigma = torch.minimum(sigma, torch.tensor(1e3))

    var = (sigma ** 2)
    log_scale = torch.log(sigma + eps)
    tmp = torch.tensor(math.pi, device='cuda')
    return -((x - mu) ** 2) / (2 * var) - log_scale - torch.log(torch.sqrt(2 * tmp))


def tobit_loss(x, mu, sigma, eps=1e-8):
    ll1 = log_truncated_normal(x, mu, sigma)
    cdf = np.float32(norm.cdf(-mu.cpu().detach().numpy() / sigma.cpu().detach().numpy()))
    ll2 = torch.log(torch.tensor(cdf, device=mu.device) + eps)
    tmp = torch.where(x > 0, ll1, ll2)
    result = - torch.mean(tmp)
    return result


def kl_loss(z_mean, z_stddev):
    return torch.mean(-0.5 * torch.sum(1 + 2 * torch.log(z_stddev) - z_mean ** 2 - z_stddev ** 2, dim=1), dim=0)


def distance_matrix(pts_src, pts_dst, p=2):
    x_col = pts_src.unsqueeze(1)
    y_row = pts_dst.unsqueeze(0)
    distance = torch.sum((torch.abs(x_col - y_row)) ** p, 2)
    return distance


def unbalanced_ot(tran, z1, z2, reg=0.1, reg_m=1.0, couple=None, device='cpu'):

    ns = z1.size(0)
    nt = z2.size(0)

    cost_pp = distance_matrix(z1, z2, p=2) + 1e-6

    p_s = torch.ones(ns, 1) / ns
    p_t = torch.ones(nt, 1) / nt

    p_s = p_s.to(device)
    p_t = p_t.to(device)

    if tran is None:
        tran = torch.ones(ns, nt) / (ns * nt)
    tran = tran.to(device)

    dual = (torch.ones(ns, 1) / ns).to(device)
    f = reg_m / (reg_m + reg)

    for m in range(10):
        if couple is not None:
            couple = couple.to(device)
            cost = cost_pp * couple
        else:
            cost = cost_pp

        kernel = torch.exp(-cost / (reg * torch.max(torch.abs(cost)))) * tran
        b = p_t / (torch.t(kernel) @ dual)
        # dual = p_s / (kernel @ b)
        for i in range(10):
            dual = (p_s / (kernel @ b)) ** f
            b = (p_t / (torch.t(kernel) @ dual)) ** f
        tran = (dual @ torch.t(b)) * kernel
    if torch.isnan(tran).sum() > 0:
        tran = (torch.ones(ns, nt) / (ns * nt)).to(device)

    d_fgw = (cost_pp * tran.detach().data).sum()

    return d_fgw, tran.detach().cpu()
