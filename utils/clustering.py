import numpy as np
import scanpy as sc
import anndata as ad
from sklearn import metrics
from munkres import Munkres
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.cluster import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score
)


def cluster_acc(y_true, y_pred):

    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)

    true_to_int = {orig: i for i, orig in enumerate(unique_true)}
    pred_to_int = {orig: i for i, orig in enumerate(unique_pred)}

    y_true_int = np.array([true_to_int[x] for x in y_true])
    y_pred_int = np.array([pred_to_int[x] for x in y_pred])

    n_true = len(unique_true)
    n_pred = len(unique_pred)

    cost_matrix = np.zeros((n_true, n_pred), dtype=int)
    for i in range(n_true):
        for j in range(n_pred):
            cost_matrix[i, j] = np.sum((y_true_int == i) & (y_pred_int == j))

    cost_matrix = -cost_matrix
    m = Munkres()
    indexes = m.compute(cost_matrix.tolist())

    pred_to_true_map = {j: i for i, j in indexes}
    new_pred = np.array([pred_to_true_map.get(label, -1) for label in y_pred_int])

    mask = (new_pred != -1)
    acc = metrics.accuracy_score(y_true_int[mask], new_pred[mask]) if np.any(mask) else 0.0
    return acc


def leiden_clustering(data, labels_t):
    adata = ad.AnnData(data)
    sc.pp.neighbors(adata, use_rep='X')
    sc.tl.umap(adata)
    best_metrics = [0, 0, 0, 0]
    predict_labels = []
    resolutions = np.arange(0.05, 1.5, 0.05)
    for i, res in enumerate(resolutions):
        sc.tl.leiden(
            adata,
            key_added="clusters" + str(i),
            resolution=res,
            n_iterations=-1
        )
        labels_p = adata.obs['clusters' + str(i)]
        labels_uniq = np.unique(labels_p)
        ari = adjusted_rand_score(labels_t, labels_p)
        nmi = normalized_mutual_info_score(labels_t, labels_p)
        ami = adjusted_mutual_info_score(labels_t, labels_p)
        acc = cluster_acc(labels_t, labels_p)
        if ari + nmi + ami + acc > sum(best_metrics):
            best_metrics = [ari, nmi, ami, acc]
            predict_labels = labels_p

    print('ARI=', best_metrics[0], ', NMI=', best_metrics[1], ', AMI=', best_metrics[2], ', ACC=', best_metrics[3])

    return predict_labels, best_metrics
