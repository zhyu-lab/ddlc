import scanpy as sc


def normalize_rna(adata, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True, use_hvg=True, pca=True, n_top_genes=3000, n_comps=256):

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)

    if size_factors:
        sc.pp.normalize_per_cell(adata)

    if logtrans_input:
        sc.pp.log1p(adata)

    if use_hvg:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        adata = adata[:, adata.var['highly_variable']]
        adata.obsm['hvg_data'] = adata.X.toarray()

    if normalize_input:
        sc.pp.scale(adata, max_value=10)

    if pca:
        sc.tl.pca(adata, n_comps=n_comps, svd_solver="auto")

    return adata


def normalize_atac(adata, filter_features=True, size_factors=True, normalize_input=True, logtrans_input=True, use_hvg=True, pca=True, n_top_genes=3000, n_comps=256):

    if filter_features:
        sc.pp.filter_genes(adata, min_counts=1)

    if size_factors:
        sc.pp.normalize_per_cell(adata)

    if logtrans_input:
        sc.pp.log1p(adata)

    if use_hvg:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        adata = adata[:, adata.var['highly_variable']]
        adata.obsm['hvg_data'] = adata.X.toarray()

    if normalize_input:
        sc.pp.scale(adata, max_value=10)

    if pca:
        sc.tl.pca(adata, n_comps=n_comps, svd_solver="auto")

    return adata


def load_ATAC_data(file_path):
    """
    Load ATAC data
    """
    ATAC_data = sc.read_h5ad(file_path)
    return ATAC_data


def load_RNA_data(file_path):
    """
    Load RNA data
    """
    RNA_data = sc.read_h5ad(file_path)
    return RNA_data

