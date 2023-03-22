from anndata import AnnData
import scanpy as sc
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.linalg import fractional_matrix_power


def preprocessing(
        sc_adata: AnnData, 
        st_adata: AnnData,
        min_features: int = 5, 
        min_cells: int = 1
    ):
    """
    Preprocessing RNA-seq and spacial data
    
    Parameters
    ----------
    min_features
        Filtered out cells that are detected in less than n genes. Default: 600.
    min_cells
        Filtered out genes that are detected in less than n cells. Default: 3.
        
    Return
    -------
    The AnnData object after preprocessing.
    """
    sc.pp.filter_cells(sc_adata, min_genes=min_features)
    sc.pp.filter_cells(st_adata, min_genes=min_features)

    sc.pp.filter_genes(sc_adata, min_cells=min_cells)
    sc.pp.filter_genes(st_adata, min_cells=min_cells)

    sc.pp.normalize_total(sc_adata)

    celltype_counts = sc_adata.obs['celltype'].value_counts()
    celltype_drop = celltype_counts.index[celltype_counts < 2]
    print(f'Drop celltype {list(celltype_drop)} contain less 2 sample')
    sc_adata = sc_adata[~sc_adata.obs['celltype'].isin(celltype_drop),].copy()
    sc.tl.rank_genes_groups(sc_adata, groupby='celltype', use_raw=False)
    markers_df = pd.DataFrame(sc_adata.uns["rank_genes_groups"]["names"]).iloc[0:200, :]
    genes_sc = np.unique(markers_df.melt().value.values)
    genes_st = st_adata.var_names.values
    genes = list(set(genes_sc).intersection(set(genes_st)))
    genes=np.unique(genes)
    st_adata = st_adata[:, genes].copy()
    sc_adata = sc_adata[:, genes].copy()

    rna_count_per_spot = np.array(st_adata.X.sum(axis=1)).squeeze()
    st_adata.obs["rna_count_based_density"] = rna_count_per_spot / np.sum(rna_count_per_spot)

    # Transform single-cell data into cell type expression matrices
    adata_ret = adata_to_cluster_expression(sc_adata)


    adata_ret.X = csr_matrix(adata_ret.X)
    st_adata.X = csr_matrix(st_adata.X)

    concat_st_sc=sc.AnnData.concatenate(adata_ret,st_adata)

    ncelltype=len(concat_st_sc.obs_names[concat_st_sc.obs["batch"]=="0"])
    obs_name=list(concat_st_sc.obs_names)
    obs_name[0:ncelltype]=concat_st_sc.obs['celltype'][0:ncelltype]
    obs_name[ncelltype:]=st_adata.obs_names
    concat_st_sc.obs_names=obs_name

    concat_st_sc = concat_st_sc[:, [gene for gene in concat_st_sc.var_names 
                  if not str(gene).startswith(tuple(['ERCC', 'MT-', 'mt-']))]]

    return concat_st_sc

def mapping_adj2pyg(adj_df,concat_st_sc):
    adj_matrix=[]
    for i in range(adj_df.shape[0]):
        if f"{adj_df.iloc[i,0]}" != f"{adj_df.iloc[i,1]}":
            if (f"{adj_df.iloc[i,0]}" in list(concat_st_sc.obs_names)) and (f"{adj_df.iloc[i,1]}" in list(concat_st_sc.obs_names)):
                edge_list=[list(concat_st_sc.obs_names).index(f"{adj_df.iloc[i,0]}"),
                        list(concat_st_sc.obs_names).index(f"{adj_df.iloc[i,1]}"),
                        adj_df.iloc[i,3]]
                adj_matrix.append(edge_list)
    
    adj_matrix=np.array(adj_matrix).T

    return adj_matrix


def mapping_adj2matrix(adj_df,concat_st_sc):
    obs_name=list(concat_st_sc.obs_names)
    adj_matrix=np.zeros((concat_st_sc.shape[0],concat_st_sc.shape[0]), dtype=np.float32)
    for i in range(len(adj_df)):
        edge=adj_df.iloc[i,:]
        #print(edge)
        if (edge[0] in obs_name) and (edge[1] in obs_name):
            i=obs_name.index(edge[0])
            j=obs_name.index(edge[1])
            adj_matrix[i,j]=edge[3]
    ssum=np.sum(adj_matrix,1)
    ssum[ssum==0]=1
    D=np.diag(ssum)
    D_hat=fractional_matrix_power(D, -0.5)
    adj_matrix=D_hat@adj_matrix@D_hat
    adj_matrix += np.eye(adj_matrix.shape[0])

    return adj_matrix


def impute_preprocessing(
        sc_adata: AnnData, 
        st_adata: AnnData,
        min_features: int = 0, 
        min_cells: int = 0):

    sc.pp.filter_cells(sc_adata, min_genes=min_features)
    sc.pp.filter_cells(st_adata, min_genes=min_features)

    sc.pp.filter_genes(sc_adata, min_cells=min_cells)
    sc.pp.filter_genes(st_adata, min_cells=min_cells)

    sc.pp.normalize_total(sc_adata)

    genes_sc = sc_adata.var_names.values
    genes_st = st_adata.var_names.values
    genes = list(set(genes_sc).intersection(set(genes_st)))
    genes=np.unique(genes)
    st_adata = st_adata[:, genes].copy()
    sc_adata = sc_adata[:, genes].copy()

    rna_count_per_spot = np.array(st_adata.X.sum(axis=1)).squeeze()
    st_adata.obs["rna_count_based_density"] = rna_count_per_spot / np.sum(rna_count_per_spot)

    # Transform single-cell data into cell type expression matrices
    adata_ret = adata_to_cluster_expression(sc_adata)


    adata_ret.X = csr_matrix(adata_ret.X)
    st_adata.X = csr_matrix(st_adata.X)

    concat_st_sc=sc.AnnData.concatenate(adata_ret,st_adata)

    ncelltype=len(concat_st_sc.obs_names[concat_st_sc.obs["batch"]=="0"])
    obs_name=list(concat_st_sc.obs_names)
    obs_name[0:ncelltype]=concat_st_sc.obs['celltype'][0:ncelltype]
    obs_name[ncelltype:]=st_adata.obs_names
    concat_st_sc.obs_names=obs_name

    concat_st_sc = concat_st_sc[:, [gene for gene in concat_st_sc.var_names 
                  if not str(gene).startswith(tuple(['ERCC', 'MT-', 'mt-']))]]

    return concat_st_sc

def adata_to_cluster_expression(sc_adata):
    # Transform single-cell data into cell type expression matrices
    value_counts = sc_adata.obs['celltype'].value_counts(normalize=True)
    unique_labels = np.unique(value_counts.index)
    new_obs = pd.DataFrame({'celltype': unique_labels})
    adata_ret = sc.AnnData(obs=new_obs, var=sc_adata.var, uns=sc_adata.uns)

    X_new = np.empty((len(unique_labels), sc_adata.shape[1]))
    for index, l in enumerate(unique_labels):
        X_new[index] = sc_adata[sc_adata.obs['celltype'] == l].X.sum(axis=0)
    adata_ret.X = X_new
    adata_ret.obs["cluster_density"] = adata_ret.obs['celltype'].map(lambda i: value_counts[i])
    adata_ret.obs_names=adata_ret.obs["celltype"]
    return adata_ret