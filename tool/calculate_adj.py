from coembedding.function import co_embedding
import numpy as np
from numba import jit
import scanpy as sc
import pandas as pd
import math
import datatable as dt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

@jit(nopython=True)
def cal_distance(X):
    adj_mt=np.zeros((X.shape[0],X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            #adj_df[i,j]=torch.sqrt(torch.sum(torch.square(X[i,:]-X[j,:])))
            adj_mt[i,j]=np.linalg.norm(X[i,:]-X[j,:])
    return adj_mt

def cal_adj(RNA_data_adata,Spatial_data_adata,outdir):
    adata_path=outdir+'/co_adata.h5ad'
    if os.path.exists(adata_path):
        print("Data exists, load it")
        adata=sc.read_h5ad(adata_path)
    else:
        adata = co_embedding(data_list=[RNA_data_adata,Spatial_data_adata],outdir=outdir)
        adata.write(adata_path)

    sc_data=adata[adata.obs['batch']=="0"]
    st_data=adata[adata.obs['batch']=="1"]

    #Calculates the distance between two pairs in co-embedding space
    final_adj_df_path=outdir+'/final_adj_df.txt'
    if os.path.exists(final_adj_df_path):
        print("load adj dataframe")
        final_adj_df=pd.read_csv(final_adj_df_path,index_col=0)
    else:
        X=adata.obsm['latent']
        adj_mt=cal_distance(X)
        print("Finish calculate distance")

        adj_df=pd.DataFrame(adj_mt,index=adata.obs_names, columns=adata.obs_names)
        adj_df = adj_df.rename_axis('index').reset_index()
        adj_df.index=adata.obs_names
        
        #
        adj_df.loc[sc_data.obs_names,sc_data.obs_names]=np.nan
        new_adj_df=pd.melt(adj_df,id_vars='index',var_name='index2', value_name='dis').dropna(axis=0,how='any')
        
        #cell
        cell_df=new_adj_df[(new_adj_df["index2"].isin(list(sc_data.obs_names)))]
        cell_df1 = cell_df.sort_values(['index2','dis'], ascending=[True, True]).groupby('index2', group_keys=False).apply(lambda x: x.head(1))
        cell_df2=cell_df1.copy()
        cell_df2[['index', 'index2']] = cell_df2[['index2', 'index']]
        final_cell_df=pd.concat([cell_df1,cell_df2],ignore_index=True)
        # final_cell_df["dis"] /= math.ceil(final_cell_df["dis"].max())
        # final_cell_df["weight"]=1-final_cell_df["dis"]
        
        #spots
        spot_df=pd.merge(new_adj_df[(new_adj_df["index2"].isin(list(st_data.obs_names)))],
                    new_adj_df[(new_adj_df["index"].isin(list(st_data.obs_names)))], 
                    how='inner')
        #Take the first 20 of the nearest distance and take the intersection
        spot_df1 = spot_df.sort_values(['index2','dis'], ascending=[True, True]).groupby('index2', group_keys=False).apply(lambda x: x.head(20))
        spot_df2 = spot_df.sort_values(['index','dis'], ascending=[True, True]).groupby('index', group_keys=False).apply(lambda x: x.head(20))
        final_spot_df=pd.merge(spot_df1, spot_df2, how='inner')
        # final_spot_df["dis"] /= final_spot_df["dis"].max()
        # final_spot_df["weight"]=1-final_spot_df["dis"]
        
        final_df=pd.concat([final_cell_df,final_spot_df],ignore_index=True)
        final_df["dis"] /= final_df["dis"].max()
        final_df["weight"]=1-final_df["dis"]
        final_df.loc[final_df["weight"]==1,["weight"]]=0
        #Combine individual cells according to cell type
        for i in range(final_df.shape[0]):
            for j in range(final_df.shape[1]):
                if final_df.iloc[i,j] in sc_data.obs_names:
                    final_df.iloc[i,j]=sc_data.obs.loc[final_df.iloc[i,j],'celltype']
        groups=final_df.groupby(['index','index2']).sum().reset_index()
        
        #Adjust the combined weights
        value_counts = sc_data.obs['celltype'].value_counts()
        final_adj_df=groups.copy()
        for i in range(final_adj_df.shape[0]):
            for j in range(final_adj_df.shape[1]):
                if final_adj_df.iloc[i,j] in list(sc_data.obs["celltype"]):
                    ncell=final_adj_df.loc[i,"dis"]+final_adj_df.loc[i,"weight"]
                    final_adj_df.loc[i,"weight"]=(ncell/value_counts[final_adj_df.iloc[i,j]])+(final_adj_df.loc[i,"weight"]/ncell)
        assert not np.isnan(np.array(final_adj_df.iloc[:,-2:])).any(), "wrong adj_df"
        final_adj_df.to_csv(final_adj_df_path)
    return final_adj_df

def another_cal_adj(adata_ret,Spatial_data_adata,outdir):
    adata_path=outdir+'/another_co_adata.h5ad'
    if os.path.exists(adata_path):
        print("Data exists, load it")
        adata=sc.read_h5ad(adata_path)
    else:
        adata = co_embedding(data_list=[adata_ret,Spatial_data_adata],outdir=outdir)
        adata.write(adata_path)

    sc_data=adata[adata.obs['batch']=="0"]
    st_data=adata[adata.obs['batch']=="1"]

    #Calculates the distance between two pairs in co-embedding space
    final_adj_df_path=outdir+'/another_final_adj_df.txt'
    if os.path.exists(final_adj_df_path):
        print("load adj dataframe")
        final_df=pd.read_csv(final_adj_df_path,index_col=0)
    else:
        X=adata.obsm['latent']
        adj_mt=cal_distance(X)
        print("Finish calculate distance")

        adj_df=pd.DataFrame(adj_mt,index=adata.obs_names, columns=adata.obs_names)
        adj_df = adj_df.rename_axis('index').reset_index()
        adj_df.index=adata.obs_names
        
        #
        adj_df.loc[sc_data.obs_names,sc_data.obs_names]=np.nan
        new_adj_df=pd.melt(adj_df,id_vars='index',var_name='index2', value_name='dis').dropna(axis=0,how='any')
        
        #cell
        cell_df=new_adj_df[(new_adj_df["index2"].isin(list(sc_data.obs_names)))]
        cell_df1 = cell_df.sort_values(['index2','dis'], ascending=[True, True]).groupby('index2', group_keys=False).apply(lambda x: x.head(20))
        cell_df2=cell_df1.copy()
        cell_df2[['index', 'index2']] = cell_df2[['index2', 'index']]
        final_cell_df=pd.concat([cell_df1,cell_df2],ignore_index=True)
        # final_cell_df["dis"] /= math.ceil(final_cell_df["dis"].max())
        # final_cell_df["weight"]=1-final_cell_df["dis"]
        
        #spots
        spot_df=pd.merge(new_adj_df[(new_adj_df["index2"].isin(list(st_data.obs_names)))],
                    new_adj_df[(new_adj_df["index"].isin(list(st_data.obs_names)))], 
                    how='inner')
        #Take the first 20 of the nearest distance and take the intersection
        spot_df1 = spot_df.sort_values(['index2','dis'], ascending=[True, True]).groupby('index2', group_keys=False).apply(lambda x: x.head(20))
        spot_df2 = spot_df.sort_values(['index','dis'], ascending=[True, True]).groupby('index', group_keys=False).apply(lambda x: x.head(20))
        final_spot_df=pd.merge(spot_df1, spot_df2, how='inner')
        # final_spot_df["dis"] /= final_spot_df["dis"].max()
        # final_spot_df["weight"]=1-final_spot_df["dis"]
        
        final_df=pd.concat([final_cell_df,final_spot_df],ignore_index=True)
        final_df["dis"] /= final_df["dis"].max()
        final_df["weight"]=1-final_df["dis"]
        final_df.loc[final_df["weight"]==1,["weight"]]=0

        final_df.to_csv(final_adj_df_path)
    return final_df


def impute_cal_adj(RNA_data_adata,Spatial_data_adata,outdir):
    adata_path=outdir+'/co_adata.h5ad'
    if os.path.exists(adata_path):
        print("Data exists, load it")
        adata=sc.read_h5ad(adata_path)
    else:
        adata = co_embedding(data_list=[RNA_data_adata,Spatial_data_adata],outdir=outdir,min_features=0, min_cells=0)
        adata.write(adata_path)

    sc_data=adata[adata.obs['batch']=="0"]
    st_data=adata[adata.obs['batch']=="1"]

    #Calculates the distance between two pairs in co-embedding space
    final_adj_df_path=outdir+'/final_adj_df.txt'
    if os.path.exists(final_adj_df_path):
        print("load adj dataframe")
        final_adj_df=pd.read_csv(final_adj_df_path,index_col=0)
    else:
        X=adata.obsm['latent']
        adj_mt=cal_distance(X)
        print("Finish calculate distance")

        adj_df=pd.DataFrame(adj_mt,index=adata.obs_names, columns=adata.obs_names)
        adj_df = adj_df.rename_axis('index').reset_index()
        adj_df.index=adata.obs_names
        
        #
        adj_df.loc[sc_data.obs_names,sc_data.obs_names]=np.nan
        new_adj_df=pd.melt(adj_df,id_vars='index',var_name='index2', value_name='dis').dropna(axis=0,how='any')
        
        #cell
        cell_df=new_adj_df[(new_adj_df["index2"].isin(list(sc_data.obs_names)))]
        cell_df1 = cell_df.sort_values(['index2','dis'], ascending=[True, True]).groupby('index2', group_keys=False).apply(lambda x: x.head(1))
        cell_df2=cell_df1.copy()
        cell_df2[['index', 'index2']] = cell_df2[['index2', 'index']]
        final_cell_df=pd.concat([cell_df1,cell_df2],ignore_index=True)
        # final_cell_df["dis"] /= math.ceil(final_cell_df["dis"].max())
        # final_cell_df["weight"]=1-final_cell_df["dis"]
        
        #spots
        spot_df=pd.merge(new_adj_df[(new_adj_df["index2"].isin(list(st_data.obs_names)))],
                    new_adj_df[(new_adj_df["index"].isin(list(st_data.obs_names)))], 
                    how='inner')
        #Take the first 20 of the nearest distance and take the intersection
        spot_df1 = spot_df.sort_values(['index2','dis'], ascending=[True, True]).groupby('index2', group_keys=False).apply(lambda x: x.head(20))
        spot_df2 = spot_df.sort_values(['index','dis'], ascending=[True, True]).groupby('index', group_keys=False).apply(lambda x: x.head(20))
        final_spot_df=pd.merge(spot_df1, spot_df2, how='inner')
        # final_spot_df["dis"] /= final_spot_df["dis"].max()
        # final_spot_df["weight"]=1-final_spot_df["dis"]
        
        final_df=pd.concat([final_cell_df,final_spot_df],ignore_index=True)
        final_df["dis"] /= final_df["dis"].max()
        final_df["weight"]=1-final_df["dis"]
        final_df.loc[final_df["weight"]==1,["weight"]]=0
        #Combine individual cells according to cell type
        for i in range(final_df.shape[0]):
            for j in range(final_df.shape[1]):
                if final_df.iloc[i,j] in sc_data.obs_names:
                    final_df.iloc[i,j]=sc_data.obs.loc[final_df.iloc[i,j],'celltype']
        groups=final_df.groupby(['index','index2']).sum().reset_index()
        
        #Adjust the combined weights
        value_counts = sc_data.obs['celltype'].value_counts()
        final_adj_df=groups.copy()
        for i in range(final_adj_df.shape[0]):
            for j in range(final_adj_df.shape[1]):
                if final_adj_df.iloc[i,j] in list(sc_data.obs["celltype"]):
                    ncell=final_adj_df.loc[i,"dis"]+final_adj_df.loc[i,"weight"]
                    final_adj_df.loc[i,"weight"]=(ncell/value_counts[final_adj_df.iloc[i,j]])+(final_adj_df.loc[i,"weight"]/ncell)
        assert not np.isnan(np.array(final_adj_df.iloc[:,-2:])).any(), "wrong adj_df"
        final_adj_df.to_csv(final_adj_df_path)
    return final_adj_df

def impute_another_cal_adj(adata_ret,Spatial_data_adata,outdir):
    adata_path=outdir+'/another_co_adata.h5ad'
    if os.path.exists(adata_path):
        print("Data exists, load it")
        adata=sc.read_h5ad(adata_path)
    else:
        adata = co_embedding(data_list=[adata_ret,Spatial_data_adata],outdir=outdir,min_features=0, min_cells=0)
        adata.write(adata_path)

    sc_data=adata[adata.obs['batch']=="0"]
    st_data=adata[adata.obs['batch']=="1"]

    #Calculates the distance between two pairs in co-embedding space
    final_adj_df_path=outdir+'/another_final_adj_df.txt'
    if os.path.exists(final_adj_df_path):
        print("load adj dataframe")
        final_df=pd.read_csv(final_adj_df_path,index_col=0)
    else:
        X=adata.obsm['latent']
        adj_mt=cal_distance(X)
        print("Finish calculate distance")

        adj_df=pd.DataFrame(adj_mt,index=adata.obs_names, columns=adata.obs_names)
        adj_df = adj_df.rename_axis('index').reset_index()
        adj_df.index=adata.obs_names
        
        #
        adj_df.loc[sc_data.obs_names,sc_data.obs_names]=np.nan
        new_adj_df=pd.melt(adj_df,id_vars='index',var_name='index2', value_name='dis').dropna(axis=0,how='any')
        
        #cell
        cell_df=new_adj_df[(new_adj_df["index2"].isin(list(sc_data.obs_names)))]
        cell_df1 = cell_df.sort_values(['index2','dis'], ascending=[True, True]).groupby('index2', group_keys=False).apply(lambda x: x.head(20))
        cell_df2=cell_df1.copy()
        cell_df2[['index', 'index2']] = cell_df2[['index2', 'index']]
        final_cell_df=pd.concat([cell_df1,cell_df2],ignore_index=True)
        # final_cell_df["dis"] /= math.ceil(final_cell_df["dis"].max())
        # final_cell_df["weight"]=1-final_cell_df["dis"]
        
        #spots
        spot_df=pd.merge(new_adj_df[(new_adj_df["index2"].isin(list(st_data.obs_names)))],
                    new_adj_df[(new_adj_df["index"].isin(list(st_data.obs_names)))], 
                    how='inner')
        #Take the first 20 of the nearest distance and take the intersection
        spot_df1 = spot_df.sort_values(['index2','dis'], ascending=[True, True]).groupby('index2', group_keys=False).apply(lambda x: x.head(20))
        spot_df2 = spot_df.sort_values(['index','dis'], ascending=[True, True]).groupby('index', group_keys=False).apply(lambda x: x.head(20))
        final_spot_df=pd.merge(spot_df1, spot_df2, how='inner')
        # final_spot_df["dis"] /= final_spot_df["dis"].max()
        # final_spot_df["weight"]=1-final_spot_df["dis"]
        
        final_df=pd.concat([final_cell_df,final_spot_df],ignore_index=True)
        final_df["dis"] /= final_df["dis"].max()
        final_df["weight"]=1-final_df["dis"]
        final_df.loc[final_df["weight"]==1,["weight"]]=0

        final_df.to_csv(final_adj_df_path)
    return final_df