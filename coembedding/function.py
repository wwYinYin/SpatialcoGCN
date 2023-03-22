import torch
import numpy as np
import os
import scanpy as sc
from anndata import AnnData

from .data import load_data
from .net.vae import VAE
from .net.utils import EarlyStopping
from .metrics import batch_entropy_mixing_score, silhouette_score
from .logger import create_logger
from .plot import embedding


def co_embedding(
        data_list=None, 
        join='inner', 
        batch_key='batch', 
        batch_name='batch',
        min_features=5, 
        min_cells=1, 
        target_sum=None,
        n_top_features=None, 
        batch_size=64, 
        lr=2e-4, 
        max_iteration=30000,
        seed=124, 
        gpu=0, 
        outdir='output/',  
        chunk_size=20000,
        show=True,
        eval=False,
    ):
    
    np.random.seed(seed) # seed
    torch.manual_seed(seed)

    if torch.cuda.is_available(): # cuda device
        device='cuda'
        torch.cuda.set_device(gpu)
    else:
        device='cpu'
    
    outdir = outdir+'/'
    os.makedirs(outdir+'/checkpoint', exist_ok=True)
    log = create_logger('', fh=outdir+'log.txt')

    adata, trainloader, testloader = load_data(
        data_list, 
        join=join,
        target_sum=target_sum,
        n_top_features=n_top_features,
        batch_size=batch_size, 
        chunk_size=chunk_size,
        min_features=min_features, 
        min_cells=min_cells,
        batch_name=batch_name, 
        batch_key=batch_key,
        log=log,
    )
    
    early_stopping = EarlyStopping(patience=10, checkpoint_file=outdir+'/checkpoint/model.pt')
    x_dim, n_domain = adata.shape[1], len(adata.obs['batch'].cat.categories)
    
    # model config
    enc = [['fc', 1024, 1, 'relu'],['fc', 30, '', '']]  # TO DO
    dec = [['fc', x_dim, n_domain, 'sigmoid']]

    model = VAE(enc, dec, n_domain=n_domain)
    
    log.info('model\n'+model.__repr__())
    model.fit(
        trainloader, 
        lr=lr, 
        max_iteration=max_iteration, 
        device=device, 
        early_stopping=early_stopping, 
    )
    #torch.save({'n_top_features':adata.var.index, 'enc':enc, 'dec':dec, 'n_domain':n_domain}, outdir+'/checkpoint/config.pt')     
        
    adata.obsm['latent'] = model.encodeBatch(testloader, device=device, eval=eval) # save latent rep
    
    model.to('cpu')
    del model
    
    #adata.write(outdir+'adata.h5ad', compression='gzip')  
    log.info('Plot umap')
    sc.pp.neighbors(adata, n_neighbors=30, use_rep='latent')
    sc.tl.umap(adata, min_dist=0.1)
    #sc.tl.leiden(adata)
    
    # UMAP visualization
    sc.settings.figdir = outdir
    sc.set_figure_params(dpi=80, figsize=(3,3))
    cols = ['batch', 'celltype', 'leiden']
    color = [c for c in cols if c in adata.obs]
    if len(color) > 0:
        sc.pl.umap(adata, color=color, save='.pdf', wspace=0.4, ncols=4, show=show)  
    
    return adata

