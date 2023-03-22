import scanpy as sc
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.nn.functional import softmax
from scipy.sparse import issparse
from tool.deconv_metric import CalDataMetric
import sys
import os
import logging
import torch
from tool.config import opt
from model.models import coGCN,pygGCN
from tool.utils import EarlyStopping
from model import losses
def coGCN_training(adata,adj,gd_results,n_celltype,device):
    in_feat=adata.shape[1]
    model=coGCN(in_feat,n_celltype).to(device)
    loss_fn = losses.MyLoss_simulate()
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.decay_LR[0], gamma=opt.decay_LR[1])
    model.train()

    sc_data=adata[adata.obs['batch']=="0"]
    st_data=adata[adata.obs['batch']=="1"]

    Input=torch.FloatTensor(adata.X.A).to(device)
    target=torch.FloatTensor(st_data.X.A).to(device)
    S=torch.FloatTensor(sc_data.X.A).to(device)
    adj = torch.FloatTensor(adj).to(device)

    max_pcc=0
    early_stopping = EarlyStopping()
    for epoch in range(opt.max_epoch):    
        pred=model(Input,adj) #[204×15]

        pred_M=pred[n_celltype:,]
        M_probs = (pred_M/pred_M.sum(axis=0))
        M_probs=torch.where(torch.isnan(M_probs), torch.full_like(M_probs, 0), M_probs)
        #M_probs = softmax(pred_M, dim=0)
        G_pred = torch.matmul(M_probs, S)
        #d_pred = torch.log(d_source @ M_probs.T) 

        loss = loss_fn(G_pred, target)
        # density_loss=density_loss_fn(d_pred,d)
        # total_loss=loss+density_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        M_probs1 = (M_probs.T/M_probs.sum(axis=1)).T.detach().numpy()
        M_probs1=pd.DataFrame(M_probs1,index=adata.obs_names[n_celltype:],columns=adata.obs_names[0:n_celltype])
        M_probs1 = M_probs1.loc[:,np.unique(M_probs1.columns)]
        M_probs1 = M_probs1.fillna(0)
        M_probs1=np.array(M_probs1)
        loss_val=losses.val_loss(M_probs1,gd_results)

        if loss_val[0]>max_pcc:
            max_pcc=loss_val[0]
            best_M=M_probs

        if (epoch+1)%10==0:
            #print(pred_M)
            print("Epoch {:03d}: Loss {}".format(
            epoch+1, loss))
            print(loss_val)
            print("max_pcc:{}".format(max_pcc))

        scheduler.step()
        early_stopping(loss.detach().numpy())
        if early_stopping.early_stop:
            print('EarlyStopping: run {} epoch'.format(epoch+1))
            break

    with torch.no_grad():
        output = best_M.cpu().numpy()
        return output
 
def start(adata,adj,gd_results,GCN_method,seed=2022):
    #device=torch.device('cuda:0')
    np.random.seed(seed) # seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device='cpu'
    n_celltype=len(adata.obs[adata.obs["batch"]=="0"])
    if GCN_method=="coGCN":
        output=coGCN_training(adata,adj,gd_results,n_celltype,device) #[189×15]
    if GCN_method=="pygGCN":
        output=pygGCN_training(adata,adj,gd_results,n_celltype,device) #[189×15]
    
    adata_map = sc.AnnData(
        X=output,
        obs=adata.obs[n_celltype:].copy(),
        var=adata.obs[0:n_celltype].copy(),
    )
    return adata_map


def pygGCN_training(adata,adj,gd_results,n_celltype,device):
    in_feat=adata.shape[1]
    model=pygGCN(in_feat,n_celltype).to(device)

    loss_fn = losses.MyLoss_simulate()
    density_loss_fn=torch.nn.KLDivLoss(reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.decay_LR[0], gamma=opt.decay_LR[1])
    model.train()

    sc_data=adata[adata.obs['batch']=="0"]
    st_data=adata[adata.obs['batch']=="1"]

    Input=torch.tensor(adata.X.A,dtype=torch.float,device=device)
    target=torch.tensor(st_data.X.A,dtype=torch.float,device=device)
    S=torch.tensor(sc_data.X.A,dtype=torch.float,device=device)
    edge_index = torch.tensor(adj[0:2,:], dtype=torch.long,device=device)
    edge_weight= torch.tensor(adj[2,:], dtype=torch.float,device=device)

    d_source = np.array(sc_data.obs["cluster_density"])
    d_source = torch.tensor(d_source, device=device, dtype=torch.float)
    d=st_data.obs["rna_count_based_density"]
    d = torch.tensor(d, device=device, dtype=torch.float)
    
    max_pcc=0
    for epoch in range(opt.max_epoch):    
        pred=model(Input,edge_index,edge_weight) #[204×15]

        pred_M=pred[n_celltype:,]
        M_probs = (pred_M/pred_M.sum(axis=0))
        M_probs=torch.where(torch.isnan(M_probs), torch.full_like(M_probs, 0), M_probs)
        #M_probs = softmax(pred_M, dim=0)
        G_pred = torch.matmul(M_probs, S)
        d_pred = torch.log(d_source @ M_probs.T) 

        loss = loss_fn(G_pred, target)
        density_loss=density_loss_fn(d_pred,d)
        total_loss=loss+density_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        M_probs1 = (M_probs.T/M_probs.sum(axis=1)).T.detach().numpy()
        M_probs1=pd.DataFrame(M_probs1,index=adata.obs_names[n_celltype:],columns=adata.obs_names[0:n_celltype])
        M_probs1 = M_probs1.loc[:,np.unique(M_probs1.columns)]
        M_probs1 = M_probs1.fillna(0)
        M_probs1=np.array(M_probs1)
        loss_val=losses.val_loss(M_probs1,gd_results)

        if loss_val[0]>max_pcc:
            max_pcc=loss_val[0]
            best_M=M_probs

        if epoch%10==0:
            #print(pred_M)
            print("Epoch {:03d}: Loss {}".format(
            epoch, loss))
            print(loss_val)
            print("max_pcc:{}".format(max_pcc))
        
        scheduler.step()
    with torch.no_grad():
        output = best_M.cpu().numpy()
        return output