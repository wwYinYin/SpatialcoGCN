import torch.nn as nn
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import scipy.stats as st
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mean_squared_error

class MyLoss_simulate(nn.Module):
    def __init__(self):
        super().__init__()
    def loss_function(self,preds,targets):
        def Pearsonr(im1,im2):
            mu1 = im1.mean()
            mu2 = im2.mean()
            sigma1 = torch.sqrt(((im1 - mu1) ** 2).mean())
            sigma2 = torch.sqrt(((im2 - mu2) ** 2).mean())
            sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
            pearsonr=sigma12/(sigma1*sigma2)
            #print(pearsonr)
            pearsonr=torch.where(torch.isnan(pearsonr), torch.full_like(pearsonr, 0), pearsonr)
            return pearsonr

        def compare_results(targets,preds,metric='ssim',axis=1):
            func = Pearsonr

            if axis == 1:
                c_tensor = torch.zeros(targets.shape[1])
                for i in range(targets.shape[1]):
                    r = func(targets[:,i], preds[:,i])
                    c_tensor[i]=r
            else:
                c_tensor = torch.zeros(targets.shape[0])
                for i in range(targets.shape[0]):
                    r = func(targets[i,:], preds[i,:])
                    c_tensor[i]=r
            return c_tensor
        
        # #pcc
        clusters_pcc_loss=1-compare_results(targets,preds,metric='pcc',axis=1).nanmean()
        # spots_pcc_loss=1-compare_results(targets,preds,metric='pcc',axis=0).nanmean()
        # # #cosin_similar
        clusters_cossim_loss=1-F.cosine_similarity(preds,targets,dim=0).mean()
        # spots_cossim_loss=1-F.cosine_similarity(preds,targets,dim=1).mean()
        total_loss=clusters_pcc_loss + clusters_cossim_loss
        return total_loss

    def forward(self,preds,targets):
        #print(preds)
        # preds = (preds.T/preds.sum(axis=1)).T
        # preds=torch.where(torch.isnan(preds), torch.full_like(preds, 0), preds)
        # targets = (targets.T/targets.sum(axis=1)).T
        # targets=torch.where(torch.isnan(targets), torch.full_like(targets, 0), targets)

        total_loss=self.loss_function(preds,targets)
        return total_loss

class My_impute_Loss(nn.Module):
    def __init__(self):
        super().__init__()
    def loss_function(self,preds,targets):
        def Pearsonr(im1,im2):
            mu1 = im1.mean()
            mu2 = im2.mean()
            sigma1 = torch.sqrt(((im1 - mu1) ** 2).mean())
            sigma2 = torch.sqrt(((im2 - mu2) ** 2).mean())
            sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
            pearsonr=sigma12/(sigma1*sigma2)
            #print(pearsonr)
            pearsonr=torch.where(torch.isnan(pearsonr), torch.full_like(pearsonr, 0), pearsonr)
            return pearsonr

        def SSIM(im1,im2,M=1):
            im1, im2 = im1/im1.sum(), im2/im2.sum()
            im1=torch.where(torch.isnan(im1), torch.full_like(im1, 0), im1)
            im2=torch.where(torch.isnan(im2), torch.full_like(im2, 0), im2)
            mu1 = im1.mean()
            mu2 = im2.mean()
            sigma1 = torch.sqrt(((im1 - mu1) ** 2).mean())
            sigma2 = torch.sqrt(((im2 - mu2) ** 2).mean())
            sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
            k1, k2, L = 0.01, 0.03, M
            C1 = (k1*L) ** 2
            C2 = (k2*L) ** 2
            C3 = C2/2
            l12 = (2*mu1*mu2 + C1)/(mu1 ** 2 + mu2 ** 2 + C1)
            c12 = (2*sigma1*sigma2 + C2)/(sigma1 ** 2 + sigma2 ** 2 + C2)
            s12 = (sigma12 + C3)/(sigma1*sigma2 + C3)
            ssim = l12 * c12 * s12
            return ssim
        
        def RMSE(x1,x2):
            mu1 = x1.mean()
            mu2 = x2.mean()
            sigma1 = torch.sqrt(((x1 - mu1) ** 2).mean())
            sigma2 = torch.sqrt(((x2 - mu2) ** 2).mean())
            x1=(x1 - mu1)/sigma1
            x2=(x2 - mu2)/sigma2
            rmse=torch.sqrt(((x1-x2)**2).nanmean())
            return rmse
        
        def JSD(x1,x2):
            def cal_KL(px,py):
                entr=px*torch.log(px/py)
                entr=torch.where(torch.isnan(entr), torch.full_like(entr, 0), entr)
                entr=torch.where(torch.isinf(entr), torch.full_like(entr, 0), entr)
                KL = entr.sum()
                return KL
            px1 = x1 / x1.sum()
            px2 = x2 / x2.sum()
            px1=torch.where(torch.isnan(px1), torch.full_like(px1, 0), px1)
            px2=torch.where(torch.isnan(px2), torch.full_like(px2, 0), px2)
            #M = (px1 + px2)/2
            #jsd = 0.5*cal_KL(px1,M)+0.5*cal_KL(px2,M)
            jsd = cal_KL(px1,px2)
            return jsd

        def compare_results(targets,preds,metric='ssim',axis=1):
            if metric=='pcc':
                func = Pearsonr
            if metric=='jsd':
                func = JSD
            if metric=='ssim':
                func = SSIM
            if metric=='rmse':
                func = RMSE

            if axis == 1:
                c_tensor = torch.zeros(targets.shape[1])
                for i in range(targets.shape[1]):
                    r = func(targets[:,i], preds[:,i])
                    c_tensor[i]=r
            else:
                c_tensor = torch.zeros(targets.shape[0])
                for i in range(targets.shape[0]):
                    r = func(targets[i,:], preds[i,:])
                    c_tensor[i]=r
            return c_tensor
        
        # #pcc
        clusters_pcc_loss=1-compare_results(targets,preds,metric='pcc',axis=1).nanmean()
        spots_pcc_loss=1-compare_results(targets,preds,metric='pcc',axis=0).nanmean()
        pcc_loss=clusters_pcc_loss+spots_pcc_loss
        #print("pcc_loss: ",pcc_loss)
        
        # # #cosin_similar
        clusters_cossim_loss=1-F.cosine_similarity(preds,targets,dim=0).mean()
        spots_cossim_loss=1-F.cosine_similarity(preds,targets,dim=1).mean()
        cossim_loss=clusters_cossim_loss+spots_cossim_loss
        # #print("cossim_loss: ",cossim_loss)

        #total loss
        alpha_pcc=1.0
        alpha_cossim=1.0
        total_loss=alpha_pcc*pcc_loss + alpha_cossim*cossim_loss
        #total_loss=jsd_loss
        #return total_loss,pcc_loss,cossim_loss,ssim_loss,rmse_loss
        return clusters_pcc_loss+clusters_cossim_loss

    def forward(self,preds,targets):
        #print(preds)
        # preds = (preds.T/preds.sum(axis=1)).T
        # preds=torch.where(torch.isnan(preds), torch.full_like(preds, 0), preds)
        # targets = (targets.T/targets.sum(axis=1)).T
        # targets=torch.where(torch.isnan(targets), torch.full_like(targets, 0), targets)

        total_loss=self.loss_function(preds,targets)
        return total_loss


def val_loss(preds,targets):
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_squared_error
    def ssim(im1,im2,M=1):
        im1, im2 = im1/im1.max(), im2/im2.max()
        mu1 = im1.mean()
        mu2 = im2.mean()
        sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
        sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
        sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
        k1, k2, L = 0.01, 0.03, M
        C1 = (k1*L) ** 2
        C2 = (k2*L) ** 2
        C3 = C2/2
        l12 = (2*mu1*mu2 + C1)/(mu1 ** 2 + mu2 ** 2 + C1)
        c12 = (2*sigma1*sigma2 + C2)/(sigma1 ** 2 + sigma2 ** 2 + C2)
        s12 = (sigma12 + C3)/(sigma1*sigma2 + C3)
        ssim = l12 * c12 * s12
        return ssim

    def rmse(x1,x2):
        #return mean_squared_error(x1,x2,squared=False)
        return np.sqrt(((x1-x2)**2).mean())
    def mae(x1,x2):
        return np.mean(np.abs(x1-x2))

    def compare_results(gd,result_list,metric='pcc',axis=1):
        if metric=='pcc':
            func = pearsonr
            r_ind = 0
        if metric=='mae':
            func = mae
            r_ind = None
        if metric=='jsd':
            func = jensenshannon
            r_ind = None
        if metric=='rmse':
            func = rmse
            r_ind = None
        if metric=='ssim':
            func = ssim
            r_ind = None
        
        if axis == 1:
            c_tensor = np.zeros(gd.shape[1])
            for i in range(gd.shape[1]):
                r = func(gd[:,i], result_list[:,i])
                if r_ind is not None:
                    r = r[r_ind]
                c_tensor[i]=r
        else:
            c_tensor = np.zeros(gd.shape[0])
            for i in range(gd.shape[0]):
                r = func(gd[i,:], result_list[i,:])
                if r_ind is not None:
                    r = r[r_ind]
                c_tensor[i]=r
        return c_tensor

    starmap_spots_pcc = compare_results(
        targets,
        preds,
        #columns = ['coGCN'],
        axis=0,
        metric='pcc'
    )

    starmap_spots_ssim = compare_results(
        targets,
        preds,
        #columns = ['coGCN'],
        axis=0,
        metric='ssim'
    )

    return np.nanmean(starmap_spots_pcc),np.nanmean(starmap_spots_ssim)



def impute_val_loss(preds,targets):
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import pearsonr
    def ssim(im1,im2,M=1):
        im1, im2 = im1/im1.max(), im2/im2.max()
        mu1 = im1.mean()
        mu2 = im2.mean()
        sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
        sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
        sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
        k1, k2, L = 0.01, 0.03, M
        C1 = (k1*L) ** 2
        C2 = (k2*L) ** 2
        C3 = C2/2
        l12 = (2*mu1*mu2 + C1)/(mu1 ** 2 + mu2 ** 2 + C1)
        c12 = (2*sigma1*sigma2 + C2)/(sigma1 ** 2 + sigma2 ** 2 + C2)
        s12 = (sigma12 + C3)/(sigma1*sigma2 + C3)
        ssim = l12 * c12 * s12
        return ssim

    def compare_results(gd,result_list,metric='pcc',axis=1):
        if metric=='pcc':
            func = pearsonr
            r_ind = 0
        if metric=='jsd':
            func = jensenshannon
            r_ind = None
        if metric=='ssim':
            func = ssim
            r_ind = None
        
        if axis == 1:
            c_tensor = np.zeros(gd.shape[1])
            for i in range(gd.shape[1]):
                r = func(gd[:,i], result_list[:,i])
                if r_ind is not None:
                    r = r[r_ind]
                c_tensor[i]=r
        else:
            c_tensor = np.zeros(gd.shape[0])
            for i in range(gd.shape[0]):
                r = func(gd[i,:], result_list[i,:])
                if r_ind is not None:
                    r = r[r_ind]
                c_tensor[i]=r
        return c_tensor

    starmap_spots_pcc = compare_results(
        targets,
        preds,
        #columns = ['coGCN'],
        axis=1,
        metric='pcc'
    )

    starmap_spots_ssim = compare_results(
        targets,
        preds,
        #columns = ['coGCN'],
        axis=1,
        metric='ssim'
    )

    starmap_spots_jsd = compare_results(
        targets,
        preds,
        #columns = ['coGCN'],
        axis=1,
        metric='jsd'
    )

    return np.nanmean(starmap_spots_pcc),np.nanmean(starmap_spots_ssim),np.nanmean(starmap_spots_jsd)
