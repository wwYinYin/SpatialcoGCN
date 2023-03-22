import numpy as np
import pandas as pd
from functools import partial
import scipy.stats as st
from scipy.spatial.distance import jensenshannon
import copy
from sklearn.model_selection import KFold
import matplotlib as mpl 
import matplotlib.pyplot as plt
import scanpy as sc
import warnings
warnings.filterwarnings('ignore')


def cal_ssim(im1,im2,M=1):
    assert im1.shape == im2.shape
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
    
class CalculateMeteics:
    def __init__(self, gd_results,coGCN_results, metric):
        self.gd_results = gd_results
        
        self.coGCN_results = coGCN_results

        self.metric = metric
        
    def SSIM(self, gt, pred):
        if gt.shape[0] == pred.shape[0]:
            result = pd.DataFrame()
            for i in range(gt.shape[0]):
                ssim = cal_ssim(gt[i,:], pred[i,:])
            
                ssim_df = pd.DataFrame(ssim, index=["SSIM"],columns=[str(i)])
                result = pd.concat([result, ssim_df],axis=1)
        else:
            print("columns error")
        return result
            
    def PCC(self, gt, pred, scale = None):
        if gt.shape[0] == pred.shape[0]:
            result = pd.DataFrame()
            for i in range(gt.shape[0]):
                pearsonr, _ = st.pearsonr(gt[i,:], pred[i,:])
                pearson_df = pd.DataFrame(pearsonr, index=["PCC"],columns=[str(i)])
                result = pd.concat([result, pearson_df],axis=1)
        else:
            print("columns error")
        return result
    
    def JS(self, gt, pred): 
        if gt.shape[0] == pred.shape[0]:
            result = pd.DataFrame()
            for i in range(gt.shape[0]):
                JS = jensenshannon(gt[i,:], pred[i,:])
                JS_df = pd.DataFrame(JS, index=["JS"],columns=[str(i)])
                result = pd.concat([result, JS_df],axis=1)
        else:
            print("columns error")
        return result
    
    def RMSE(self, gt, pred, scale = 'zscore'):
        if gt.shape[0] == pred.shape[0]:
            result = pd.DataFrame()
            for i in range(gt.shape[0]):
                RMSE = np.sqrt(((gt[i,:] - pred[i,:]) ** 2).mean())

                RMSE_df = pd.DataFrame(RMSE, index=["RMSE"],columns=[str(i)])
                result = pd.concat([result, RMSE_df],axis=1)
        else:
            print("columns error")
        return result       
        
    def compute_all(self):
        gt = self.gd_results
        pred = self.coGCN_results

        SSIM = self.SSIM(gt,pred)
        Pearson = self.PCC(gt, pred)
        JS = self.JS(gt, pred)
        RMSE = self.RMSE(gt, pred)
        
        result_all = pd.concat([Pearson, SSIM, RMSE, JS],axis=0)
        return result_all


def CalDataMetric(gd_results,coGCN_results):
    print ('We are calculating the : deconvolution matrix' + '\n')
    metric = ['PCC','SSIM','RMSE','JS']
    CM = CalculateMeteics(gd_results = gd_results, coGCN_results = coGCN_results, metric = metric)
    result_metric=CM.compute_all()

    return result_metric