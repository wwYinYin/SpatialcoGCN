# SpatialcoGCN: deconvolution and spatial information–aware simulation of spatial transcriptomics data via deep graph co-embedding
Spatial transcriptomics (ST) data has emerged as a pivotal approach to comprehending the function and interplay of cells within intricate tissues. Nevertheless, analyses of ST data are restricted by the low spatial resolution and limited number of RNA transcripts that can be detected for several popular ST techniques. In addition, the lack of spatial information-aware simulation data also hindered the development of ST analysis methods. In this study, we propose that both of the above issues can be significantly improved by introducing deep graph co-embedding framework. More specifically, we firstly present a graph deep learning model termed SpatialcoGCN that leverages single-cell data to deconvolve the cell mixtures in spatial data. SpatialcoGCN is a self-supervised framework that links scRNA-seq and ST data through co-embedding without requirement of pre-training. Evaluations of SpatialcoGCN on a series of simulated ST data and two real ST datasets from developing human heart and mouse brain suggest that SpatialcoGCN could outperform other state-of-the-art cell type deconvolution methods in estimating per spot cell composition. Moreover, SpatialcoGCN could also recover the spatial distribution of transcripts that are not detected by the ST data with competitive accuracy. With similar co-embedding framework, we further established a spatial information-aware ST data simulation method SpatialcoGCN-Sim. SpatialcoGCN-Sim could generate simulated ST data with high similarity to real datasets, which would serve as a helpful reference date for further development of ST data analysis method. Together, our approaches provide efficient tools for studying the spatial organization of heterogeneous cells within complex tissues.
## Install Guidelines
* We recommend you to use an Anaconda virtual environment. Install PyTorch >= 1.13 according to your GPU driver and Python >= 3.9, and run：

```
pip install -r requirements.txt
```
## Data preparation
* Download the simulated data from [SpatialcoGCN_data](https://figshare.com/articles/dataset/SpatialcoGCN_data/22682611) 
* Then put the downloaded dataset under the data directory
  ```
  -data\
        SimualtedSpatialData
        SpatialcoGCN-SimData
  ```
## Tutorial
For deconvolution:
  ```
  run_deconvolution.ipynb
  ```

SpatialcoGCN-Sim simulates ST data with awareness of spatial topology
  ```
  simulated_data.ipynb
  ```

For recoverring the undetected genes in enhancement of spatial transcriptomics
  ```
  run_impute.ipynb
  ```
For real ST data (the development of human heart and mouse brain) deconvolution:
  ```
  run_realdata.ipynb
  ```
  
## visualization
Reproduce the Figure 4 and Figure 5 of the paper:
  ```
  plot_deconvolution.ipynb
  ```
Reproduce the Figure 6 of the paper:
  ```
  humanheart.ipynb
  ```
  
Reproduce the Figure 7 of the paper:
  ```
  mousebrain.ipynb
  ```
  
Reproduce the Figure 8 of the paper:
  ```
  plot_impute.ipynb
  ```

## Citation
If you use SpatialcoGCN, please cite:
SpatialcoGCN: deconvolution and spatial information–aware simulation of spatial transcriptomics data via deep graph co-embedding (Briefings In Bioinformatics)  
Wang Yin, You Wan, Yuan Zhou  
https://academic.oup.com/bib/article/25/3/bbae130/7638268?login=true
