import numpy as np
import torch as tc
import pandas as pd
dtype = tc.cuda.FloatTensor


def quantile_metric_gpu(x=tc.zeros([0]).type(dtype)):
    """This function computes a dissimilarity measure on a set of quantile normalized vectors based upon their\
   rank and variance amongst their ranked values"""
    # Compute dimensions
    n, m = x.shape

    # Initialize measure
    d = tc.zeros([n, n]).type(dtype)

    # Compute variance, mean, and dispersion over features
    var = x.var(1)
    mean = x.mean(1)
    disp = var / mean
    weights = tc.triu(disp * disp.reshape(-1, 1), diagonal=1)

    # Create thick x, each layer is x reordered by ith column
    z = tc.zeros([n, m, m]).type(dtype)
    for i in np.arange(m):
        z[:, :, i] = x[x[:, i].argsort(0), :]

    # Create 4-d tensor which compares ordered ith column to jth column with respect to the ordering
    index = (z[None, :, :, :] > z[:, None, :, :]).type(dtype)

    # Contract weight tensor with index tensor
    d = tc.einsum('klji,kl->ij', [index, weights])

    return d

#df = pd.read_csv('/home/razer/mydata.txt', delimiter = "\t",index_col=0)
#y = tc.tensor(df.values).type(dtype)
#y = y[[1,3,5,7,9,11,13,15,17],:]
#y = y[:,[1,3,5,7,9,11,13,15,17]]
#d = quantile_metric_gpu(y)

from scipy import stats

#Load data
df = pd.read_csv(filepath_or_buffer='~/GEO_DATA/GSE73072_raw.txt',delimiter='\t',index_col=0)
x = tc.tensor(df.values, device=0)

#Calculate dispersion ranks for each subject
subject_dict = dict()
subject_num  = dict()
subjects = set(sorted(gse.phenotype_data['characteristics_ch1.3.subject']))
for subject in subjects:
    tf = [(subject in gse.phenotype_data['characteristics_ch1.3.subject'][sample]) for sample in gse.gsms]
    x_sub = x[:, tf]
    disp = x_sub.var(1)/x_sub.mean(1)
    subject_dict[subject] = (-disp).argsort().cpu().numpy()

#Calculate top r genes for each study
r = 20
study_dict = dict()
study_num  = dict()
studies = ['RSV DEE1', 'H3N2 DEE2', 'H1N1 DEE3', 'H1N1 DEE 4', 'H3N2 DEE5', 'HRV UVA', 'HRV DUKE']
for study in studies:
    study_dict[study] = []
    for subject in subject_dict.keys():
        if study in subject:
            study_dict[study].append(subject_dict[subject][:r])
    study_dict[study] = np.array(study_dict[study])
    study_dict[study] = (-np.array([np.count_nonzero(study_dict[study] == i) for i in range(2886)])).argsort()
    study_dict[study] = study_dict[study][:r]


r = 10
x_rank = x.argsort(0)
x_top = x_rank[:r,:]


