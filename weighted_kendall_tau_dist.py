import numpy as np

def weighted_kendall_tau_distance( x=np.zeros([0]) ):
    """This function computes a dissimilarity measure on a set of quantile normalized vectors based upon their\
    rank and variance amongst their ranked values"""
    #Compute dimensions
    m = x.shape[0]
    n = x.shape[1]

    #Initialize measure
    d = np.zeros([n,n])

    #Compute variance over features
    var = x.var(1)

    #Create rank matrix
    rank = x.argsort(0)+1

    for i in np.arange(n):
        for j in np.arange(i+1,n):
            for k in np.arange(m):
                for l in np.arange(k+1,m)
                    if rank[k,i] > rank[l,j]:
                        d[i,j] += (var[k]*var[l])/(x[k,i]*x[l,j])
    return d