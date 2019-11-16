import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap

sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

#Set circular color map
n = 1000
sns.set_palette(sns.color_palette("husl", n))
dx = 2*np.pi/n

#Distribute data on a circle non-uniformly
x1 = np.random.normal(0*np.pi/2, np.pi, 100)
x2 = np.random.normal(1*np.pi/2, np.pi, 100)
x3 = np.random.normal(2*np.pi/2, np.pi, 100)
x4 = np.random.normal(3*np.pi/2, np.pi, 100)

x = np.concatenate((x1, x2, x3, x4), axis=None)
x_colors = np.mod(np.floor(x/dx), n)
X = np.array([np.cos(x), np.sin(x)]).transpose()

#Plot circle
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X[:, 0], X[:, 1],  c=[sns.color_palette()[i] for i in x_colors.astype(int)])

#Plot uniform circle
fit = umap.UMAP(min_dist=.1, n_neighbors=15)
u = fit.fit_transform(X)
fig_umap = plt.figure()
ax_umap = fig_umap.add_subplot(111)
ax_umap.scatter(u[:,0], u[:,1], c=[sns.color_palette()[i] for i in x_colors.astype(int)])
