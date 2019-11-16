import numpy as np

def draw_umap(n_neighbors=20, min_dist=0.1, n_components=3, metric='precomputed', title='',d=1):
    fit = umap.UMAP(
        n_neighbors = n_neighbors,
        min_dist = min_dist,
        n_components = n_components,
        metric = metric,
        )

    u = fit.fit_transform(d)
    fig_umap = plt.figure()
    if n_components == 1:
        ax_umap = fig_umap.add_subplot(111)
        ax_umap.scatter(u[:,0], range(len(u)))
    if n_components == 2:
        ax_umap = fig_umap.add_subplot(111)
        ax_umap.scatter(u[:,0], u[:,1])
    if n_components == 3:
        ax_umap = fig_umap.add_subplot(111, projection='3d')
        ax_umap.scatter(u[:,0], u[:,1], u[:,2], s=100)
    plt.title(title, fontsize=18)