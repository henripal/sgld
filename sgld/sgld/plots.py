import matplotlib.pyplot as plt
import numpy as np

def plot_densities(data, n_indices, n_bins, alpha,
    delta=None,
    burnin=100):
    n = data.shape[0]
    indices = np.linspace(burnin, n-1, n_indices).astype(int)

    toplot = [data[i, :] for i in indices]

    color_idx = np.linspace(0, 1, n_indices)
    mn = np.infty
    mx =  0
    mean = 0

    if delta is None:
        for data in toplot:
            newmin = np.min(data)
            newmax = np.max(data)
            if newmin < mn: mn = newmin
            if newmax > mx: mx = newmax
    else:
        for data in toplot:
            mean += np.mean(data)
        mean = mean/len(toplot)
        mn = mean - delta/2
        mx = mean + delta/2




    for ci, data in zip(color_idx, toplot):
        x, y = np.histogram(data, range=(mn, mx),
        bins=n_bins)
#   x, y = np.histogram(toplot, bins=n_bins)
    # delta = y[-1] - y[0]


        plt.plot(y[:-1], x, color=plt.cm.RdBu(ci), alpha=alpha)
        
    plt.xlabel('Weight Value')
    plt.ylabel('Count')
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu)
    sm._A = []
    cbar = plt.colorbar(sm, ticks= [0.1, .9])
    cbar.ax.set_yticklabels(['early training', 'late training']) 
    plt.show()

    return mx - mn