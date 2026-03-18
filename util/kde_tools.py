import numpy as np
from scipy.stats import gaussian_kde

def get_kde_from_saliency(saliency_map, bw_method=0.1):
    h, w = saliency_map.shape
    x_flat = np.arange(w)
    y_flat = np.arange(h)
    kde_x = gaussian_kde(x_flat, weights=np.sum(saliency_map, axis=0), bw_method="scott")
    kde_y = gaussian_kde(y_flat, weights=np.sum(saliency_map, axis=1), bw_method="scott")
    kde_x.set_bandwidth(bw_method=bw_method)
    kde_y.set_bandwidth(bw_method=bw_method)
    kde_x_est = kde_x.evaluate(x_flat)
    kde_y_est = kde_y.evaluate(y_flat)
    return kde_x_est, kde_y_est
