import numpy as np
from scipy.stats import gaussian_kde

def get_kde_from_saliency(saliency_map, bw_method=0.1):
    h, w = saliency_map.shape
    x_flat = np.arange(w)
    y_flat = np.arange(h)
    if np.any(saliency_map != 0):
        x_weights = np.sum(saliency_map, axis=0) + 1e-10
        y_weights = np.sum(saliency_map, axis=1) + 1e-10
        kde_x = gaussian_kde(x_flat, weights=x_weights, bw_method="scott")
        kde_y = gaussian_kde(y_flat, weights=y_weights, bw_method="scott")
        kde_x.set_bandwidth(bw_method=bw_method)
        kde_y.set_bandwidth(bw_method=bw_method)
        kde_x_est = kde_x.evaluate(x_flat)
        kde_y_est = kde_y.evaluate(y_flat)
        return kde_x_est, kde_y_est
    else:
        kde_x_est = np.ones(w, dtype=float)/w
        kde_y_est = np.ones(h, dtype=float)/h
        return kde_x_est, kde_y_est


def calculate_spectral_flatness(kde_values):
    x = np.array(kde_values)
    x = np.clip(x, 1e-10, None)
    arithmetic_mean = np.mean(x)
    geometric_mean = np.exp(np.mean(np.log(x)))
    spec_flat = geometric_mean / arithmetic_mean
    if np.isnan(spec_flat) or np.isinf(spec_flat):
        return 1.0
    else:
        return spec_flat
    