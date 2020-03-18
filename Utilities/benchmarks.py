# This script contains all benchmarks for sum-rate and log-utility optimization
# for the work "Spatial Deep Learning for Wireless Scheduling",
# available at https://ieeexplore.ieee.org/document/8664604.

# For any reproduce, further research or development, please kindly cite our JSAC journal paper:
# @Article{spatial_learn,
#    author = "W. Cui and K. Shen and W. Yu",
#    title = "Spatial Deep Learning for Wireless Scheduling",
#    journal = "{\it IEEE J. Sel. Areas Commun.}",
#    year = 2019,
#    volume = 37,
#    issue = 6,
#    pages = "1248-1261",
#    month = "June",
# }

import numpy as np
import FPLinQ
import helper_functions
from scipy.optimize import linprog


# Wrapper function to call FPLinQ optimizer
def FP(general_para, gains, weights):
    n_layouts, N, _ = np.shape(gains)
    assert N==general_para.n_links
    assert np.shape(weights)==(n_layouts, N)
    FP_allocs = FPLinQ.FP_optimize(general_para, gains, weights)
    FP_schedules = (np.sqrt(FP_allocs) > 0.5).astype(int)
    return FP_schedules

def Strongest_Link_Scheduling(general_para, gains_diagonal):
    N, n_layouts = general_para.n_links, np.shape(gains_diagonal)[0]
    assert np.shape(gains_diagonal)==(n_layouts, N)
    strongest_links = np.argmax(gains_diagonal, axis=1)
    allocs = np.zeros([n_layouts, N])
    allocs[np.arange(n_layouts), strongest_links] = 1
    return allocs

def Max_Weight_Scheduling(general_para, proportional_fairness_weights):
    N, n_layouts = general_para.n_links, np.shape(proportional_fairness_weights)[0]
    assert np.shape(proportional_fairness_weights)==(n_layouts, N)
    allocs = np.zeros([n_layouts, N])
    max_weight_links = np.argmax(proportional_fairness_weights, axis=1)
    allocs[np.arange(n_layouts), max_weight_links] = 1
    return allocs

# greedy scheduling with weighted version: O(N^2) implementation
def Greedy_Scheduling(general_para, gains_diagonal, gains_nondiagonal, prop_weights):
    n_layouts, N = np.shape(gains_diagonal)
    assert np.shape(prop_weights) == (n_layouts, N)
    SNRS = gains_diagonal * general_para.tx_power / general_para.output_noise_power
    direct_rates = general_para.bandwidth * np.log2(1 + SNRS / general_para.SNR_gap)  # layouts X N; O(N) computation complexity
    sorted_links_indices = np.argsort(prop_weights * direct_rates, axis=1)
    allocs = np.zeros([n_layouts, N])
    previous_weighted_sum_rates = np.zeros([n_layouts])
    for j in range(N - 1, -1, -1):
        # schedule the ith shortest links
        allocs[np.arange(n_layouts), sorted_links_indices[:, j]] = 1
        rates = helper_functions.compute_rates(general_para, allocs, gains_diagonal, gains_nondiagonal)
        weighted_sum_rates = np.sum(rates * prop_weights, axis=1)  # (number of layouts,)
        # schedule the ith shortest pair for samples that have sum rate improved
        allocs[np.arange(n_layouts), sorted_links_indices[:, j]] = (weighted_sum_rates > previous_weighted_sum_rates).astype(int)
        previous_weighted_sum_rates = np.maximum(weighted_sum_rates, previous_weighted_sum_rates)
    return allocs