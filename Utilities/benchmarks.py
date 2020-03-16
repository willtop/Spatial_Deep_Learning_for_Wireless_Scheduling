# all benchmarks for evaluating different objectives
import numpy as np
import FPLinQ
import utils
from scipy.optimize import linprog


# Wrapper function to call FPLinQ optimizer
def FP(general_para, gains, weights, scheduling_output=False):
    number_of_layouts, N, _ = np.shape(gains)
    assert N==general_para.number_of_links
    assert np.shape(weights)==(number_of_layouts, N)
    FP_allocs = FPLinQ.FP_optimize(general_para, gains, weights)
    if(scheduling_output):
        FP_allocs = (np.sqrt(FP_allocs) > 0.5).astype(int)
    return FP_allocs

def Strongest_Link(general_para, gains_diagonal):
    N, number_of_layouts = general_para.number_of_links, np.shape(gains_diagonal)[0]
    assert np.shape(gains_diagonal)==(number_of_layouts, N)
    strongest_links = np.argmax(gains_diagonal, axis=1)
    allocs = np.zeros([number_of_layouts, N])
    allocs[np.arange(number_of_layouts), strongest_links] = 1
    return allocs

def convex_Dinkelbach_onestep(general_para, gains_diagonal, gains_nondiagonal, y):
    N = general_para.number_of_links
    assert np.shape(gains_diagonal) == (N,)
    assert np.shape(gains_nondiagonal) == (N,N)
    # objective
    c = [0]*N+[-1]
    # constraint on min dummy variable
    A_ub = gains_nondiagonal*y-np.diag(gains_diagonal)
    A_ub = np.concatenate([A_ub, np.ones([N,1])],axis=1)
    b_ub = -np.ones([N,1])*(general_para.output_noise_power / general_para.tx_power)
    # constraint on power control variables 0~1
    bounds = []
    for i in range(N):
        bounds.append((0,1))
    bounds.append((0, None))
    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    x = res['x'][:-1]
    t = res['x'][-1]
    return x, t

# Taking one layout a time
def maxmin_Dinkelbach_solver(general_para, gains_diagonal, gains_nondiagonal):
    N = general_para.number_of_links
    assert np.shape(gains_diagonal) == (N, )
    assert np.shape(gains_nondiagonal) == (N, N)
    x = np.ones(N, dtype=np.float64)
    tolerance = 1e-12
    iteration_upperbound = 500
    iteration_count = 0
    while True:
        SINRs_numerators = x * gains_diagonal  # (N, )
        SINRs_denominators = np.squeeze(np.matmul(gains_nondiagonal, np.expand_dims(x, axis=-1))) + general_para.output_noise_power / general_para.tx_power  # (N, )
        SINRs = SINRs_numerators / SINRs_denominators  # (N, )
        min_SINR = np.min(SINRs)
        x, t = convex_Dinkelbach_onestep(general_para, gains_diagonal, gains_nondiagonal, min_SINR)
        iteration_count += 1
        if(t<=tolerance):
            whether_debug = False
            break
        if(iteration_count==iteration_upperbound):
            print("Completed {} iterations, the F(lambda) value: {}".format(iteration_count, t))
            whether_debug = True
            break
    return x, whether_debug

# Due to large computation time, try to load first if results have been computed
def MaxMin_Dinkelbach(general_para, gains_diagonal, gains_nondiagonal):
    print("==============================Max Min Dinkelbach Power Control==================")
    N, number_of_layouts = general_para.number_of_links, np.shape(gains_diagonal)[0]
    assert np.shape(gains_diagonal) == (number_of_layouts, N)
    assert np.shape(gains_nondiagonal) == (number_of_layouts, N, N)
    allocs_store_path = general_para.test_dir+general_para.file_names["MinMaxDinkelbach"]
    try:
        allocs = np.load(allocs_store_path)
    except:
        # for now, use GP to solve one layout at a time
        allocs = []
        for i in range(number_of_layouts):
            if ((i + 1) * 100 / number_of_layouts % 10 == 0):
                print("At {}/{} layouts.".format(i + 1, number_of_layouts))
            allocs_one_layout, whether_debug = maxmin_Dinkelbach_solver(general_para, gains_diagonal[i], gains_nondiagonal[i])
            if(whether_debug):
                print("Max Min Dinkelbach encountered one layout optimization non-converging!")
                print("Allocations: ", allocs_one_layout)
                SINRs_numerators = allocs_one_layout * gains_diagonal[i]  # N
                SINRs_denominators = np.squeeze(np.matmul(gains_nondiagonal[i], np.expand_dims(allocs_one_layout, axis=-1))) + general_para.output_noise_power / general_para.tx_power  # N
                SINRs = SINRs_numerators / SINRs_denominators  # N
                print("SINRs: ", SINRs)
            allocs.append(allocs_one_layout)
        allocs = np.array(allocs)
        assert np.shape(allocs)==(number_of_layouts, N)
        print("Saving computed Dinkelbach allocations at: ", allocs_store_path)
        np.save(allocs_store_path, allocs)
    else:
        print("Loaded existing allocations from {}".format(allocs_store_path))
    finally:
        assert np.shape(allocs)==(number_of_layouts, N)
        return allocs

def directLink_inverse_proportions(general_para, gains_diagonal):
    print("==============================Direct Link Inverse Proportions=====================================")
    N, number_of_layouts = general_para.number_of_links, np.shape(gains_diagonal)[0]
    assert np.shape(gains_diagonal) == (number_of_layouts, N)
    allocs = np.ones([number_of_layouts, N]) * np.min(gains_diagonal, axis=1, keepdims=True) / gains_diagonal
    return allocs

# greedy scheduling with weighted version: O(N^2) implementation
def greedy_scheduling(general_para, gains_diagonal, gains_nondiagonal, prop_weights):
    number_of_layouts, N = np.shape(gains_diagonal)
    assert np.shape(prop_weights) == (number_of_layouts, N)
    SNRS = gains_diagonal * general_para.tx_power / general_para.output_noise_power
    direct_rates = general_para.bandwidth * np.log2(1 + SNRS / general_para.SNR_gap)  # layouts X N; O(N) computation complexity
    sorted_links_indices = np.argsort(prop_weights * direct_rates, axis=1)
    allocs = np.zeros([number_of_layouts, N])
    previous_weighted_sum_rates = np.zeros([number_of_layouts])
    for j in range(N - 1, -1, -1):
        # schedule the ith shortest links
        allocs[np.arange(number_of_layouts), sorted_links_indices[:, j]] = 1
        rates = utils.compute_rates(general_para, allocs, gains_diagonal, gains_nondiagonal)
        weighted_sum_rates = np.sum(rates * prop_weights, axis=1)  # (number of layouts,)
        # schedule the ith shortest pair for samples that have sum rate improved
        allocs[np.arange(number_of_layouts), sorted_links_indices[:, j]] = (weighted_sum_rates > previous_weighted_sum_rates).astype(int)
        previous_weighted_sum_rates = np.maximum(weighted_sum_rates, previous_weighted_sum_rates)
    return allocs