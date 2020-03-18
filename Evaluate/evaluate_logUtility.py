# This repository contains the script for log-utility evaluation
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
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
sys.path.append("../Neural_Network_Model/")
import Convolutional_Neural_Network_Model
sys.path.append("../Utilities/")
import general_parameters
import benchmarks
import helper_functions


INCLUDE_FAST_FADING = False
n_timeSlots = 500

def update_proportional_fairness_weights(weights, rates):
    alpha = 0.95
    return 1 / (alpha / weights + (1 - alpha) * rates)

# Find binary approximation of importance weights parallelly over multiple layouts
def binarize_proportional_fairness_weights(general_para, weights):
    N = general_para.n_links
    n_layouts = np.shape(weights)[0]
    assert np.shape(weights) == (n_layouts, N)
    sorted_indices = np.argsort(weights, axis=1)
    weights_normalized = weights / np.linalg.norm(weights,axis=1,keepdims=True) # normalize to l2 norm 1
    # initialize variables
    binary_weights = np.zeros([n_layouts, N])
    max_dot_product = np.zeros(n_layouts)
    # use greedy to activate one at a time
    for i in range(N-1, -1, -1):
        binary_weights[np.arange(n_layouts), sorted_indices[:,i]] = 1
        binary_weights_normalized = binary_weights/np.linalg.norm(binary_weights,axis=1,keepdims=True)
        current_dot_product = np.einsum('ij,ij->i', weights_normalized, binary_weights_normalized)
        binary_weights[np.arange(n_layouts), sorted_indices[:,i]] = (current_dot_product >= max_dot_product).astype(int)
        max_dot_product = np.maximum(max_dot_product, current_dot_product)
    return binary_weights

if(__name__ =='__main__'):
    general_para = general_parameters.parameters()
    N = general_para.n_links
    print("[Evaluate Log-Utility] Evaluate Setting: ", general_para.setting_str)
    layouts = np.load("../Data/layouts_{}.npy".format(general_para.setting_str))
    path_losses = np.load("../Data/path_losses_{}.npy".format(general_para.setting_str))
    if (INCLUDE_FAST_FADING):
        print("Testing under CSI including fast fading realizations...")
        channel_losses = helper_functions.add_shadowing(path_losses)
        channel_losses = helper_functions.add_fast_fading(channel_losses)
    else:
        print("Testing under CSI consists of path losses only...")
        channel_losses = path_losses
    n_layouts = np.shape(layouts)[0]
    assert np.shape(layouts) == (n_layouts, N, 4)
    assert np.shape(channel_losses) == (n_layouts, N, N)
    directLink_channel_losses = helper_functions.get_directLink_channel_losses(channel_losses)
    crossLink_channel_losses = helper_functions.get_crossLink_channel_losses(channel_losses)

    allocs_all_methods = {}
    rates_all_methods = {}
    superSets_all_methods = {} # only for FP and neural network

    print("FP Log Utility Optimization...")
    allocs_all_timeSlots = []
    rates_all_timeSlots = []
    proportional_fairness_weights = np.ones([n_layouts, N])
    for i in range(n_timeSlots):
        if ((i + 1) * 100 / n_timeSlots % 25 == 0):
            print("At {}/{} time slots...".format(i + 1, n_timeSlots))
        allocs = benchmarks.FP(general_para, channel_losses, proportional_fairness_weights)
        rates = helper_functions.compute_rates(general_para, allocs, directLink_channel_losses, crossLink_channel_losses)
        allocs_all_timeSlots.append(allocs)
        rates_all_timeSlots.append(rates)
        proportional_fairness_weights = update_proportional_fairness_weights(proportional_fairness_weights, rates)
    allocs_all_timeSlots = np.transpose(np.array(allocs_all_timeSlots), (1, 0, 2))
    rates_all_timeSlots = np.transpose(np.array(rates_all_timeSlots), (1, 0, 2))
    assert np.shape(allocs_all_timeSlots) == np.shape(rates_all_timeSlots) == (n_layouts, n_timeSlots, N)
    allocs_all_methods["FP"] = allocs_all_timeSlots
    rates_all_methods["FP"] = rates_all_timeSlots

    print("FP Not Knowing Fading Log Utility Optimization...")
    allocs_all_timeSlots = []
    rates_all_timeSlots = []
    proportional_fairness_weights = np.ones([n_layouts, N])
    for i in range(n_timeSlots):
        if ((i + 1) * 100 / n_timeSlots % 25 == 0):
            print("At {}/{} time slots...".format(i + 1, n_timeSlots))
        allocs = benchmarks.FP(general_para, path_losses, proportional_fairness_weights)
        rates = helper_functions.compute_rates(general_para, allocs, directLink_channel_losses, crossLink_channel_losses)
        allocs_all_timeSlots.append(allocs)
        rates_all_timeSlots.append(rates)
        proportional_fairness_weights = update_proportional_fairness_weights(proportional_fairness_weights, rates)
    allocs_all_timeSlots = np.transpose(np.array(allocs_all_timeSlots), (1, 0, 2))
    rates_all_timeSlots = np.transpose(np.array(rates_all_timeSlots), (1, 0, 2))
    assert np.shape(allocs_all_timeSlots) == np.shape(rates_all_timeSlots) == (n_layouts, n_timeSlots, N)
    allocs_all_methods["FP Not Knowing Fading"] = allocs_all_timeSlots
    rates_all_methods["FP Not Knowing Fading"] = rates_all_timeSlots

    print("FP with Binary Re-weighting Log Utility Optimization...")
    allocs_all_timeSlots = []
    rates_all_timeSlots = []
    superSets_all_timeSlots = []
    proportional_fairness_weights = np.ones([n_layouts, N])
    proportional_fairness_weights_binary = np.ones([n_layouts, N])
    for i in range(n_timeSlots):
        if ((i + 1) * 100 / n_timeSlots % 25 == 0):
            print("At {}/{} time slots...".format(i + 1, n_timeSlots))
        allocs = benchmarks.FP(general_para, channel_losses, proportional_fairness_weights_binary, scheduling_output=True)
        rates = helper_functions.compute_rates(general_para, allocs, directLink_channel_losses, crossLink_channel_losses)
        allocs_all_timeSlots.append(allocs)
        rates_all_timeSlots.append(rates)
        superSets_all_timeSlots.append(proportional_fairness_weights_binary)
        proportional_fairness_weights = update_proportional_fairness_weights(proportional_fairness_weights, rates)
        proportional_fairness_weights_binary = binarize_proportional_fairness_weights(general_para, proportional_fairness_weights)
    allocs_all_timeSlots = np.transpose(np.array(allocs_all_timeSlots), (1, 0, 2))
    rates_all_timeSlots = np.transpose(np.array(rates_all_timeSlots), (1, 0, 2))
    superSets_all_timeSlots = np.transpose(np.array(superSets_all_timeSlots), (1, 0, 2))
    assert np.shape(allocs_all_timeSlots) == np.shape(rates_all_timeSlots) == np.shape(superSets_all_timeSlots) == (n_layouts, n_timeSlots, N)
    allocs_all_methods["FP Binary Re-Weighting"] = allocs_all_timeSlots
    rates_all_methods["FP Binary Re-Weighting"] = rates_all_timeSlots
    superSets_all_methods["FP Binary Re-Weighting"] = superSets_all_timeSlots

    print("Weighted Greedy Log Utility Optimization")
    n_layouts, N = np.shape(directLink_channel_losses)
    allocs_all_timeSlots = []
    rates_all_timeSlots = []
    proportional_fairness_weights = np.ones([n_layouts, N])
    for i in range(n_timeSlots):
        if ((i + 1) * 100 / n_timeSlots % 25 == 0):
            print("At {}/{} time slots...".format(i + 1, n_timeSlots))
        allocs = benchmarks.greedy_scheduling(general_para, directLink_channel_losses, crossLink_channel_losses, proportional_fairness_weights)
        rates = helper_functions.compute_rates(general_para, allocs, directLink_channel_losses, crossLink_channel_losses)
        allocs_all_timeSlots.append(allocs)
        rates_all_timeSlots.append(rates)
        proportional_fairness_weights = update_proportional_fairness_weights(general_para, rates, proportional_fairness_weights)
    allocs_all_timeSlots = np.transpose(np.array(allocs_all_timeSlots), (1, 0, 2))
    rates_all_timeSlots = np.transpose(np.array(rates_all_timeSlots), (1, 0, 2))
    assert np.shape(allocs_all_timeSlots) == np.shape(rates_all_timeSlots) == (n_layouts, n_timeSlots, N)
    allocs_all_methods["Weighted Greedy"] = allocs_all_timeSlots
    rates_all_methods["Weighted Greedy"] = rates_all_timeSlots

    print("Max Weight Link Only Log Utility Optimization")
    n_layouts, N = np.shape(directLink_channel_losses)
    allocs_all_timeSlots = []
    rates_all_timeSlots = []
    proportional_fairness_weights = np.ones([n_layouts, N])
    for i in range(n_timeSlots):
        allocs = benchmarks.Max_Weight_scheduling(general_para, proportional_fairness_weights)
        rates = helper_functions.compute_rates(general_para, allocs, directLink_channel_losses, crossLink_channel_losses)
        allocs_all_timeSlots.append(allocs)
        rates_all_timeSlots.append(rates)
        proportional_fairness_weights = update_proportional_fairness_weights(general_para, rates, proportional_fairness_weights)
    allocs_all_timeSlots = np.transpose(np.array(allocs_all_timeSlots), (1, 0, 2))
    rates_all_timeSlots = np.transpose(np.array(rates_all_timeSlots), (1, 0, 2))
    assert np.shape(allocs_all_timeSlots) == np.shape(rates_all_timeSlots) == (n_layouts, n_timeSlots, N)
    allocs_all_methods["Max Weight"] = allocs_all_timeSlots
    rates_all_methods["Max Weight"] = rates_all_timeSlots

    print("All Active Log Utility Optimization")
    n_layouts, N = np.shape(directLink_channel_losses)
    allocs = np.ones([n_layouts, N]).astype(float)
    rates = helper_functions.compute_rates(general_para, allocs, directLink_channel_losses, crossLink_channel_losses)
    allocs_all_timeSlots = np.tile(np.expand_dims(allocs, axis=0), (n_timeSlots, 1, 1))
    rates_all_timeSlots = np.tile(np.expand_dims(rates, axis=0), (n_timeSlots, 1, 1))
    allocs_all_timeSlots = np.transpose(np.array(allocs_all_timeSlots), (1, 0, 2))
    rates_all_timeSlots = np.transpose(np.array(rates_all_timeSlots), (1, 0, 2))
    assert np.shape(allocs_all_timeSlots) == np.shape(rates_all_timeSlots) == (n_layouts, n_timeSlots, N)
    allocs_all_methods["All Active"] = allocs_all_timeSlots
    rates_all_methods["All Active"] = rates_all_timeSlots

    print("Random Scheduling Log Utility Optimization")
    n_layouts, N = np.shape(directLink_channel_losses)
    allocs_all_timeSlots = np.random.randint(2, size=(n_timeSlots, n_layouts, N)).astype(float)
    rates_all_timeSlots = []
    for i in range(n_timeSlots):
        rates_oneslot = helper_functions.compute_rates(general_para, allocs_all_timeSlots[i], directLink_channel_losses, crossLink_channel_losses)
        rates_all_timeSlots.append(rates_oneslot)
    allocs_all_timeSlots = np.transpose(np.array(allocs_all_timeSlots), (1, 0, 2))
    rates_all_timeSlots = np.transpose(np.array(rates_all_timeSlots), (1, 0, 2))
    assert np.shape(allocs_all_timeSlots) == np.shape(rates_all_timeSlots) == (n_layouts, n_timeSlots, N)
    allocs_all_methods["Random"] = allocs_all_timeSlots
    rates_all_methods["Random"] = rates_all_timeSlots

    print("<<<<<<<<<<<<<<<<EVALUATION>>>>>>>>>>>>>>>>>")
    print("[Percentages of Scheduled Links Over All Time Slots] ")
    for method_key in allocs_all_methods.keys():
        print("[{}]: {}%; ".format(method_key, round(np.mean(allocs_all_methods[method_key]) * 100, 2)), end="")
    print("\n")

    print("[Sum Log Mean Rates in Mbps (Avg over {} layouts)]:".format(n_layouts))
    link_mean_rates_all_methods = {}
    global_max_mean_rate = 0 # for plotting upperbound
    for method_key in allocs_all_methods.keys():
        link_mean_rates = np.mean(rates_all_methods[method_key], axis=1)
        assert np.shape(link_mean_rates) == (n_layouts, N), "Wrong shape: {}".format(np.shape(link_mean_rates))
        link_mean_rates_all_methods[method_key] = link_mean_rates.flatten() # (test_layouts X N, )
        global_max_mean_rate = max(global_max_mean_rate, np.max(link_mean_rates))
        log_utilities = np.mean(np.sum(np.log(link_mean_rates/1e6 + 1e-9),axis=1))
        print("[{}]: {}; ".format(method_key, log_utilities, end=""))
    print("\n")

    print("[Bottom 5-Percentile Mean Rate over all links among all layouts]:")
    for method_key in allocs_all_methods.keys():
        mean_rate_5pert = np.percentile((link_mean_rates_all_methods[method_key]).flatten(), 5)
        print("[{}]: {}; ".format(method_key, mean_rate_5pert), end="")
    print("\n")

    # Produce the CDF plot for each link's mean rate aggregated
    line_styles = dict()
    line_styles["Deep Learning"] = 'r-'
    line_styles["FP"] = 'g-.'
    line_styles["FP Not Knowing Small Scale Fading"] = "b:"
    line_styles["FP Binary Re-Weighting"] = 'm-.'
    line_styles["Weighted Greedy"] = 'k--'
    line_styles["Max Weight Only"] = 'c-.'
    line_styles["All Active"] = 'y-'
    line_styles["Random"] = 'm--'
    fig = plt.figure()
    ax = fig.gca()
    plt.xlabel("Mean Rate for each link (Mbps)")
    plt.ylabel("Cumulative Distribution Function")
    plt.grid(linestyle="dotted")
    ax.set_xlim(left=0, right=0.45*global_max_mean_rate/1e6)
    ax.set_ylim(ymin=0)
    for method_key in allocs_all_methods.keys():
        plt.plot(np.sort(link_mean_rates_all_methods[method_key])/1e6, np.arange(1, n_layouts*N + 1) / (n_layouts*N), line_styles[method_key], label=method_key)
    plt.legend()
    plt.show()

    print("Plotting sequential time slots scheduling, and where applicable, supersets...")
    layout_indices = np.random.randint(low=0, high=n_layouts-1, size=3)
    for layout_index in layout_indices:
        for method_key in allocs_all_methods.keys():
            if (method_key in ["All Active", "Random", "Max Weight Only"]):
                continue  # Don't plot for these trivial allocations
            plt.title("{} Sequential Scheduling on {}th Layout".format(method_key, layout_index))
            for i in range(1, 16):  # visualize first several time steps for each layout
                ax = fig.add_subplot(3, 5, i)
                ax.set_title("{}th TimeSlot".format(i))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                helper_functions.visualize_layout(ax, layouts[layout_index])
                if method_key in superSets_all_methods.keys():
                    helper_functions.visualize_superset_on_layout(ax, layouts[layout_index], superSets_all_methods[method_key][layout_index])
                helper_functions.visualize_schedules_on_layout(ax, layouts[layout_index], allocs_all_methods[method_key][layout_index])
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.show()

    print("Script Completed Successfully!")
