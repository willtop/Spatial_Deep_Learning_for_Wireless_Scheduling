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
        if ((i + 1) * 100 / n_timeSlots % 50 == 0):
            print("[Greedy Log Util] At {}/{} time slots...".format(i + 1, n_timeSlots))
        allocs = benchmarks.greedy_scheduling(general_para, directLink_channel_losses, crossLink_channel_losses,
                                              proportional_fairness_weights)
        rates = helper_functions.compute_rates(general_para, allocs, directLink_channel_losses, crossLink_channel_losses)
        allocs_all_timeSlots.append(allocs)
        rates_all_timeSlots.append(rates)
        proportional_fairness_weights = update_proportional_fairness_weights(general_para, rates, proportional_fairness_weights)
    allocs_all_timeSlots = np.transpose(np.array(allocs_all_timeSlots), (1, 0, 2))
    rates_all_timeSlots = np.transpose(np.array(rates_all_timeSlots), (1, 0, 2))
    assert np.shape(allocs_all_timeSlots) == np.shape(rates_all_timeSlots) == (n_layouts, n_timeSlots, N)

    print("All Active Log Utility Optimization")
    n_layouts, N = np.shape(directLink_channel_losses)
    allocs = np.ones([n_layouts, N]).astype(float)
    rates = helper_functions.compute_rates(general_para, allocs, directLink_channel_losses, crossLink_channel_losses)
    allocs_all_timeSlots = np.tile(np.expand_dims(allocs, axis=0), (n_timeSlots, 1, 1))
    rates_all_timeSlots = np.tile(np.expand_dims(rates, axis=0), (n_timeSlots, 1, 1))
    allocs_all_timeSlots = np.transpose(np.array(allocs_all_timeSlots), (1, 0, 2))
    rates_all_timeSlots = np.transpose(np.array(rates_all_timeSlots), (1, 0, 2))
    assert np.shape(allocs_all_timeSlots) == np.shape(rates_all_timeSlots) == (
    n_layouts, n_timeSlots, N)

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


    print("<<<<<<<<<<<<<<<<EVALUATION>>>>>>>>>>>>>>>>>")
    print("[Percentages of Scheduled Links Over All Time Slots] ")
    for method_key in allocs_all_methods.keys():
        print("[{}]: {}%; ".format(method_key, round(np.mean(allocs_all_methods[method_key]) * 100, 2)), end="")
    print("\n")

    # Sum log mean rates evaluation
    print("----------------------------------------------------------------")
    print("[Sum Log Mean Rates (Mean over {} layouts)]:".format(n_layouts))
    all_sum_log_mean_rates = dict()
    all_link_mean_rates = dict()
    global_max_mean_rate = 0 # for plotting upperbound
    for method_key in allocs_all_methods.keys():
        link_mean_rates = np.mean(rates_all_methods[method_key], axis=1); assert np.shape(link_mean_rates) == (test_layouts, N), "Wrong shape: {}".format(np.shape(link_mean_rates))
        all_link_mean_rates[method_key] = link_mean_rates.flatten() # (test_layouts X N, )
        global_max_mean_rate = max(global_max_mean_rate, np.max(link_mean_rates))
        all_sum_log_mean_rates[method_key] = np.mean(np.sum(np.log(link_mean_rates/1e6 + 1e-5),axis=1))
    for method_key in allocs_all_methods.keys():
        print("[{}]: {}; ".format(method_key, all_sum_log_mean_rates[method_key]), end="")
    print("\n")

    print("[Bottom 5-Percentile Mean Rate (Aggregate over all layouts)]:")
    for method_key in allocs_all_methods.keys():
        meanR_5pert = np.percentile((all_link_mean_rates[method_key]).flatten(), 5)
        print("[{}]: {}; ".format(method_key, meanR_5pert), end="")
    print("\n")

    # Produce the CDF plot for mean rates achieved by single links
    if(formal_CDF_legend_option):
        line_styles = dict()
        line_styles["Deep Learning"] = 'r-'
        line_styles["FP"] = 'g:'
        line_styles["Weighted Greedy"] = 'm-.'
        line_styles["Max Weight Only"] = 'b--'
        line_styles["All Active"] = 'c-.'
        line_styles["Random"] = 'y--'
    fig = plt.figure(); ax = fig.gca()
    plt.xlabel("Mean Rate for each link (Mbps)")
    plt.ylabel("Cumulative Distribution Function")
    plt.grid(linestyle="dotted")
    ax.set_xlim(left=0, right=0.45*global_max_mean_rate/1e6)
    ax.set_ylim(ymin=0)
    if(formal_CDF_legend_option):
        allocs_keys_ordered = ["Deep Learning", "FP", "Weighted Greedy", "Max Weight Only", "All Active", "Random"]
        for method_key in allocs_keys_ordered:
            if(method_key not in all_link_mean_rates.keys()):
                print("[{}] Allocation not computed! Skipping...".format(method_key))
                continue
            plt.plot(np.sort(all_link_mean_rates[method_key])/1e6, np.arange(1, test_layouts*N + 1) / (test_layouts*N), line_styles[method_key], label=method_key)
    else:
        for method_key in all_link_mean_rates.keys():
            plt.plot(np.sort(all_link_mean_rates[method_key])/1e6, np.arange(1, test_layouts*N + 1) / (test_layouts*N), label=method_key)
    plt.legend()
    plt.show()
    print("----------------------------------------------------------------")

    # Plot scheduling and subsets sent as scheduling candidates
    print("Plotting subset selection for each scheduling")
    v_layout = np.random.randint(low=0, high=test_layouts)
    v_layout = 5
    # Form a plot for every method having binary weights subset links scheduling
    for method_key in all_subsets.keys():
        print("[subsets_select_plotting] Plotting {} allocations...".format(method_key))
        v_subsets = all_subsets[method_key][v_layout]
        v_allocs = allocs_all_methods[method_key][v_layout]
        v_locations = test_locations[v_layout]
        fig = plt.figure(); plt.title("{} for layout #{}".format(method_key, v_layout))
        ax = fig.gca(); ax.set_xticklabels([]); ax.set_yticklabels([])
        for i in range(24):  # visualize first several time steps for each layout
            ax = fig.add_subplot(4, 6, i + 1); ax.set_xticklabels([]); ax.set_yticklabels([])
            tx_locs = v_locations[:, 0:2]; rx_locs = v_locations[:, 2:4]
            plt.scatter(tx_locs[:, 0], tx_locs[:, 1], c='r', s=2); plt.scatter(rx_locs[:, 0], rx_locs[:, 1], c='b', s=2)
            for j in range(N):  # plot both subsets and activated links (assume scheduling outputs)
                if v_subsets[i][j] == 1:
                    plt.plot([tx_locs[j, 0], rx_locs[j, 0]], [tx_locs[j, 1], rx_locs[j, 1]], 'b', linewidth=2.2, alpha=0.25)
                if v_allocs[i][j] == 1:
                    plt.plot([tx_locs[j, 0], rx_locs[j, 0]], [tx_locs[j, 1], rx_locs[j, 1]], 'r', linewidth=0.8)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()

    # Sequential Plotting
    # plot for one randomly selected layout over all methods
    print("Plotting sequential plotting of schedulings...")
    for method_key in allocs_all_methods.keys():
        if (method_key in ["All Active", "Random"]):
            continue  # Don't plot for these trivial allocations
        print("[sequential_timeslots_plotting] Plotting {} allocations...".format(method_key))
        v_allocs = allocs_all_methods[method_key][v_layout]
        v_locations = test_locations[v_layout]
        fig = plt.figure(); plt.title("{} for layout #{}".format(method_key, v_layout))
        ax = fig.gca(); ax.set_xticklabels([]); ax.set_yticklabels([])
        for i in range(24): # visualize first several time steps for each layout
            ax = fig.add_subplot(4, 6, i+1); ax.set_xticklabels([]); ax.set_yticklabels([])
            v_allocs_oneslot = v_allocs[i]
            tx_locs = v_locations[:, 0:2];  rx_locs = v_locations[:, 2:4]
            plt.scatter(tx_locs[:, 0], tx_locs[:, 1], c='r', s=3); plt.scatter(rx_locs[:, 0], rx_locs[:, 1], c='b', s=3)
            for j in range(N):  # plot all activated links
                line_color = 1-v_allocs_oneslot[j]
                if line_color==0:
                    line_color = 0.0 # deal with 0 formatting error problem
                plt.plot([tx_locs[j, 0], rx_locs[j, 0]], [tx_locs[j, 1], rx_locs[j, 1]], '{}'.format(line_color))# have to do 1 minus since the smaller the number the darker it gets
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()

    print("Script Completed Successfully!")
