# This repository contains the script for sum-rate evaluation
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
# to silent tensorflow WARNING
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import matplotlib.pyplot as plt
import general_parameters
import helper_functions
import benchmarks
import time
# import functions for DL WMMSE modules
import sys
sys.path.append("../Neural_Network_Model/")
import Convolutional_Neural_Network_Model
import Direct_Allocs_BackProp


display_ratios_option = False
save_allocs_option = False
formal_CDF_legend_option = True


if(__name__ =='__main__'):
    general_para = general_parameters.parameters()
    N = general_para.n_links
    print("[Evaluate SumRate] Evaluate Setting: ", general_para.setting_str)
    layouts = np.load(general_para.data_dir + general_para.filenames["layouts"])
    path_losses = np.load(general_para.data_dir + general_para.filenames["path_losses"])
    channel_losses = np.load(general_para.data_dir + general_para.filenames["channel_losses"])
    n_layouts = np.shape(layouts)[0]
    assert np.shape(layouts) == (n_layouts, N, 4)
    assert np.shape(channel_losses) == (n_layouts, N, N)
    directLink_channel_losses = helper_functions.get_directLink_channel_losses(channel_losses)
    crossLink_channel_losses = helper_functions.get_crossLink_channel_losses(channel_losses)

    all_allocs = {}

    all_allocs["FP"] = benchmarks.FP(general_para, channel_losses, np.ones([n_layouts, N]))

    all_allocs["FP Not Knowing Small Scale Fading"] = benchmarks.FP(general_para, path_losses, np.ones([n_layouts, N]))

    all_allocs["Deep Learning"] = Convolutional_Neural_Network_Model.network_inference(general_para, time_complexity_option)

    if ("conv_net_sumrate_v9" in all_models):
        print("========================================Convolutional Neural Network Sum Rate V9==================================")
        method_key = "Neural Network V9"
        allocs, _ = Conv_Net_SumRate_V9.network_inference(general_para, time_complexity_option)
        assert np.shape(allocs) == (test_layouts, N), "Wrong shape: {}".format(np.shape(allocs))
        all_allocs[method_key] = allocs
        print("Successfully Computed {} for Testing Samples.".format(method_key))

    ###################################### Heuristics and Benchmarks ###################################
    # All activation benchmark
    if("all_active" in all_benchmarks):
        print("==================================All Activation Benchmark===============================")
        all_active_allocs = np.ones((test_layouts, N))
        all_allocs["All Active"] = all_active_allocs

    # Random scheduling allocation benchmark
    if("random_sch" in all_benchmarks):
        print("==================================Random Scheduling Allocation Benchmark==============================")
        random_allocs_scheduling_tmp = np.random.rand(test_layouts, N)
        random_allocs_scheduling = (random_allocs_scheduling_tmp>=0.5).astype(int)
        all_allocs["Random"] = random_allocs_scheduling

    # Greedy Heuristic Scheduling: O(N^2)
    if ("greedy_sch" in all_benchmarks):
        print("==================================Greedy Scheduling Heuristic==============================")
        method_key = "Greedy"
        save_path = general_para.data_folder + general_para.test_data_info['folder'] + "Greedy_{}_meters_{}_pairs_allocs.npy".format(general_para.field_length, N)
        # Test if greedy scheduling is already pre-computed
        try:
            allocs = np.load(save_path)
        except FileNotFoundError:
            print("Can't find pre-computed greedy allocation at {}! Compute from fresh...".format(save_path))
            compute_fresh = True
        else:
            all_allocs[method_key] = allocs
            print("Successfully loaded greedy allocation from {}!".format(save_path))
            compute_fresh = False
        if (compute_fresh):
            print("Start Computing Greedy Scheduling...")
            allocs, run_time = utils.greedy_sumrate(general_para, test_gains_diagonal, test_gains_nondiagonal)
            all_allocs[method_key] = allocs
            print("Finish {}. Total time: {} seconds, per layout: {} seconds".format(method_key, run_time, run_time / test_layouts))
            if(save_allocs_option):
                np.save(save_path, allocs)
                print("{} allocs saved at {}!".format(method_key, save_path))

    if("covariance_sampling_sch" in all_benchmarks):
        print("==========================Covariance Sampling Scheduling Benchmark====================")
        method_key = "Covariance Sampling"
        test_distances = np.load(data_dir+general_para.dists_file)
        allocs, run_time = utils.cov_vectors_sampling(test_distances)
        all_allocs[method_key] = allocs
        print("Finish {}. Total time: {} seconds, per layout: {} seconds".format(method_key, run_time, run_time / test_layouts))

    # Strongest links allocation benchmark (if without fast fading, then same as shortest links benchmark
    if("strongest_links_FP_ratio_sch" in all_benchmarks):
        print("==========================Strongest Links Scheduling Benchmark with FP activation rate====================")
        active_amount_FP = np.ceil(np.sum(all_allocs["FP"]) / test_layouts).astype(int)  # according to FP allocation, use ceil to make it having a bit more activation
        assert active_amount_FP > 0, "[Evaluate] At shortest links allocation with FP activation rate benchmark, having non-positive {} activation amount computed".format(active_amount_FP)
        scheduling_thresholds = np.expand_dims(np.sort(test_gains_diagonal, axis=1)[:,-active_amount_FP],axis=-1) # note: pick the kth largest for threshold
        strongest_links_allocs_FPrate = (test_gains_diagonal >= scheduling_thresholds).astype(int)
        assert np.shape(strongest_links_allocs_FPrate) == (test_layouts, N), "Wrong shape: {}".format(np.shape(strongest_links_allocs_FPrate))
        all_allocs["Strongest Links"] = strongest_links_allocs_FPrate

    if("direct_allocs_backprop" in all_benchmarks):
        print("=========================Direct Allocations Back-Propagation=================================")
        method_key = "Direct Allocs BackProp"
        all_allocs[method_key] = Direct_Allocs_BackProp.compute_allocations(test_gains_diagonal, test_gains_nondiagonal)
        print("Finished {} Allocations Collection!".format(method_key))

    print("##########################EVALUATION AND COMPARISON########################")
    all_sum_rates = dict()
    for method_key in all_allocs.keys():
        all_links_rates = utils.compute_rates(general_para, all_allocs[method_key], test_gains_diagonal, test_gains_nondiagonal) # Test Layouts X N
        sum_rates = np.sum(all_links_rates,axis=-1)
        assert np.shape(sum_rates) == (test_layouts,)
        all_sum_rates[method_key] = sum_rates
    print("[Mean Rate per Link] Averaged over all layouts")
    for method_key in all_allocs.keys():
        mean_rate_per_link = np.mean(all_sum_rates[method_key]) / N
        print("[{}]: {}Mbps; ".format(method_key, round(mean_rate_per_link/1e6,2)), end="")
    print("\n")
    print("----------------------------------------------------------------")
    print("[All ratios' avg for Sum Rates]")
    assert "FP" in all_allocs.keys(), "[evaluate.py] Didn't include FP in testing rate computation"
    for method_key in all_allocs.keys():
        if(method_key == "FP"):
            continue
        ratios = all_sum_rates[method_key]/all_sum_rates["FP"]*100; assert np.shape(ratios) == (test_layouts,)
        print("[{}]: avg {}%, max {}% of FP;".format(method_key, round(np.mean(ratios),2), round(np.max(ratios),2)), end="")
    print("\n")
    print("----------------------------------------------------------------")
    if(display_ratios_option):
        print("[Minimum; 5-Percentile; Maximum Comparison]")
        FP_min = np.min(all_sum_rates["FP"]); FP_5pert = np.percentile(all_sum_rates["FP"], 5); FP_max = np.max(all_sum_rates["FP"])
        for method_key in all_allocs.keys():
            if(method_key == "FP"):
                continue
            print("[{}]: min {}%, 5-percentile {}%, max {}% of FP;".format(method_key,
                    round(np.min(all_sum_rates[method_key])/FP_min*100,2),
                    round(np.percentile(all_sum_rates[method_key], 5)/FP_5pert*100,2),
                    round(np.max(all_sum_rates[method_key])/FP_max*100,2)), end="")
        print("\n")
        print("----------------------------------------------------------------")
    print("[Activation Portion] ")
    for method_key in all_allocs.keys():
        print("[{}]: {}%; ".format(method_key, round(np.mean(all_allocs[method_key]) * 100, 2)), end="")
    print("\n")

    # Plot CDF curve
    if(formal_CDF_legend_option):
        line_styles = dict()
        line_styles["Neural Network"] = "r-"
        line_styles["FP"] = "g-."
        line_styles["FP Not Knowing Fastfading"] = "b--" # most time not used
        line_styles["Greedy"] = "k:"
        line_styles["All Active"] = "y-"
        line_styles["Random"] = "m--"
        line_styles["Strongest Links"] = "c-."
    fig = plt.figure(); ax = fig.gca()
    plt.xlabel("Sum Rates (bps)")
    plt.ylabel("Cumulative Distribution of D2D Network Sum Rate")
    plt.grid(linestyle="dotted")
    ax.set_ylim(bottom=0)
    # Ensure fixed Ordering
    if(formal_CDF_legend_option):
        allocs_keys_ordered = ["Neural Network", "FP", "FP Not Knowing Fastfading", "Greedy", "Strongest Links", "All Active", "Random"]
        for method_key in allocs_keys_ordered:
            if(method_key not in all_sum_rates.keys()):
                print("{} doesn't have sum rate computed, skipping and check it later...".format(method_key))
                continue
            sum_rates = np.sort(all_sum_rates[method_key])
            plt.plot(sum_rates, np.arange(1, test_layouts + 1) / test_layouts, line_styles[method_key], label=method_key)
    else:
        for method_key in all_sum_rates.keys():
            sum_rates = np.sort(all_sum_rates[method_key])
            plt.plot(sum_rates, np.arange(1, test_layouts + 1) / test_layouts, label=method_key)
    plt.legend()
    plt.show()

    # plot the activations for randomly selected three of the testing sample comparing FP allocation, FC net, and conv net allocation
    v_layouts_amount = 5
    v_layouts_indices = np.random.randint(low=0, high=test_layouts-1, size=v_layouts_amount)
    x_range = general_para.field_length; y_range = x_range
    x_cell = general_para.cell_length; y_cell = x_cell
    for i in range(v_layouts_amount):
        v_layout_index = v_layouts_indices[i]
        print("Plotting for {}th randomly selected layout at original testing set index {}...".format(i, v_layout_index))
        # segeneral_parate Tx and Rx locations
        tx_locs = test_locations[v_layout_index][:, 0:2]; rx_locs = test_locations[v_layout_index][:, 2:4]
        # [First] plot scatters and all transceiver pairs
        fig = plt.figure(); ax = fig.gca()
        plt.title("Original Generated Set Up for Sample #{}".format(i + 1))
        plt.scatter(tx_locs[:, 0], tx_locs[:, 1], c='r', label='Tx', s=10); plt.scatter(rx_locs[:, 0], rx_locs[:, 1], c='b', label='Rx', s=10)
        for j in range(N):  # plot pairwise connections
            plt.plot([tx_locs[j, 0], rx_locs[j, 0]], [tx_locs[j, 1], rx_locs[j, 1]], 'b')
        ax.set_xticks(np.arange(start=0, stop=x_range+1, step=x_range/10)); ax.set_xticks(np.arange(start=0, stop=x_range, step=x_cell), minor=True)
        ax.set_yticks(np.arange(start=0, stop=y_range+1, step=y_range/10)); ax.set_yticks(np.arange(start=0, stop=y_range, step=y_cell), minor=True)
        plt.grid(color="y", alpha=0.5, which='minor')
        plt.legend()
        # Then plot all methods allocations
        for method_key in all_allocs.keys():
            if(method_key in ["All Active", "Random"]):
                continue # Don't plot for these trivial allocations
            v_alloc = all_allocs[method_key][v_layout_index]
            fig = plt.figure(); ax = fig.gca()
            ax.set_xlim(left=0,right=x_range); ax.set_ylim(bottom=0,top=y_range)
            plt.title("Activation Result by {} on Sample #{}".format(method_key, i+1))
            plt.scatter(tx_locs[:, 0], tx_locs[:, 1], c='r', label='Tx', s=7)
            plt.scatter(rx_locs[:, 0], rx_locs[:, 1], c='b', label='Rx', s=7)
            for j in range(N): # plot all activated links
                if(method_key in all_supersets.keys()):
                    if(all_supersets[method_key][v_layout_index][j] == 1):
                        plt.plot([tx_locs[j, 0], rx_locs[j, 0]], [tx_locs[j, 1], rx_locs[j, 1]], "b", linewidth=3.0, alpha=0.3)
                if(v_alloc[j] == 0):
                    continue # don't plot anything
                plt.plot([tx_locs[j, 0], rx_locs[j, 0]], [tx_locs[j, 1], rx_locs[j, 1]], "k")
            ax.set_xticks(np.arange(start=0, stop=x_range+1, step=x_range / 10)); ax.set_xticks(np.arange(start=0, stop=x_range, step=x_cell), minor=True)
            ax.set_yticks(np.arange(start=0, stop=y_range+1, step=y_range / 10)); ax.set_yticks(np.arange(start=0, stop=y_range, step=y_cell), minor=True)
            plt.grid(color="y", alpha=0.5, which='minor')
            plt.legend()
        plt.show()

    print("Script Completed Successfully!")
