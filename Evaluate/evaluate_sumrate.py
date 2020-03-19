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
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
sys.path.append("../Utilities/")
import general_parameters
import helper_functions
import benchmarks
sys.path.append("../Neural_Network_Model/")
import Deep_Learning_Scheduling

INCLUDE_FADING = False

if(__name__ =='__main__'):
    general_para = general_parameters.parameters()
    N = general_para.n_links
    print("[Evaluate SumRate] Evaluate Setting: ", general_para.setting_str)
    layouts = np.load("../Data/layouts_{}.npy".format(general_para.setting_str))
    path_losses = np.load("../Data/path_losses_{}.npy".format(general_para.setting_str))
    if(INCLUDE_FADING):
        print("Testing under CSI including fading realizations...")
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
    sum_rates_all_methods = {}

    allocs_all_methods["FP"] = benchmarks.FP(general_para, channel_losses, np.ones([n_layouts, N]))

    allocs_all_methods["FP Not Knowing Fading"] = benchmarks.FP(general_para, path_losses, np.ones([n_layouts, N]))

    allocs_all_methods["Deep Learning"] = Deep_Learning_Scheduling.sumRate_scheduling(general_para, layouts)

    allocs_all_methods["Greedy"] = benchmarks.Greedy_Scheduling(general_para, directLink_channel_losses, crossLink_channel_losses, np.ones([n_layouts, N]))

    allocs_all_methods["Strongest Link Only"] = benchmarks.Strongest_Link_Scheduling(general_para, directLink_channel_losses)

    allocs_all_methods["Random"] = np.random.randint(2, size=[n_layouts, N]).astype(float)

    allocs_all_methods["All Active"] = np.ones([n_layouts, N]).astype(float)


    print("<<<<<<<<<<<<<<<<EVALUATION>>>>>>>>>>>>>>>>>>>")
    print("[Percentages of Scheduled Links] ")
    for method_key in allocs_all_methods.keys():
        print("[{}]: {}%; ".format(method_key, round(np.mean(allocs_all_methods[method_key]) * 100, 2)), end="")
    print("\n")

    for method_key in allocs_all_methods.keys():
        all_links_rates = helper_functions.compute_rates(general_para, allocs_all_methods[method_key], directLink_channel_losses, crossLink_channel_losses) # n_layouts X N
        assert np.shape(all_links_rates) == (n_layouts, N)
        sum_rates = np.sum(all_links_rates,axis=-1)
        sum_rates_all_methods[method_key] = sum_rates
    print("[Sum-Rate Performance Ratios Averaged over all Layouts]")
    assert "FP" in allocs_all_methods.keys(), "[evaluate.py] Didn't include FP in sum-rate computation"
    for method_key in allocs_all_methods.keys():
        if(method_key == "FP"):
            continue
        ratios = sum_rates_all_methods[method_key]/sum_rates_all_methods["FP"]*100
        print("[{}]: avg {}% of FP;".format(method_key, round(np.mean(ratios),2)), end="")
    print("\n")


    # Plot CDF curve
    line_styles = dict()
    line_styles["Deep Learning"] = "r-"
    line_styles["FP"] = "g-."
    line_styles["FP Not Knowing Fading"] = "b:" # most time not used
    line_styles["Greedy"] = "k--"
    line_styles["Strongest Link Only"] = "c-."
    line_styles["All Active"] = "y-"
    line_styles["Random"] = "m--"
    fig = plt.figure()
    ax = fig.gca()
    plt.xlabel("Sum Rates (Mbps)")
    plt.ylabel("Cumulative Distribution of D2D Network Sum Rate")
    plt.grid(linestyle="dotted")
    ax.set_ylim(bottom=0)
    for method_key in allocs_all_methods.keys():
        sum_rates = np.sort(sum_rates_all_methods[method_key])
        plt.plot(sum_rates/1e6, np.arange(1, n_layouts+1) / n_layouts, line_styles[method_key], label=method_key)
    plt.legend()
    plt.show()

    print("Plotting the schedulings for randomly selected layouts...")
    layout_indices = np.random.randint(low=0, high=n_layouts-1, size=4)
    for method_key in allocs_all_methods.keys():
        if (method_key in ["All Active", "Random", "Strongest Link Only"]):
            continue  # Don't plot for these trivial allocations
        plt.suptitle("{} Scheduling".format(method_key))
        for i, layout_index in enumerate(layout_indices):
            ax = plt.subplot(221 + i)
            ax.set_title("{}th Layout".format(layout_index))
            helper_functions.visualize_layout(ax, layouts[layout_index])
            helper_functions.visualize_schedules_on_layout(ax, layouts[layout_index], allocs_all_methods[method_key][layout_index])
        plt.show()

    print("Script Completed Successfully!")
