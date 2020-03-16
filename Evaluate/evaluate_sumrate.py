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
import general_parameters
import helper_functions
import benchmarks
import sys
sys.path.append("../Neural_Network_Model/")
import Deep_Learning_Scheduling_Computation


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

    all_allocs["Deep Learning"] = Deep_Learning_Scheduling_Computation.sumRate_scheduling(general_para, layouts)

    all_allocs["All Active"] = benchmarks.all_active(general_para, n_layouts)

    all_allocs["Random"] = benchmarks.random_scheduling(general_para, n_layouts)

    all_allocs["Greedy"] = benchmarks.greedy_scheduling(general_para, directLink_channel_losses, crossLink_channel_losses)

    all_allocs["Strongest Link"] = benchmarks.strongest_link_scheduling(general_para, directLink_channel_losses)

    print("##############EVALUATION#################")
    print("[Percentages of Scheduled Links] ")
    for method_key in all_allocs.keys():
        print("[{}]: {}%; ".format(method_key, round(np.mean(all_allocs[method_key]) * 100, 2)), end="")
    print("\n")

    sum_rates_all_methods = dict()
    for method_key in all_allocs.keys():
        all_links_rates = helper_functions.compute_rates(general_para, all_allocs[method_key], directLink_channel_losses, crossLink_channel_losses) # n_layouts X N
        assert np.shape(all_links_rates) == (n_layouts, N)
        sum_rates = np.sum(all_links_rates,axis=-1)
        sum_rates_all_methods[method_key] = sum_rates
    print("[Sum-Rate Performance Ratios Averaged over all Layouts]")
    assert "FP" in all_allocs.keys(), "[evaluate.py] Didn't include FP in sum-rate computation"
    for method_key in all_allocs.keys():
        if(method_key == "FP"):
            continue
        ratios = sum_rates_all_methods[method_key]/sum_rates_all_methods["FP"]*100
        print("[{}]: avg {}% of FP;".format(method_key, round(np.mean(ratios),2)), end="")
    print("\n")


    # Plot CDF curve
    line_styles = dict()
    line_styles["Deep Learning"] = "r-"
    line_styles["FP"] = "g-."
    line_styles["FP Not Knowing Small Scale Fading"] = "b--" # most time not used
    line_styles["Greedy"] = "k:"
    line_styles["All Active"] = "y-"
    line_styles["Random"] = "m--"
    line_styles["Strongest Link"] = "c-."
    fig = plt.figure()
    ax = fig.gca()
    plt.xlabel("Sum Rates (Mbps)")
    plt.ylabel("Cumulative Distribution of D2D Network Sum Rate")
    plt.grid(linestyle="dotted")
    ax.set_ylim(bottom=0)
    for method_key in all_allocs.keys():
        sum_rates = np.sort(sum_rates_all_methods[method_key])
        plt.plot(sum_rates/1e6, np.arange(1, n_layouts) / n_layouts, line_styles[method_key], label=method_key)
    plt.legend()
    plt.show()

    # Plot the schedulings for randomly selected layouts
    layout_indices = np.random.randint(low=0, high=n_layouts-1, size=4)
    x_range = general_para.field_length; y_range = x_range
    x_cell = general_para.cell_length; y_cell = x_cell
    for method_key in all_allocs.keys():
        if (method_key in ["All Active", "Random", "Strongest Link"]):
            continue  # Don't plot for these trivial allocations
        plt.title("{} Scheduling".format(method_key))
        for i, layout_index in enumerate(layout_indices):
            ax = plt.subplot(221 + i)
            helper_functions.visualize_schedules_on_layout(ax, layouts[layout_index], all_allocs[method_key][layout_index], "{}th Layout".format(layout_index))
        plt.show()

    print("Script Completed Successfully!")
