# Script for evaluating

import numpy as np
# to silent tensorflow WARNING
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import matplotlib.pyplot as plt
import general_parameters
import utils
import FPLinQ
import Direct_Allocs_BackProp
import time
# import functions for DL WMMSE modules
import sys
sys.path.append("../Conv_Net_Model_SumRate_V4/")
sys.path.append("../Conv_Net_Model_SumRate_V9/")
sys.path.append("../Conv_Net_Model_SumRate_V10/")
# import source codes implementing methods
#import Conv_Net_SumRate_V4
#import Conv_Net_SumRate_V9
import Conv_Net_SumRate_V10

# selectors on which ones to compute and compare
all_models = [
    # "stepbystep_sumrate_v4",
    # "stepbystep_sumrate_FP",
    # "stepbystep_sumrate_Random",
    # "stepbystep_sumrate_v9",
    "stepbystep_sumrate_v10"
]

all_benchmarks = [
    "weighted_greedy_sch",
    "all_active",
    "random_sch",
    "max_weight_sch",
    #"direct_allocs_backprop"
]

formal_CDF_legend_option = True

if(__name__ =='__main__'):
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--N', help='Amount of D2D Links Per Layout', default=50)
    args = parser.parse_args()
    N = int(args.N)
    print("[EvaluateSumRate] Pair amount: {}".format(N))
    # Loading files
    general_para = general_parameters.parameters(N)
    test_layouts, test_slots_per_layout = general_para.test_data_info["layouts"], general_para.test_data_info["slots_per_layout"]
    data_dir = general_para.data_folder + general_para.test_data_info["folder"]
    test_locations = np.load(data_dir + general_para.locs_file)
    test_gains = np.load(data_dir + general_para.G_file)
    assert np.shape(test_locations) == (test_layouts, N, 4), "Wrong shape: {}".format(np.shape(test_locations))
    assert np.shape(test_gains) == (test_layouts, N, N), "Wrong shape: {}".format(np.shape(test_gains))
    test_gains_diagonal = np.diagonal(test_gains, axis1=1, axis2=2)  # layouts X N
    test_gains_nondiagonal = test_gains * ((np.identity(N) < 1).astype(int))  # layouts X N X N

    all_allocs = dict() # expect layouts X timeslots X N
    all_rates = dict() # expect layouts X timeslots X N
    all_subsets = dict() # expect layouts X timeslots X N

    # FP Allocation
    # Load the previously computed FP allocation outputs for testing samples
    print("==============================Fractional Programming=============================")
    method_key = "FP"
    test_FP_allocs = np.load(data_dir+general_para.FP_alloc_file)
    test_FP_allocs = np.reshape(test_FP_allocs, [test_layouts, test_slots_per_layout, N])
    rates = []; mean_rates_diffs = []; all_prop_weights = []; prop_weights = np.ones([test_layouts, N])
    rates_old = np.zeros([1, test_layouts, N]) # dummy variable
    for i in range(test_slots_per_layout):
        rates_oneslot = utils.compute_rates(general_para, test_FP_allocs[:,i,:], test_gains_diagonal, test_gains_nondiagonal)
        all_prop_weights.append(prop_weights)
        prop_weights = utils.proportional_update_weights(general_para, prop_weights, rates_oneslot)
        rates.append(rates_oneslot)
    rates = np.transpose(np.array(rates),(1,0,2)); assert np.shape(rates)==(test_layouts, test_slots_per_layout, N)
    all_prop_weights = np.transpose(np.array(all_prop_weights),(1,0,2)); assert np.shape(all_prop_weights)==(test_layouts, test_slots_per_layout, N)
    np.save(general_para.base_dir+"SanityChecks/Weighted_SumRate_Opt/FP/"+"allocs.npy", test_FP_allocs)
    np.save(general_para.base_dir+"SanityChecks/Weighted_SumRate_Opt/FP/"+"prop_weights.npy", all_prop_weights)
    all_allocs[method_key] = test_FP_allocs
    all_rates[method_key] = rates
    print("Successfully Collected FP Allocation Output for Testing Samples!")
    # np.save(para.data_folder + para.test_data_info['folder'] + "FP_{}_meters_{}_pairs_rates.npy".format(para.field_length, N), rates)

    if("stepbystep_sumrate_v4" in all_models):
        print("==============================Step by Step Sum Rate V4==========================")
        method_key = "Deep Learning V4"
        allocs_path = general_para.data_folder + general_para.test_data_info['folder'] + "netV4_{}_meters_{}_pairs_allocs.npy".format(general_para.field_length, N)
        try:
            allocs = np.load(allocs_path)
        except FileNotFoundError:
            print("Can't find pre-computed V4 allocation at {}! Compute from fresh...".format(allocs_path))
            need_compute = True
        else:
            all_allocs[method_key] = allocs
            print("Successfully loaded V4 allocation from {}! Computing achieved rates...".format(allocs_path))
            need_compute = False
            V4_rates = []
            start_time = time.time()
            for i in range(test_slots_per_layout):
                rates_oneslot = utils.compute_rates(general_para, allocs[:, i, :], test_gains_diagonal, test_gains_nondiagonal)
                V4_rates.append(rates_oneslot)
            V4_rates = np.transpose(np.array(V4_rates), (1, 0, 2)); assert np.shape(V4_rates) == (test_layouts, test_slots_per_layout, N)
            all_rates[method_key] = V4_rates
            print("Finished V4 Rates Computation given loaded allocs! Took {} seconds".format(time.time() - start_time))
        if(need_compute):
            allocs, V4_rates, subsets = Conv_Net_SumRate_V4.network_inference_weighted(general_para, test_gains_diagonal, test_gains_nondiagonal)
            all_allocs[method_key] = allocs
            all_rates[method_key] = V4_rates
            all_subsets[method_key] = subsets
            print("Successfully Collected Step-by-Step SumRate V4 Allocs Computation Results for Testing Samples.")
            np.save(allocs_path, allocs)
            print("{} allocs saved at {}!".format(method_key, allocs_path))

    if ("stepbystep_sumrate_v9" in all_models):
        print("==============================Step by Step Sum Rate V9==========================")
        method_key = "Deep Learning V9"
        allocs_path = general_para.data_folder + general_para.test_data_info['folder'] + "netV9_{}_meters_{}_pairs_allocs.npy".format(general_para.field_length, N)
        try:
            allocs = np.load(allocs_path)
        except FileNotFoundError:
            print("Can't find pre-computed V9 allocation at {}! Compute from fresh...".format(allocs_path))
            need_compute = True
        else:
            all_allocs[method_key] = allocs
            print("Successfully loaded V9 allocation from {}! Computing achieved rates...".format(allocs_path))
            need_compute = False
            V9_rates = []
            start_time = time.time()
            for i in range(test_slots_per_layout):
                rates_oneslot = utils.compute_rates(general_para, allocs[:, i, :], test_gains_diagonal, test_gains_nondiagonal)
                V9_rates.append(rates_oneslot)
            V9_rates = np.transpose(np.array(V9_rates), (1, 0, 2));
            assert np.shape(V9_rates) == (test_layouts, test_slots_per_layout, N)
            all_rates[method_key] = V9_rates
            print("Finished V9 Rates Computation given loaded allocs! Took {} seconds".format(time.time() - start_time))
        if (need_compute):
            allocs, V9_rates, subsets = Conv_Net_SumRate_V9.network_inference_weighted(general_para, test_gains_diagonal, test_gains_nondiagonal)
            all_allocs[method_key] = allocs
            all_rates[method_key] = V9_rates
            all_subsets[method_key] = subsets
            print("Successfully Collected Step-by-Step SumRate V9 Allocs Computation Results for Testing Samples.")
            np.save(allocs_path, allocs)
            print("{} allocs saved at {}!".format(method_key, allocs_path))

    if ("stepbystep_sumrate_v10" in all_models):
        print("==============================Step by Step Sum Rate V10==========================")
        method_key = "Deep Learning"
        allocs_path = general_para.data_folder + general_para.test_data_info['folder'] + "netV10_{}_meters_{}_pairs_allocs.npy".format(general_para.field_length, N)
        try:
            allocs = np.load(allocs_path)
        except FileNotFoundError:
            print("Can't find pre-computed V10 allocation at {}! Compute from fresh...".format(allocs_path))
            need_compute = True
        else:
            all_allocs[method_key] = allocs
            print("Successfully loaded V10 allocation from {}! Computing achieved rates...".format(allocs_path))
            need_compute = False
            V10_rates = []
            start_time = time.time()
            for i in range(test_slots_per_layout):
                rates_oneslot = utils.compute_rates(general_para, allocs[:, i, :], test_gains_diagonal, test_gains_nondiagonal)
                V10_rates.append(rates_oneslot)
            V10_rates = np.transpose(np.array(V10_rates), (1, 0, 2));
            assert np.shape(V10_rates) == (test_layouts, test_slots_per_layout, N)
            all_rates[method_key] = V10_rates
            print("Finished V10 Rates Computation given loaded allocs! Took {} seconds".format(time.time() - start_time))
        if (need_compute):
            allocs, V10_rates, subsets = Conv_Net_SumRate_V10.network_inference_weighted(general_para, test_gains_diagonal, test_gains_nondiagonal)
            all_allocs[method_key] = allocs
            all_rates[method_key] = V10_rates
            all_subsets[method_key] = subsets
            print("Successfully Collected Step-by-Step SumRate V10 Allocs Computation Results for Testing Samples.")
            np.save(allocs_path, allocs)
            print("{} allocs saved at {}!".format(method_key, allocs_path))

    if("stepbystep_sumrate_FP" in all_models):
        print("==============================Step by Step Sum Rate FP Heuristic==========================")
        para = parameters.fc_parameters("dummy", N)
        allocs = []; rates = []; mean_rates_diffs = []; subsets = []
        weights_orig = np.ones([test_layouts, N]); weights_binary = np.ones([test_layouts, N])
        rates_old = np.zeros([1, test_layouts, N]) # dummy variable
        start_time = time.time()
        for i in range(test_slots_per_layout):
            if(((i+1) / test_slots_per_layout * 100) % 20 == 0):
                print("{}/{} time slots".format(i, test_slots_per_layout))
            allocs_oneslot = []
            for j in range(test_layouts): # FP allocation has to be obtained one by one
                allocs_oneslot_onelayout, _ = FPLinQ.optimize_discretize(para, test_gains[j], test_gains_diagonal[j], test_gains_nondiagonal[j], np.expand_dims(weights_binary[j],axis=-1), para.FP_iter_amount)
                allocs_oneslot.append(allocs_oneslot_onelayout)
            allocs_oneslot = np.array(allocs_oneslot)
            rates_oneslot = utils.compute_rates(para, allocs_oneslot, test_gains_diagonal, test_gains_nondiagonal)
            allocs.append(allocs_oneslot); rates.append(rates_oneslot); subsets.append(weights_binary)
            weights_orig = utils.proportional_update_weights(para, weights_orig, rates_oneslot)
            weights_binary = utils.binary_importance_weights_approx(para, weights_orig)
            mean_rates = np.mean(rates, axis=0); mean_rates_old = np.mean(rates_old, axis=0)
            mean_rates_diffs.append(np.mean(np.linalg.norm(mean_rates - mean_rates_old, axis=1)))
            rates_old = rates.copy()  # ensure that they are not same array
        allocs = np.transpose(np.array(allocs), (1, 0, 2)); assert np.shape(allocs) == (test_layouts, test_slots_per_layout, N), "Wrong shape: {}".format(np.shape(allocs))
        rates = np.transpose(np.array(rates), (1, 0, 2)); assert np.shape(rates) == (test_layouts, test_slots_per_layout, N)
        mean_rates_diffs = np.array(mean_rates_diffs); assert np.shape(mean_rates_diffs) == (test_slots_per_layout,)
        subsets = np.transpose(np.array(subsets), (1, 0, 2)); assert np.shape(subsets) == (test_layouts, test_slots_per_layout, N)
        all_allocs["Step-by-Step SumRate FP"] = allocs
        all_rates["Step-by-Step SumRate FP"] = rates
        all_subsets["Step-by-Step SumRate FP"] = subsets
        print("Successfully Collected Step-by-Step SumRate FP Heuristic Output for Testing Samples! Took {} seconds".format(time.time()-start_time))

    if ("stepbystep_sumrate_Random" in all_models):
        print("==============================Step by Step Sum Rate Random Heuristic==========================")
        method_key = "Step-by-Step Random"
        para = parameters.fc_parameters("dummy", N)
        allocs = []; rates = []
        weights_orig = np.ones([test_layouts, N]); weights_binary = np.ones([test_layouts, N])
        start_time = time.time()
        for i in range(test_slots_per_layout):
            allocs_oneslot = np.random.randint(2, size=[test_layouts, N]) * weights_binary
            rates_oneslot = utils.compute_rates(para, allocs_oneslot, test_gains_diagonal, test_gains_nondiagonal)
            allocs.append(allocs_oneslot); rates.append(rates_oneslot)
            weights_orig = utils.proportional_update_weights(para, weights_orig, rates_oneslot)
            weights_binary = utils.binary_importance_weights_approx(para, weights_orig)
        allocs = np.transpose(np.array(allocs), (1, 0, 2)); assert np.shape(allocs) == (test_layouts, test_slots_per_layout, N), "Wrong shape: {}".format(np.shape(allocs))
        rates = np.transpose(np.array(rates), (1, 0, 2)); assert np.shape(rates) == (test_layouts, test_slots_per_layout, N)
        all_allocs[method_key] = allocs
        all_rates[method_key] = rates
        print("Successfully Collected {} Output for Testing Samples! Took {} seconds".format(method_key, time.time() - start_time))

    ###################################### Heuristics and Benchmarks ###################################
    # All activation benchmark
    if("all_active" in all_benchmarks):
        print("==================================All Activation Benchmark===============================")
        method_key = "All Active"
        all_active_allocs = np.ones([test_layouts, test_slots_per_layout, N])
        all_allocs[method_key] = all_active_allocs
        rates = []
        for i in range(test_slots_per_layout):
            rates_oneslot = utils.compute_rates(general_para, all_active_allocs[:, i, :], test_gains_diagonal, test_gains_nondiagonal)
            rates.append(rates_oneslot)
        rates = np.transpose(np.array(rates), (1, 0, 2)); assert np.shape(rates) == (test_layouts, test_slots_per_layout, N)
        all_rates[method_key] = rates

    # Random scheduling allocation benchmark
    if("random_sch" in all_benchmarks):
        print("==================================Random Scheduling Allocation Benchmark==============================")
        method_key = "Random"
        random_schedule_tmp = np.random.rand(test_layouts, test_slots_per_layout, N)
        random_schedule = (random_schedule_tmp>=0.5).astype(int)
        all_allocs[method_key] = random_schedule
        rates = []
        for i in range(test_slots_per_layout):
            rates_oneslot = utils.compute_rates(general_para, random_schedule[:, i, :], test_gains_diagonal, test_gains_nondiagonal)
            rates.append(rates_oneslot)
        rates = np.transpose(np.array(rates), (1, 0, 2)); assert np.shape(rates) == (test_layouts, test_slots_per_layout, N)
        all_rates[method_key] = rates

    # weighted greedy scheduling heuristic
    if("weighted_greedy_sch" in all_benchmarks):
        print("=============================Weighted Greedy Scheduling Heuristic==============================")
        method_key = "Weighted Greedy"
        allocs_path = general_para.data_folder + general_para.test_data_info['folder'] + "wGreedy_{}_meters_{}_pairs_allocs.npy".format(general_para.field_length, N)
        # Test if the weighted greedy scheduling is already pre-computed
        try:
            greedy_allocs = np.load(allocs_path)
        except FileNotFoundError:
            print("Can't find pre-computed weighted greedy allocation file at {}! Compute from fresh...".format(allocs_path))
            compute_greedy = True
        else:
            all_allocs[method_key] = greedy_allocs
            print("Successfully loaded weighted greedy allocation from {}! Computing weighted greedy achieved rates...".format(allocs_path))
            compute_greedy = False
            rates = []
            start_time = time.time()
            for i in range(test_slots_per_layout):
                rates_oneslot = utils.compute_rates(general_para, greedy_allocs[:, i, :], test_gains_diagonal, test_gains_nondiagonal)
                rates.append(rates_oneslot)
            rates = np.transpose(np.array(rates), (1, 0, 2)); assert np.shape(rates) == (test_layouts, test_slots_per_layout, N)
            all_rates[method_key] = rates
            print("Finished Weighted Greedy Rates Computation given loaded allocs! Took {} seconds".format(time.time()-start_time))
        if(compute_greedy):
            print("Start Computing Weighted Greedy Scheduling...")
            # Parameters preparation
            direct_rates = utils.compute_direct_rates(general_para, test_gains_diagonal)
            # Variables to update
            weights = np.ones([test_layouts, N])
            allocs = []; rates = []
            start_time = time.time()
            for i in range(test_slots_per_layout):
                if(((i+1) * 100 / test_slots_per_layout) % 10 == 0):
                    print("At {}th/{} time slot...".format(i+1, test_slots_per_layout))
                sorted_links_indices = np.argsort(weights * direct_rates, axis=-1)
                allocs_oneslot = np.zeros([test_layouts, N])
                previous_weighted_sum_rates = np.zeros([test_layouts])
                for j in range(N-1, -1, -1):
                    # schedule the ith shortest links
                    allocs_oneslot[np.arange(test_layouts), sorted_links_indices[:, j]] = 1
                    rates_oneslot = utils.compute_rates(general_para, allocs_oneslot, test_gains_diagonal, test_gains_nondiagonal)
                    weighted_sum_rates = np.sum(rates_oneslot * weights, axis=-1)  # (layouts,)
                    # schedule the ith shortest pair for samples that have sum rate improved
                    allocs_oneslot[np.arange(test_layouts), sorted_links_indices[:, j]] = (weighted_sum_rates > previous_weighted_sum_rates).astype(int)
                    previous_weighted_sum_rates = np.maximum(weighted_sum_rates, previous_weighted_sum_rates)
                # end of this time slot scheduling
                allocs.append(allocs_oneslot)
                rates_oneslot = utils.compute_rates(general_para, allocs_oneslot, test_gains_diagonal, test_gains_nondiagonal)
                rates.append(rates_oneslot)
                # update weights
                weights = utils.proportional_update_weights(general_para, weights, rates_oneslot)
            allocs = np.transpose(np.array(allocs), (1, 0, 2)); assert np.shape(allocs) == (test_layouts, test_slots_per_layout, N)
            rates = np.transpose(np.array(rates), (1, 0, 2)); assert np.shape(rates) == (test_layouts, test_slots_per_layout, N)
            all_allocs[method_key] = allocs
            all_rates[method_key] = rates
            print("{} with time spent: {} seconds".format(method_key, time.time() - start_time))
            np.save(allocs_path, allocs)
            print("{} allocs saved at {}!".format(method_key, allocs_path))

    # Max weight activation heuristic
    if("max_weight_sch" in all_benchmarks):
        print("==================================Max Weight Only Scheduling heuristic===========================")
        method_key = "Max Weight Only"
        allocs = []; rates = []
        weights = np.ones([test_layouts, N])
        start_time = time.time()
        for i in range(test_slots_per_layout):  # could make the entire process a function and only feeding in different allocation rule, but for now just plainly write it
            allocs_oneslot = np.zeros([test_layouts, N])
            allocs_oneslot[np.arange(test_layouts), np.argmax(weights,axis=1)] = 1
            allocs.append(allocs_oneslot)
            # compute rates parallely over many layouts
            rates_oneslot = utils.compute_rates(general_para, allocs_oneslot, test_gains_diagonal, test_gains_nondiagonal) # layouts X N
            rates.append(rates_oneslot)
            weights = utils.proportional_update_weights(general_para, weights, rates_oneslot)
        allocs = np.transpose(np.array(allocs), (1,0,2)); assert np.shape(allocs) == (test_layouts, test_slots_per_layout, N)
        rates = np.transpose(np.array(rates), (1,0,2)); assert np.shape(rates) == (test_layouts, test_slots_per_layout, N)
        all_allocs[method_key] = allocs
        all_rates[method_key] = rates
        print("{} with time spent: {}".format(method_key, time.time() - start_time))

    if("direct_allocs_backprop" in all_benchmarks):
        print("==================================Direct Allocations Backprop Scheduling heuristic===========================")
        method_key = "Direct Allocs BackProp"
        allocs, rates = Direct_Allocs_BackProp.direct_backprop(general_para, test_gains_diagonal, test_gains_nondiagonal)
        all_allocs[method_key] = allocs
        all_rates[method_key] = rates

    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<EVALUATION AND COMPARISON>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("[Activation Portion] ")
    for method_key in all_allocs.keys():
        print("[{}]: {}%; ".format(method_key, round(np.mean(all_allocs[method_key]) * 100, 2)), end="")
    print("\n")

    # Sum log mean rates evaluation
    print("----------------------------------------------------------------")
    print("[Sum Log Mean Rates (Mean over {} layouts)]:".format(test_layouts))
    all_sum_log_mean_rates = dict()
    all_link_mean_rates = dict()
    global_max_mean_rate = 0 # for plotting upperbound
    for method_key in all_allocs.keys():
        link_mean_rates = np.mean(all_rates[method_key], axis=1); assert np.shape(link_mean_rates) == (test_layouts, N), "Wrong shape: {}".format(np.shape(link_mean_rates))
        all_link_mean_rates[method_key] = link_mean_rates.flatten() # (test_layouts X N, )
        global_max_mean_rate = max(global_max_mean_rate, np.max(link_mean_rates))
        all_sum_log_mean_rates[method_key] = np.mean(np.sum(np.log(link_mean_rates/1e6 + 1e-5),axis=1))
    for method_key in all_allocs.keys():
        print("[{}]: {}; ".format(method_key, all_sum_log_mean_rates[method_key]), end="")
    print("\n")

    print("[Bottom 5-Percentile Mean Rate (Aggregate over all layouts)]:")
    for method_key in all_allocs.keys():
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
        v_allocs = all_allocs[method_key][v_layout]
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
    for method_key in all_allocs.keys():
        if (method_key in ["All Active", "Random"]):
            continue  # Don't plot for these trivial allocations
        print("[sequential_timeslots_plotting] Plotting {} allocations...".format(method_key))
        v_allocs = all_allocs[method_key][v_layout]
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
