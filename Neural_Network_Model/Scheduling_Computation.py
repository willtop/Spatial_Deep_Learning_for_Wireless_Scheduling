# This repository contains the optimization computation using our novel convolutional neural network
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
import Convolutional_Neural_Network_Model
import sys
sys.path.append("../Utilities/")
import helper_functions

# Scheduling for Sum-Rate Optimization
def sumRate_scheduling(general_para, layouts):
    N = general_para.n_links
    n_layouts = np.shape(layouts)[0]
    neural_net = Convolutional_Neural_Network_Model.Conv_Network(general_para, n_layouts)
    neural_net.build_network()
    neural_net_inputs = helper_functions.process_layouts_inputs(general_para, layouts)

    with neural_net.TFgraph.as_default():
        saver = tf.train.Saver()
        with tf.Session() as sess:
            print("Restoring model from: {}".format(neural_net.model_filename))
            saver.restore(sess, neural_net.model_filename)
            schedules = sess.run(neural_net.outputs_final,
                                feed_dict={neural_net.placeholders['tx_indices_hash']: neural_net_inputs['tx_indices_hash'],
                                           neural_net.placeholders['rx_indices_hash']: neural_net_inputs['rx_indices_hash'],
                                           neural_net.placeholders['tx_indices_extract']: neural_net_inputs['tx_indices_ext'],
                                           neural_net.placeholders['rx_indices_extract']: neural_net_inputs['rx_indices_ext'],
                                           neural_net.placeholders['pair_tx_convfilter_indices']: neural_net_inputs['pair_tx_convfilter_indices'],
                                           neural_net.placeholders['pair_rx_convfilter_indices']: neural_net_inputs['pair_rx_convfilter_indices'] })
    schedules = np.array(schedules)
    assert np.shape(schedules) == (n_layouts, N), "Wrong shape: {}".format(np.shape(schedules))

    return schedules


# Scheduling for Log Utility Optimization
def logUtil_scheduling(general_para, layouts, gains_diagonal, gains_nondiagonal):
    N = general_para.n_links
    n_layouts = np.shape(layouts)[0]
    neural_net = Convolutional_Neural_Network_Model.Conv_Network(general_para, n_layouts)
    neural_net.build_network()
    neural_net_inputs = helper_functions.process_layouts_inputs(general_para, layouts)
    n_timeSlots = general_para.n_timeSlots

    with neural_net.TFgraph.as_default():
        saver = tf.train.Saver()
        with tf.Session() as sess:
            print("Restoring model from: {}".format(neural_net.model_filename))
            saver.restore(sess, neural_net.model_filename)
            allocs_all_timeSlots = []
            rates_all_timeSlots = []
            subsets_all_timeSlots = []
            prop_weights_all_timeSlots = []
            prop_weights = np.ones([n_layouts, N])
            prop_weights_binary = np.ones([n_layouts, N])
            for i in range(n_timeSlots):
                if (((i+1) / n_timeSlots * 100) % 20 == 0):
                    print("{}/{} time slots".format(i, n_timeSlots))
                allocs_oneslot = sess.run(neural_net.outputs_final,
                                          feed_dict={placeholders['tx_indices_hash']: test_data['tx_indices_hash'][0],
                                                     placeholders['rx_indices_hash']: test_data['rx_indices_hash'][0],
                                                     placeholders['tx_indices_extract']: test_data['tx_indices_ext'][0],
                                                     placeholders['rx_indices_extract']: test_data['rx_indices_ext'][0],
                                                     placeholders['pair_tx_convfilter_indices']:
                                                         test_data['pair_tx_convfilter_indices'][0],
                                                     placeholders['pair_rx_convfilter_indices']:
                                                         test_data['pair_rx_convfilter_indices'][0],
                                                     placeholders['subset_links']: weights_binary})
                total_time += time.time() - start_time
                allocs_oneslot = allocs_oneslot * weights_binary  # zero out links not to be scheduled
                orig_prop_weights.append(weights_orig)
                subsets.append(weights_binary)
                allocs.append(allocs_oneslot)
                rates_oneslot = utils.compute_rates(general_para, allocs_oneslot, gains_diagonal, gains_nondiagonal)
                rates.append(rates_oneslot)
                weights_orig = utils.proportional_update_weights(general_para, weights_orig, rates_oneslot)
                start_time = time.time()
                weights_binary = utils.binary_importance_weights_approx(general_para, weights_orig)
                total_time += time.time() - start_time
    print("{} layouts with {} links over {} timeslots, it took {} seconds.".format(layouts_amount, N, slots_per_layout,
                                                                                   total_time))
    allocs = np.transpose(np.array(allocs), (1, 0, 2))
    assert np.shape(allocs) == (layouts_amount, slots_per_layout, N), "Wrong shape: {}".format(np.shape(allocs))
    rates = np.transpose(np.array(rates), (1, 0, 2))
    assert np.shape(rates) == (layouts_amount, slots_per_layout, N), "Wrong shape: {}".format(np.shape(rates))
    subsets = np.transpose(np.array(subsets), (1, 0, 2))
    assert np.shape(subsets) == (layouts_amount, slots_per_layout, N), "Wrong shape: {}".format(np.shape(subsets))
    orig_prop_weights = np.transpose(np.array(orig_prop_weights), (1, 0, 2))
    assert np.shape(orig_prop_weights) == (layouts_amount, slots_per_layout, N), "Wrong shape: {}".format(
        np.shape(orig_prop_weights))
    np.save(general_para.base_dir + "SanityChecks/Weighted_SumRate_Opt/Conv_V10/" + "allocs.npy", allocs)
    np.save(general_para.base_dir + "SanityChecks/Weighted_SumRate_Opt/Conv_V10/" + "subsets.npy", subsets)
    np.save(general_para.base_dir + "SanityChecks/Weighted_SumRate_Opt/Conv_V10/" + "prop_weights.npy",
            orig_prop_weights)
    return allocs, rates, subsets
