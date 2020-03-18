# This script contains the optimization computation using our novel convolutional neural network
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
import matplotlib.pyplot as plt


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
            saver.restore(sess, neural_net.model_filename)
            print("Restored model from: {}!".format(neural_net.model_filename))
            schedules = sess.run(neural_net.outputs_final,
                                feed_dict={neural_net.placeholders['tx_indices_hash']: neural_net_inputs['tx_indices_hash'],
                                           neural_net.placeholders['rx_indices_hash']: neural_net_inputs['rx_indices_hash'],
                                           neural_net.placeholders['tx_indices_extract']: neural_net_inputs['tx_indices_ext'],
                                           neural_net.placeholders['rx_indices_extract']: neural_net_inputs['rx_indices_ext'],
                                           neural_net.placeholders['pair_tx_convfilter_indices']: neural_net_inputs['pair_tx_convfilter_indices'],
                                           neural_net.placeholders['pair_rx_convfilter_indices']: neural_net_inputs['pair_rx_convfilter_indices'] })
    schedules = np.array(schedules)
    assert np.shape(schedules) == (n_layouts, N), "Wrong shape: {}".format(np.shape(schedules))
    print("Successfully Computed Neural Network's Scheduling for Sum-Rate Optimization!")
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
            saver.restore(sess, neural_net.model_filename)
            print("Restored model from: {}!".format(neural_net.model_filename))
            schedules_all_timeSlots = []
            rates_all_timeSlots = []
            subsets_all_timeSlots = []
            prop_weights_all_timeSlots = []
            prop_weights = np.ones([n_layouts, N])
            prop_weights_binary = np.ones([n_layouts, N])
            for i in range(n_timeSlots):
                if (((i+1) / n_timeSlots * 100) % 20 == 0):
                    print("{}/{} time slots".format(i, n_timeSlots))
                schedules = sess.run(neural_net.outputs_final,
                                      feed_dict={neural_net.placeholders['tx_indices_hash']: neural_net_inputs['tx_indices_hash'],
                                                 neural_net.placeholders['rx_indices_hash']: neural_net_inputs['rx_indices_hash'],
                                                 neural_net.placeholders['tx_indices_extract']: neural_net_inputs['tx_indices_ext'],
                                                 neural_net.placeholders['rx_indices_extract']: neural_net_inputs['rx_indices_ext'],
                                                 neural_net.placeholders['pair_tx_convfilter_indices']: neural_net_inputs['pair_tx_convfilter_indices'],
                                                 neural_net.placeholders['pair_rx_convfilter_indices']: neural_net_inputs['pair_rx_convfilter_indices'],
                                                 neural_net.placeholders['subset_links']: prop_weights_binary})
                schedules = schedules * prop_weights_binary  # zero out links not to be scheduled
                prop_weights_all_timeSlots.append(prop_weights)
                subsets_all_timeSlots.append(prop_weights_binary)
                schedules_all_timeSlots.append(schedules)
                rates = helper_functions.compute_rates(general_para, schedules, gains_diagonal, gains_nondiagonal)
                rates_all_timeSlots.append(rates)
                prop_weights = helper_functions.proportional_update_weights(general_para, prop_weights, rates)
                prop_weights_binary = helper_functions.binary_importance_weights_approx(general_para, prop_weights)
    schedules_all_timeSlots = np.transpose(np.array(schedules_all_timeSlots), (1, 0, 2))
    assert np.shape(schedules_all_timeSlots) == (n_layouts, n_timeSlots, N), "Wrong shape: {}".format(np.shape(schedules_all_timeSlots))
    rates_all_timeSlots = np.transpose(np.array(rates_all_timeSlots), (1, 0, 2))
    assert np.shape(rates_all_timeSlots) == (n_layouts, n_timeSlots, N), "Wrong shape: {}".format(np.shape(rates_all_timeSlots))
    subsets_all_timeSlots = np.transpose(np.array(subsets_all_timeSlots), (1, 0, 2))
    assert np.shape(subsets_all_timeSlots) == (n_layouts, n_timeSlots, N), "Wrong shape: {}".format(np.shape(subsets_all_timeSlots))
    prop_weights_all_timeSlots = np.transpose(np.array(prop_weights_all_timeSlots), (1, 0, 2))
    assert np.shape(prop_weights_all_timeSlots) == (n_layouts, n_timeSlots, N), "Wrong shape: {}".format(np.shape(prop_weights_all_timeSlots))

    return schedules_all_timeSlots, rates_all_timeSlots, subsets_all_timeSlots


# Function for interpretability: start a standalone tensorflow session for convolutional filter visualization
def visualize_convolutional_filter(general_para):
    # number of layouts here doesn't matter just for weight visualization
    neural_net = Convolutional_Neural_Network_Model.Conv_Network(general_para, 10)
    neural_net.build_network()
    with neural_net.TFgraph.as_default():
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, neural_net.model_filename)
            print("Restored model from: {}!".format(neural_net.model_filename))
            weights_tensorlist = tf.get_collection("conv")
            conv_weights_tensor = weights_tensorlist[0]
            assert conv_weights_tensor.name == "conv_lyr/w:0", "Tensor extraction failed, with wrong name: {}".format(conv_weights_tensor.name)
            # the weights in convolutional computation are exponentialized
            conv_weights_log_scale = sess.run(conv_weights_tensor)
    conv_weights_log_scale = np.array(conv_weights_log_scale)
    assert np.shape(conv_weights_log_scale) == (neural_net.filter_size, neural_net.filter_size, 1, 1), "Wrong shape: {}".format(np.shape(conv_weights_log_scale))
    plt.title("Convolutioanl Filter Weights Visualization (log scale)")
    img1 = plt.imshow(np.squeeze(conv_weights_log_scale), cmap=plt.get_cmap("Greys"), origin="lower")
    plt.colorbar(img1, cmap=plt.get_cmap("Greys"))
    plt.show()
    print("Convolutional Filter Visualization Complete!")
    return