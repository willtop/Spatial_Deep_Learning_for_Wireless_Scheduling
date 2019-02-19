# This repository contains our novel convolutional neural network model code implementation 
# for the work "Spatial Deep Learning for Wireless Scheduling", 
# available at https://arxiv.org/abs/1808.01486.

# For any reproduce, further research or development, please kindly cite our Arxiv paper: 
# @misc{spatial_learning18, 
#       author = "W. Cui and K. Shen and W. Yu", 
#       title = "Spatial Deep Learning for Wireless Scheduling", 
#       month = dec, 
#       year = 2018, 
#       note = {[Online] Available: https://arxiv.org/abs/1808.01486} 
# }

import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import sys
sys.path.append("../Tools/")
import general_parameters
import model_parameters
import utils

# Global parameters
Unsupervised_Training = True
N = 50
general_para = general_parameters.parameters(N)
model_para = model_parameters.parameters("conv_net_sumrate_v10")
amount_per_batch, stddev_init, learning_rate, grid_amount = \
        general_para.amount_per_batch, general_para.stddev_init, general_para.learning_rate, general_para.grid_amount
output_noise_power, tx_power, SNR_gap = general_para.output_noise_power, general_para.tx_power, general_para.SNR_gap
filter_size_full = model_para.conv_filter_size_full

def get_placeholders():
    placeholders = dict()
    tx_indices_hash = tf.placeholder(tf.int64, shape=[amount_per_batch * N, 4], name='tx_indices_placeholder')
    rx_indices_hash = tf.placeholder(tf.int64, shape=[amount_per_batch * N, 4], name='rx_indices_placeholder')
    tx_indices_extract = tf.placeholder(tf.int32, shape=[amount_per_batch, N, 3], name='tx_indices_extract_placeholder')  # extra index is for indicating index within the batch
    rx_indices_extract = tf.placeholder(tf.int32, shape=[amount_per_batch, N, 3], name='rx_indices_extract_placeholder')  # extra index is for indicating index within the batch
    pair_tx_convfilter_indices = tf.placeholder(tf.int32, shape=[amount_per_batch, N, 2], name='pair_tx_convfilter_indices_placeholder')  # for cancellating pair itself convolution contribution
    pair_rx_convfilter_indices = tf.placeholder(tf.int32, shape=[amount_per_batch, N, 2], name='pair_rx_convfilter_indices_placeholder')  # for cancellating pair itself convolution contribution
    schedule_label = tf.placeholder(tf.float32, shape=[amount_per_batch, N], name='scheduling_target_placeholder')
    placeholders['tx_indices_hash'] = tx_indices_hash; placeholders['rx_indices_hash'] = rx_indices_hash
    placeholders['tx_indices_extract'] = tx_indices_extract; placeholders['rx_indices_extract'] = rx_indices_extract
    placeholders['pair_tx_convfilter_indices'] = pair_tx_convfilter_indices; placeholders['pair_rx_convfilter_indices'] = pair_rx_convfilter_indices
    placeholders['schedule_label'] = schedule_label
    return placeholders

# Create Variables to be Reused
def get_parameters():
    with tf.variable_scope("conv_lyr"):
        weights = tf.get_variable(name="w", shape=[filter_size_full, filter_size_full, 1, 1], initializer=tf.constant_initializer(0))
        bias = tf.get_variable(name="b", shape=[], initializer=tf.constant_initializer(-3))
        tf.add_to_collection("conv", weights)
        tf.add_to_collection("conv", bias)
    with tf.variable_scope("fc_lyr1"):
        weights = tf.get_variable(name="w", shape=[6, 30], initializer=tf.truncated_normal_initializer(stddev=stddev_init))
        biases = tf.get_variable(name="b", shape=[30], initializer=tf.constant_initializer(0))
        tf.add_to_collection("fc", weights)
        tf.add_to_collection("fc", biases)
    with tf.variable_scope("fc_lyr2"):
        weights = tf.get_variable(name="w", shape=[30, 30], initializer=tf.truncated_normal_initializer(stddev=stddev_init))
        biases = tf.get_variable(name="b", shape=[30], initializer=tf.constant_initializer(0))
        tf.add_to_collection("fc", weights)
        tf.add_to_collection("fc", biases)
    with tf.variable_scope("fc_lyr3"):
        weights = tf.get_variable(name="w", shape=[30, 1], initializer=tf.truncated_normal_initializer(stddev=stddev_init))
        bias = tf.get_variable(name="b", shape=[], initializer=tf.constant_initializer(0))
        tf.add_to_collection("fc", weights)
        tf.add_to_collection("fc", bias)

# Regular 2D convolution operation
def convolution_layer(inputs, filters, bias):
    return tf.add(tf.nn.conv2d(inputs, filters, strides=[1, 1, 1, 1], padding="SAME"), bias)

# Define fully connected layers
def fully_connected_layer(inputs, weights, biases):
    return tf.nn.relu(tf.add(tf.matmul(inputs, weights), biases))

def fully_connected_layer_final(inputs, weights, bias):
    return tf.add(tf.matmul(inputs, weights), bias)

def iteration_step(allocs_state, placeholders):
    # flatten allocs_state as grid values
    grid_values = tf.reshape(allocs_state, [amount_per_batch * N])
    tx_grids = tf.SparseTensor(placeholders['tx_indices_hash'], grid_values, [amount_per_batch, grid_amount[0], grid_amount[1], 1])
    tx_grids = tf.sparse_reduce_sum(tx_grids, reduction_axes=3, keepdims=True)
    rx_grids = tf.SparseTensor(placeholders['rx_indices_hash'], grid_values, [amount_per_batch, grid_amount[0], grid_amount[1], 1])
    rx_grids = tf.sparse_reduce_sum(rx_grids, reduction_axes=3, keepdims=True)

    with tf.variable_scope("conv_lyr", reuse=True):
        weights = tf.get_variable(name="w")
        bias = tf.get_variable(name="b")
        # full region interferences
        tx_density_map_full = convolution_layer(tx_grids, tf.exp(weights), tf.exp(bias))
        rx_density_map_full = convolution_layer(rx_grids, tf.exp(weights), tf.exp(bias))
        pairing_tx_strength_full = tf.gather_nd(params=tf.squeeze(tf.exp(weights)), indices=placeholders['pair_tx_convfilter_indices'])
        pairing_rx_strength_full = tf.gather_nd(params=tf.squeeze(tf.exp(weights)), indices=placeholders['pair_rx_convfilter_indices'])
        pairing_tx_contrib_full = pairing_tx_strength_full * allocs_state
        pairing_rx_contrib_full = pairing_rx_strength_full * allocs_state

    # Select tx locations from rx convolution output, and vise versa (both shapes should be: amount_in_batch X N)
    tx_surroundings_full = tf.gather_nd(params=tf.squeeze(rx_density_map_full, axis=-1), indices=placeholders['tx_indices_extract']) - pairing_rx_contrib_full
    rx_surroundings_full = tf.gather_nd(params=tf.squeeze(tx_density_map_full, axis=-1), indices=placeholders['rx_indices_extract']) - pairing_tx_contrib_full

    direct_link_strength = pairing_tx_strength_full # amount_in_batch X N
    direct_link_strength_max = tf.tile(tf.reduce_max(direct_link_strength, axis=-1, keepdims=True),[1, N]) # amount_in_batch X N
    direct_link_strength_min = tf.tile(tf.reduce_min(direct_link_strength, axis=-1, keepdims=True),[1, N]) # amount_in_batch X N
    # Combine to obtain feature vectors
    pairs_features = tf.stack([tx_surroundings_full, rx_surroundings_full, direct_link_strength, direct_link_strength_max, direct_link_strength_min, allocs_state], axis=-1)
    with tf.variable_scope("fc_lyr1", reuse=True):
        weights = tf.get_variable(name="w")
        biases = tf.get_variable(name="b")
        pairs_features = tf.reshape(pairs_features, [-1, 6])
        fc_lyr1_outputs = fully_connected_layer(pairs_features, weights, biases)

    with tf.variable_scope("fc_lyr2", reuse=True):
        weights = tf.get_variable(name="w")
        bias = tf.get_variable(name="b")
        fc_lyr2_outputs = fully_connected_layer(fc_lyr1_outputs, weights, bias)

    with tf.variable_scope("fc_lyr3", reuse=True):
        weights = tf.get_variable(name="w")
        bias = tf.get_variable(name="b")
        fc_lyr3_outputs = fully_connected_layer_final(fc_lyr2_outputs, weights, bias)

    network_outputs = tf.reshape(fc_lyr3_outputs, [-1, N])

    return network_outputs  # return the current state of allocations (before taking sigmoid)

# [Train Graph]
def train_network():
    steps_amount = 5
    g_train = tf.Graph()
    with g_train.as_default():
        placeholders = get_placeholders()
        if(Unsupervised_Training):
            placeholders['gains_diagonal'] = tf.placeholder(tf.float32, shape=[amount_per_batch, N], name='diagonal_gains_placeholder')
            placeholders['gains_nondiagonal'] = tf.placeholder(tf.float32, shape=[amount_per_batch, N, N], name='non_diagonal_gains_placeholder')
        get_parameters()
        allocs_state = tf.ones(shape=[amount_per_batch, N])
        for i in range(0, steps_amount):
            allocs_state_logits = iteration_step(allocs_state, placeholders)
            stochastic_mask = tf.cast(tf.random_uniform(shape=[amount_per_batch, N]) >= 0.5, tf.float32)
            allocs_state = allocs_state * (tf.ones([amount_per_batch, N], tf.float32) - stochastic_mask) + tf.sigmoid(allocs_state_logits) * stochastic_mask
        CE = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=placeholders['schedule_label'], logits=allocs_state_logits))
        if(Unsupervised_Training):
            SINRs_numerators = allocs_state * placeholders['gains_diagonal']  # layouts X N
            SINRS_denominators = tf.squeeze(tf.matmul(placeholders['gains_nondiagonal'], tf.expand_dims(allocs_state, axis=-1))) + output_noise_power / tx_power  # layouts X N
            SINRS = SINRs_numerators / SINRS_denominators  # layouts X N
            rough_rates = tf.log(1 + SINRS / SNR_gap)  # layouts X N
            sumrate = tf.reduce_sum(rough_rates)
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(-sumrate)
        else:
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(CE)
            sumrate = tf.constant(0)  # dummy return placeholder
        outputs_final = tf.cast(allocs_state >= 0.5, tf.int32, name="casting_to_scheduling_output")
    return g_train, CE, sumrate, train_step, outputs_final, placeholders

def test_network(steps_amount):
    g_test = tf.Graph()
    with g_test.as_default():
        placeholders = get_placeholders()
        placeholders['subset_links'] = tf.placeholder_with_default(input=tf.ones([amount_per_batch, N]), shape=[amount_per_batch, N], name='subset_links_placeholder')
        get_parameters() # set up parameters to be reused in iteration_step function call
        all_timesteps_allocs = []
        allocs_state = tf.ones(shape=[amount_per_batch, N]) * placeholders['subset_links']
        for i in range(0, steps_amount):
            allocs_state_logits = iteration_step(allocs_state, placeholders)
            stochastic_mask = tf.cast(tf.random_uniform(shape=[amount_per_batch, N]) >= 0.5, tf.float32)
            allocs_state = allocs_state * (tf.ones([amount_per_batch, N], tf.float32) - stochastic_mask) + tf.sigmoid(allocs_state_logits) * stochastic_mask
            allocs_state = allocs_state * placeholders['subset_links']
            all_timesteps_allocs.append(allocs_state)
        outputs_final = tf.cast(allocs_state >= 0.5, tf.int32, name="casting_to_scheduling_output")
    print("Tensorflow Graph Built Successfully!")
    return g_test, outputs_final, all_timesteps_allocs, placeholders

# SumRate testing
def network_inference(general_para, time_complexity=False):
    steps_amount_test = 20
    N_test, layouts_amount = general_para.pairs_amount, general_para.test_data_info["layouts"]
    global N
    N = N_test
    raw_data = utils.load_raw_data(general_para, model_para, ['test'])
    if (time_complexity):
        layouts_amount = 1
        for field_key in ['locations', 'pair_dists', 'tx_indices', 'rx_indices', 'pair_tx_convfilter_indices', 'pair_rx_convfilter_indices']:
            raw_data['test'][field_key] = np.expand_dims(raw_data['test'][field_key][0], axis=0)
    general_para.amount_per_batch = min(general_para.amount_per_batch, layouts_amount)
    global amount_per_batch
    amount_per_batch = general_para.amount_per_batch
    test_data = utils.prepare_batches(general_para, raw_data['test'])
    test_data = utils.add_appended_indices(general_para, test_data)
    batches_amount = np.shape(test_data['locations'])[0]

    # create the network graph
    g_test, outputs_final, all_timesteps_allocs, placeholders = test_network(steps_amount=steps_amount_test)

    model_loc = general_para.base_dir + model_para.model_loc
    with g_test.as_default():
        saver = tf.train.Saver()
        with tf.Session() as sess:
            print("Restoring model from: {}".format(model_loc))
            saver.restore(sess, model_loc)
            allocs = []
            neural_net_time = 0
            for j in range(batches_amount):
                if (((j + 1) / batches_amount * 100) % 10 == 0):
                    print("{}/{} minibatch... ".format(j + 1, batches_amount))
                start_time = time.time()
                allocs_perBatch = sess.run(outputs_final,
                                        feed_dict={placeholders['tx_indices_hash']: test_data['tx_indices_hash'][j],
                                                   placeholders['rx_indices_hash']: test_data['rx_indices_hash'][j],
                                                   placeholders['tx_indices_extract']: test_data['tx_indices_ext'][j],
                                                   placeholders['rx_indices_extract']: test_data['rx_indices_ext'][j],
                                                   placeholders['pair_tx_convfilter_indices']: test_data['pair_tx_convfilter_indices'][j],
                                                   placeholders['pair_rx_convfilter_indices']: test_data['pair_rx_convfilter_indices'][j] })
                neural_net_time += time.time() - start_time
                allocs.append(allocs_perBatch)
    neural_net_time_per_layout = neural_net_time / layouts_amount
    print("{} layouts with {} links, it took {} seconds; {} seconds per layout.".format(
        batches_amount * general_para.amount_per_batch, N, neural_net_time, neural_net_time_per_layout))
    allocs = np.array(allocs)
    assert np.shape(allocs) == (batches_amount, general_para.amount_per_batch, N), "Wrong shape: {}".format(np.shape(allocs))
    allocs = np.reshape(allocs, [-1, N])

    return allocs, neural_net_time_per_layout

# Weighted SumRate testing
def network_inference_weighted(general_para, gains_diagonal, gains_nondiagonal):
    steps_amount_test = 20
    N_test, layouts_amount, slots_per_layout = general_para.pairs_amount, general_para.test_data_info["layouts"], general_para.test_data_info["slots_per_layout"]
    global N
    N = N_test
    print("[ConvNetSumRateV10 network inference weighted] Starting with N={}; {} Layouts; {} Time slots......".format(
        N, layouts_amount, slots_per_layout))
    general_para.amount_per_batch = layouts_amount  # for weighted sumrate case, should be small enough amount of layouts
    global amount_per_batch
    amount_per_batch = general_para.amount_per_batch
    # load test data
    raw_data = utils.load_raw_data(general_para, model_para, ['test'])
    test_data = utils.prepare_batches(general_para, raw_data['test'])
    test_data = utils.add_appended_indices(general_para, test_data)
    batches_amount = np.shape(test_data['locations'])[0]
    print("Test batch amount: {}; with {} samples per batch".format(batches_amount, general_para.amount_per_batch))

    # create the network graph
    g_test, outputs_final, all_timesteps_allocs, placeholders = test_network(steps_amount=steps_amount_test)

    model_loc = general_para.base_dir + model_para.model_loc
    with g_test.as_default():
        saver = tf.train.Saver()
        with tf.Session() as sess:
            print("Restoring previously trained model from: {}".format(model_loc))
            saver.restore(sess, model_loc)
            total_time = 0
            allocs = []; rates = []; subsets = []; orig_prop_weights = []
            weights_orig = np.ones([layouts_amount, N]); weights_binary = np.ones([layouts_amount, N])
            for i in range(1, slots_per_layout + 1):
                if ((i / slots_per_layout * 100) % 20 == 0):
                    print("{}/{} time slots".format(i, slots_per_layout))
                start_time = time.time()
                allocs_oneslot = sess.run(outputs_final,
                                          feed_dict={placeholders['tx_indices_hash']: test_data['tx_indices_hash'][0],
                                                     placeholders['rx_indices_hash']: test_data['rx_indices_hash'][0],
                                                     placeholders['tx_indices_extract']: test_data['tx_indices_ext'][0],
                                                     placeholders['rx_indices_extract']: test_data['rx_indices_ext'][0],
                                                     placeholders['pair_tx_convfilter_indices']: test_data['pair_tx_convfilter_indices'][0],
                                                     placeholders['pair_rx_convfilter_indices']: test_data['pair_rx_convfilter_indices'][0],
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
    print("{} layouts with {} links over {} timeslots, it took {} seconds.".format(layouts_amount, N, slots_per_layout, total_time))
    allocs = np.transpose(np.array(allocs), (1, 0, 2))
    assert np.shape(allocs) == (layouts_amount, slots_per_layout, N), "Wrong shape: {}".format(np.shape(allocs))
    rates = np.transpose(np.array(rates), (1, 0, 2))
    assert np.shape(rates) == (layouts_amount, slots_per_layout, N), "Wrong shape: {}".format(np.shape(rates))
    subsets = np.transpose(np.array(subsets), (1, 0, 2))
    assert np.shape(subsets) == (layouts_amount, slots_per_layout, N), "Wrong shape: {}".format(np.shape(subsets))
    orig_prop_weights = np.transpose(np.array(orig_prop_weights), (1, 0, 2))
    assert np.shape(orig_prop_weights) == (layouts_amount, slots_per_layout, N), "Wrong shape: {}".format(np.shape(orig_prop_weights))
    np.save(general_para.base_dir+"SanityChecks/Weighted_SumRate_Opt/Conv_V10/"+"allocs.npy", allocs)
    np.save(general_para.base_dir+"SanityChecks/Weighted_SumRate_Opt/Conv_V10/"+"subsets.npy", subsets)
    np.save(general_para.base_dir+"SanityChecks/Weighted_SumRate_Opt/Conv_V10/"+"prop_weights.npy", orig_prop_weights)
    return allocs, rates, subsets


def plot_weights():
    get_parameters()
    saver = tf.train.Saver()
    model_loc = general_para.base_dir+model_para.model_loc
    with tf.Session() as sess:
        print("Restoring parameters from: {}".format(model_loc))
        saver.restore(sess, model_loc)
        weights_tensorlist = tf.get_collection("conv"); conv_weights_tensor = weights_tensorlist[0]
        assert conv_weights_tensor.name == "conv_lyr/w:0", "Tensor extraction failed, with wrong name: {}".format(conv_weights_tensor.name)
        conv_weights_raw = sess.run(conv_weights_tensor)
    conv_weights_raw = np.array(conv_weights_raw); assert np.shape(conv_weights_raw) == (filter_size_full, filter_size_full, 1, 1), "Wrong shape: {}".format(np.shape(conv_weights_raw))
    # Plot only raw weights (for better visualization)
    plt.title("Convolutional Filter Weights Plots")
    plt.title("Raw Weight Parameters")
    img1 = plt.imshow(np.squeeze(conv_weights_raw), cmap=plt.get_cmap("Greys"), origin="lower")
    plt.colorbar(img1, cmap=plt.get_cmap("Greys"))
    plt.show()
    return

def plot_allocs_evolution(layoutIndex):
    global amount_per_batch
    amount_per_batch = 1
    general_para.amount_per_batch = amount_per_batch
    steps_amount = 24
    raw_data = utils.load_raw_data(general_para, model_para, ['test']); raw_data = raw_data['test']
    for key in raw_data.keys():
        raw_data[key] = np.expand_dims(raw_data[key][layoutIndex], axis=0)  # maintain 1 X N shape
    show_weights = False
    try:
        weights_binary = np.load(general_para.data_folder + general_para.valid_data_info["folder"] + general_para.weights_file)
        weights_binary = np.expand_dims(weights_binary[layoutIndex], axis=0) # 1XN
        show_weights = True
    except FileNotFoundError:
        print("No importance weights defining subset: only evaluating sum rate!")
        weights_binary = np.ones([1, N])
    # form one batch (with 1 X 1 X N shape)
    sample_batch = utils.prepare_batches(general_para, raw_data)
    sample_batch = utils.add_appended_indices(general_para, sample_batch)

    g_test, outputs_final, all_timesteps_allocs, placeholders = test_network(steps_amount)

    model_loc = general_para.base_dir + model_para.model_loc
    with g_test.as_default():
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, model_loc)
            evo_allocs = sess.run(all_timesteps_allocs,
                                  feed_dict={placeholders['tx_indices_hash']: sample_batch['tx_indices_hash'][0],
                                             placeholders['rx_indices_hash']: sample_batch['rx_indices_hash'][0],
                                             placeholders['tx_indices_extract']: sample_batch['tx_indices_ext'][0],
                                             placeholders['rx_indices_extract']: sample_batch['rx_indices_ext'][0],
                                             placeholders['pair_tx_convfilter_indices']: sample_batch['pair_tx_convfilter_indices'][0],
                                             placeholders['pair_rx_convfilter_indices']: sample_batch['pair_rx_convfilter_indices'][0],
                                             placeholders['subset_links']: weights_binary})
    evo_allocs = np.array(evo_allocs); assert np.shape(evo_allocs) == (steps_amount, 1, N)
    evo_allocs = np.squeeze(evo_allocs)  # feedback steps amount X N
    fig = plt.figure(); plt.title("Allocation Evolution Over Network Iterations on Layout # {}".format(layoutIndex))
    ax = fig.gca(); ax.set_xticklabels([]); ax.set_yticklabels([])
    for i in range(steps_amount):
        ax = fig.add_subplot(4, 6, i + 1)
        v_locs = raw_data['locations'][0]; v_allocs = evo_allocs[i]
        tx_locs = v_locs[:, 0:2]; rx_locs = v_locs[:, 2:4]
        plt.scatter(tx_locs[:, 0], tx_locs[:, 1], c='r', label='Tx', s=5); plt.scatter(rx_locs[:, 0], rx_locs[:, 1], c='b', label='Rx', s=5)
        for j in range(N):  # plot all activated links
            if (show_weights):  # also plot binary weights subset
                if (np.squeeze(weights_binary)[j] == 1):
                    plt.plot([tx_locs[j, 0], rx_locs[j, 0]], [tx_locs[j, 1], rx_locs[j, 1]], 'b', linewidth=2.1)
            line_color = 1 - v_allocs[j]
            if line_color == 0:
                line_color = 0.0  # deal with 0 formatting error problem
            plt.plot([tx_locs[j, 0], rx_locs[j, 0]], [tx_locs[j, 1], rx_locs[j, 1]], '{}'.format(line_color))  # have to do 1 minus since the smaller the number the darker it gets
        ax.set_xticklabels([]); ax.set_yticklabels([])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

    fig = plt.figure(); plt.title("Quantized Allocation Evolution Over Network Iterations on Layout # {}".format(layoutIndex))
    plt.axis('off')
    for i in range(steps_amount):
        ax = fig.add_subplot(4, 6, i + 1)
        v_locs = raw_data['locations'][0]; v_allocs = evo_allocs[i]
        tx_locs = v_locs[:, 0:2]; rx_locs = v_locs[:, 2:4]
        plt.scatter(tx_locs[:, 0], tx_locs[:, 1], c='r', label='Tx', s=5); plt.scatter(rx_locs[:, 0], rx_locs[:, 1], c='b', label='Rx', s=5)
        for j in range(N):  # plot all activated links
            if (show_weights):  # also plot binary weights subset
                if (np.squeeze(weights_binary)[j] == 1):
                    plt.plot([tx_locs[j, 0], rx_locs[j, 0]], [tx_locs[j, 1], rx_locs[j, 1]], 'b', linewidth=2.1)
            if v_allocs[j] >= 0.5:
                plt.plot([tx_locs[j, 0], rx_locs[j, 0]], [tx_locs[j, 1], rx_locs[j, 1]], 'r')
        ax.set_xticklabels([]); ax.set_yticklabels([])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    return

if (__name__ == '__main__'):
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--plot', help='Whether plotting Conv/FC weights', default=False)
    parser.add_argument('--evoIndex', help='Index for plotting allocation evolution', default=False)
    parser.add_argument('--initFC', help='Whether initialize FC parameters', default=False)
    parser.add_argument('--initMisc', help='Whether refreshing misc parameters within network', default=False)
    parser.add_argument('--initAll', help='Whether train completely from scratch', default=False)
    args = parser.parse_args()
    if (args.plot):
        print("Plotting Weights...")
        plot_weights()
        print("Plotting Finished Successfully!")
        exit(0)
    if (args.evoIndex):
        layoutIndex = int(args.evoIndex)
        print("Plotting allocation evolution on layout indexed {} in validation set...".format(layoutIndex))
        plot_allocs_evolution(layoutIndex)
        print("Plotting Finished Successfully!")
        exit(0)

    print("Loading raw data...")
    raw_data = utils.load_raw_data(general_para, model_para, ['train', 'valid'])
    train_batches_amount = round(np.shape(raw_data['train']['locations'])[0] / general_para.amount_per_batch)
    valid_batches_amount = round(np.shape(raw_data['valid']['locations'])[0] / general_para.amount_per_batch)
    FP_valid_active_ratio = np.mean(raw_data['valid']['FP_allocations']); FP_valid_std = np.std(raw_data['valid']['FP_allocations'])
    print("Training {} batches; Validation {} batches.".format(train_batches_amount, valid_batches_amount))

    g_train, CE, sumrate, train_step, outputs_final, placeholders = train_network()
    model_loc = general_para.base_dir + model_para.model_loc
    with g_train.as_default():
        with tf.Session() as sess:
            save_saver = tf.train.Saver()
            train_costs = []; valid_costs = []
            train_sumrates = []; valid_sumrates = []
            # Dividing and processing training/validation data
            train_batches = utils.divide_data_batches(general_para, raw_data['train'])
            train_batches = utils.add_appended_indices(general_para, train_batches)
            valid_batches = utils.divide_data_batches(general_para, raw_data['valid'])
            valid_batches = utils.add_appended_indices(general_para, valid_batches)
            if (args.initFC):
                print("Train Option: Initialize FC parameters to train from scratch...")
                load_saver = tf.train.Saver(tf.get_collection("conv"))
                print("First initialize all parameters from scratch...")
                sess.run(tf.global_variables_initializer())
                print("Restoring previously trained model conv parameters from: {}".format(model_loc))
                load_saver.restore(sess, model_loc)
            elif (args.initMisc):
                print("Train Option: Refresh Misc Parameters for Adam Optimizer...")
                conv_load_saver = tf.train.Saver(tf.get_collection("conv"))
                fc_load_saver = tf.train.Saver(tf.get_collection("fc"))
                print("First initialize all parameters from scratch...")
                sess.run(tf.global_variables_initializer())
                print("Restoring conv parameters from: {}".format(model_loc))
                conv_load_saver.restore(sess, model_loc)
                print("Restoring FC parameters from: {}".format(model_loc))
                fc_load_saver.restore(sess, model_loc)
            elif (args.initAll):
                print("Train Option: Train network completely from scratch")
                sess.run(tf.global_variables_initializer())
            else:
                print("[Train Option] Reload all parameters from {} for resume training...".format(model_loc))
                save_saver.restore(sess, model_loc)
            print("Model Parameters Preparation finished!")
            for i in range(1, general_para.epoches_amount + 1):
                print("Epoch #{}:".format(i))
                train_cost_sum = 0
                train_sumrate_sum = 0
                for j in range(train_batches_amount):
                    if (not Unsupervised_Training):
                        train_dict = {placeholders['tx_indices_hash']: train_batches['tx_indices_hash'][j],
                                      placeholders['rx_indices_hash']: train_batches['rx_indices_hash'][j],
                                      placeholders['tx_indices_extract']: train_batches['tx_indices_ext'][j],
                                      placeholders['rx_indices_extract']: train_batches['rx_indices_ext'][j],
                                      placeholders['pair_tx_convfilter_indices']:
                                          train_batches['pair_tx_convfilter_indices'][j],
                                      placeholders['pair_rx_convfilter_indices']:
                                          train_batches['pair_rx_convfilter_indices'][j],
                                      placeholders['schedule_label']: train_batches['FP_allocations'][j]}
                    else:
                        train_dict = {placeholders['tx_indices_hash']: train_batches['tx_indices_hash'][j],
                                      placeholders['rx_indices_hash']: train_batches['rx_indices_hash'][j],
                                      placeholders['tx_indices_extract']: train_batches['tx_indices_ext'][j],
                                      placeholders['rx_indices_extract']: train_batches['rx_indices_ext'][j],
                                      placeholders['pair_tx_convfilter_indices']:
                                          train_batches['pair_tx_convfilter_indices'][j],
                                      placeholders['pair_rx_convfilter_indices']:
                                          train_batches['pair_rx_convfilter_indices'][j],
                                      placeholders['schedule_label']: train_batches['FP_allocations'][j],
                                      placeholders['gains_diagonal']: train_batches['gains_diagonal'][j],
                                      placeholders['gains_nondiagonal']: train_batches['gains_nondiagonal'][j]}
                    train_cost_minibatch, train_sumrate_minibatch, _ = sess.run([CE, sumrate, train_step], feed_dict=train_dict)
                    train_cost_sum += train_cost_minibatch
                    train_sumrate_sum += train_sumrate_minibatch
                    if((j+1)%100 == 0):
                        train_costs.append(train_cost_sum/(j+1))
                        train_sumrates.append(train_sumrate_sum/(j+1))
                        # Validation
                        valid_cost_sum = 0
                        valid_sumrate_sum = 0
                        valid_allocations = []
                        for k in range(valid_batches_amount):
                            if (not Unsupervised_Training):
                                valid_dict = {placeholders['tx_indices_hash']: valid_batches['tx_indices_hash'][k],
                                              placeholders['rx_indices_hash']: valid_batches['rx_indices_hash'][k],
                                              placeholders['tx_indices_extract']: valid_batches['tx_indices_ext'][k],
                                              placeholders['rx_indices_extract']: valid_batches['rx_indices_ext'][k],
                                              placeholders['pair_tx_convfilter_indices']: valid_batches['pair_tx_convfilter_indices'][k],
                                              placeholders['pair_rx_convfilter_indices']: valid_batches['pair_rx_convfilter_indices'][k],
                                              placeholders['schedule_label']: valid_batches['FP_allocations'][k]}
                            else:
                                valid_dict = {placeholders['tx_indices_hash']: valid_batches['tx_indices_hash'][k],
                                              placeholders['rx_indices_hash']: valid_batches['rx_indices_hash'][k],
                                              placeholders['tx_indices_extract']: valid_batches['tx_indices_ext'][k],
                                              placeholders['rx_indices_extract']: valid_batches['rx_indices_ext'][k],
                                              placeholders['pair_tx_convfilter_indices']: valid_batches['pair_tx_convfilter_indices'][k],
                                              placeholders['pair_rx_convfilter_indices']: valid_batches['pair_rx_convfilter_indices'][k],
                                              placeholders['schedule_label']: valid_batches['FP_allocations'][k],
                                              placeholders['gains_diagonal']: valid_batches['gains_diagonal'][k],
                                              placeholders['gains_nondiagonal']: valid_batches['gains_nondiagonal'][k]}
                            valid_cost_minibatch, valid_sumrate_minibatch, valid_allocations_batch = sess.run([CE, sumrate, outputs_final], feed_dict=valid_dict)
                            valid_cost_sum += valid_cost_minibatch
                            valid_sumrate_sum += valid_sumrate_minibatch
                            valid_allocations.append(valid_allocations_batch)
                        valid_costs.append(valid_cost_sum/valid_batches_amount)
                        valid_sumrates.append(valid_sumrate_sum/valid_batches_amount)
                        print("Minibatch #{}/{} [T] avg cost: {} | [V] avg cost: {}".format(j + 1, train_batches_amount,
                                                                                            round(train_costs[-1], 3),
                                                                                            round(valid_costs[-1], 3)))
                        print("                 [V] net active ratio: {}% | FP active ratio: {}%".format(
                            round(np.mean(valid_allocations) * 100, 2), round(FP_valid_active_ratio * 100, 2)))
                        print(
                            "                 [V] net std: {} | FP std: {}".format(round(np.std(valid_allocations), 2),
                                                                                   round(FP_valid_std, 2)))
                        if (Unsupervised_Training):
                            print("                 [T] sum rate: {} | [V] sum rate: {}".format(train_sumrates[-1],
                                                                                                valid_sumrates[-1]))
                        np.save("train_costs.npy", train_costs)
                        np.save("valid_costs.npy", valid_costs)
                        np.save("train_sumrates.npy", train_sumrates)
                        np.save("valid_sumrates.npy", valid_sumrates)
                        save_path = save_saver.save(sess, model_loc)
                        print("Model saved at {}!".format(save_path))
            # Finished training, save model
            print("Training iterations finished, saving model...")
            save_path = save_saver.save(sess, model_loc)
            print("Model saved at {}!".format(save_path))
    print("Training Session finished successfully!")

    print("Script Finished Successfully!")
