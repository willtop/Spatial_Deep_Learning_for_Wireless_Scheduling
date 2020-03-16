# Utility functions used by other scripts

import numpy as np
import general_parameters
import model_parameters
import FPLinQ
import time

def rate_capping(general_para, rates):
    links_capped = np.sum(rates>general_para.max_rate_allowed)
    rates[rates>general_para.max_rate_allowed] = general_para.max_rate_allowed
    return rates, links_capped

def proportional_update_weights(general_para, weights, rates):
    return 1 / (general_para.alpha / weights + (1 - general_para.alpha) * rates)

# Input: allocs: layouts X N
#        gains_diagonal: layouts X N; gains_nondiagonal: layouts X N X N
# Output: SINRs: layouts X N
def compute_SINRs(general_para, allocs, gains_diagonal, gains_nondiagonal):
    assert np.shape(gains_diagonal) == np.shape(allocs)
    SINRs_numerators = allocs * gains_diagonal  # layouts X N
    SINRs_denominators = np.squeeze(np.matmul(gains_nondiagonal, np.expand_dims(allocs, axis=-1))) + general_para.output_noise_power / general_para.tx_power  # layouts X N
    SINRs = SINRs_numerators / SINRs_denominators  # layouts X N
    return SINRs

# Input: allocs: layouts X N
#        gains_diagonal: layouts X N; gains_nondiagonal: layouts X N X N
# Output: rates: layouts X N
def compute_rates(general_para, allocs, gains_diagonal, gains_nondiagonal):
    SINRs = compute_SINRs(general_para, allocs, gains_diagonal, gains_nondiagonal)
    rates = general_para.bandwidth * np.log2(1 + SINRs/general_para.SNR_gap) # layouts X N
    # Cap at max rate upper limit
    # rates, links_capped = rate_capping(general_para, rates)
    # return rates, links_capped
    return rates

def compute_direct_rates(general_para, gains_diagonal):
    # for now, assume constant channel tx power and weight
    SINRS = gains_diagonal*general_para.tx_power / general_para.output_noise_power
    rates = general_para.bandwidth * np.log2(1 + SINRS/general_para.SNR_gap)  # layouts X N
    return rates

def compute_direct_path_loss_decibel(general_para, pairwise_dists):
    N = general_para.pairs_amount; samples_amount = np.shape(pairwise_dists)[0]
    assert np.shape(pairwise_dists) == (samples_amount, N), "Wrong shape {}!".format(np.shape(pairwise_dists))
    Rbp = 4*general_para.tx_height*general_para.rx_height/(2.998e8 / general_para.carrier_f)
    exponents = np.zeros([samples_amount, N])
    exponents[pairwise_dists>Rbp] = -4; exponents[pairwise_dists<=Rbp] = -2
    return np.log10(np.power(pairwise_dists, exponents))

# compute channel gains
def compute_gains(general_para, distances, fast_fading=False):
    # extract general_parameters about channel setting
    N = np.shape(distances)[0] # Make this adjustable for plotting actual path loss script
    h1 = general_para.tx_height
    h2 = general_para.rx_height
    signal_lambda = 2.998e8 / general_para.carrier_f
    antenna_gain_decibel = general_para.antenna_gain_decibel
    # compute relevant quantity
    Rbp = 4*h1*h2 / signal_lambda
    Lbp = abs(20*np.log10(np.power(signal_lambda, 2) / (8*np.pi*h1*h2)))
    # compute coefficient matrix for each Tx/Rx pair
    sum_term = 20*np.log10(distances / Rbp)
    Tx_over_Rx = Lbp + 6 + sum_term + ((distances>Rbp).astype(int))*sum_term # adjust for longer path loss
    gains = -Tx_over_Rx + np.eye(N)*antenna_gain_decibel # only add antenna gain for direct channel
    # gain = gain + np.random.normal(size=[N,N]) * 10 # Optional adding shadow term
    gains = np.power(10, (gains/10)) # convert from decibel to absolute
    if(fast_fading):
        fast_fadings = (np.power(np.random.normal(loc=0, scale=1, size=[N,N]),2) + np.power(np.random.normal(loc=0, scale=1, size=[N,N]),2)) / 2
        gains_fade = gains * fast_fadings
        return gains, gains_fade
    return gains

# Generate uniformly distributed transceiver location pairs
# Use N sent from generator.py for synchronization reliability
def locs_generate(general_para):
    N = general_para.pairs_amount
    # first, generate transmitters' coordinates
    tx_xs = np.random.uniform(low=0, high=general_para.field_length, size=[N,1])
    tx_ys = np.random.uniform(low=0, high=general_para.field_length, size=[N,1])
    # generate rx one by one rather than N together to ensure checking validity one by one
    rx_xs = []; rx_ys = []
    for i in range(N):
        got_valid_rx = False
        while(not got_valid_rx):
            pair_dist = np.random.uniform(low=general_para.shortest_dist, high=general_para.longest_dist)
            pair_angles = np.random.uniform(low=0, high=np.pi*2)
            rx_x = tx_xs[i] + pair_dist * np.cos(pair_angles)
            rx_y = tx_ys[i] + pair_dist * np.sin(pair_angles)
            if(0<=rx_x<=general_para.field_length and 0<=rx_y<=general_para.field_length):
                got_valid_rx = True
        rx_xs.append(rx_x); rx_ys.append(rx_y)
    # For now, assuming equal weights and equal power, so not generating them
    locs = np.concatenate((tx_xs, tx_ys, rx_xs, rx_ys), axis=1)
    distances = np.zeros([N, N])
    # compute distance between every possible Tx/Rx pair
    for rx_index in range(N):
        for tx_index in range(N):
            tx_coor = locs[tx_index][0:2]
            rx_coor = locs[rx_index][2:4]
            # according to paper notation convention, Hij is from jth transmitter to ith receiver
            distances[rx_index][tx_index] = np.linalg.norm(tx_coor - rx_coor)

    return locs, distances

# Generate Location Samples with Specified Amount at Specified Locations
def generate(general_para, data_type_list, rand_pairdist=False, fast_fading=False):
    N, alpha = general_para.pairs_amount, general_para.alpha
    pairdist_arg = "random lower/upperbounds" if rand_pairdist else "{}~{}m".format(general_para.shortest_dist, general_para.longest_dist)
    fast_fading_arg = "Yes" if fast_fading else "No"
    print("<<<<<<<<<<<<utils.generate(): Pairs Amount: {}; Region side length {}; Pairdists Distribution: {}; Fast Fading: {}>>>>>>>>>>>>".format(
        N, general_para.field_length, pairdist_arg, fast_fading_arg))
    for data_type in data_type_list:
        data_info = general_para.data_info[data_type]
        layouts, slots_per_layout, store_folder = data_info["layouts"], data_info["slots_per_layout"], data_info["folder"]
        print("<<<<<<<<Start {} data generation; with {} layouts and {} slots per layout...>>>>>>>".format(data_type, layouts, slots_per_layout))
        all_locs = []; all_gains = []; all_FP_allocs = []
        if(fast_fading):
            all_gains_fade = []; all_FP_allocs_fade = []
        all_FP_durations = []
        for i in range(1, layouts+1):
            if(rand_pairdist):
                general_para.shortest_dist = np.random.uniform(low=2, high=70)
                general_para.longest_dist = np.random.uniform(low=general_para.shortest_dist, high=70)
            locs, dists = locs_generate(general_para); all_locs.append(locs) #all_dists.append(dists)  # generated_locs: NX4; distances: NXN
            if(not fast_fading):
                gains = compute_gains(general_para, dists)
            else:
                gains, gains_fade = compute_gains(general_para, dists, fast_fading)
                all_gains_fade.append(gains_fade)
                gains_diagonal_fade = np.diag(gains_fade)
                gains_nondiagonal_fade = gains_fade * ((np.identity(N) < 1).astype(int));
            all_gains.append(gains)
            gains_diagonal = np.diag(gains)
            gains_nondiagonal = gains * ((np.identity(N) < 1).astype(int))
            weights = np.ones([N])
            for j in range(1, slots_per_layout+1):
                FP_allocs, FP_duration = FPLinQ.FP_optimize(general_para, gains, gains_diagonal, gains_nondiagonal, weights, general_para.FP_iter_amount)
                if(fast_fading):
                    FP_allocs_fade, FP_duration = FPLinQ.FP_optimize(general_para, gains_fade, gains_diagonal_fade, gains_nondiagonal_fade, weights, general_para.FP_iter_amount)
                    all_FP_allocs_fade.append(FP_allocs_fade)
                all_FP_allocs.append(FP_allocs); all_FP_durations.append(FP_duration)
                if(slots_per_layout>1):
                    assert not fast_fading
                    rates = np.squeeze(compute_rates(general_para, np.expand_dims(FP_allocs,axis=0), np.expand_dims(gains_diagonal,axis=0), np.expand_dims(gains_nondiagonal,axis=0)))
                    weights = proportional_update_weights(general_para, weights, rates)
            # save intermediate results
            data_save = dict()
            data_save["locs"] = all_locs
            data_save["FP"] = all_FP_allocs
            data_save["gains"] = all_gains
            if(fast_fading):
                data_save["FP_fade"] = all_FP_allocs_fade
                data_save["gains_fade"] = all_gains_fade
            if (i % 5000 == 0):
                print("At {}/{} layout, saving intermediate stage...".format(i, layouts))
                save_generated_files(general_para, store_folder, data_save)
        print("Mean FP duration: {} over {} times".format(np.mean(all_FP_durations), np.size(all_FP_durations)))
        print("Finished all generation, saving final data...")
        save_generated_files(general_para, store_folder, data_save)
    print("Generator Function Completed Successfully!")
    return

def save_generated_files(general_para, store_folder, data_save):
    print("[save_generated_files] Saving to {}...".format(general_para.data_folder + store_folder))
    for key in data_save.keys():
        data = np.array(data_save[key]); file_name = general_para.file_names[key]
        np.save(general_para.data_folder + store_folder + file_name, data)
    print("[save_generated_files] File saving Completed!")
    return

# Find binary approximation of importance weights, general_parallely over multiple layouts
def binary_importance_weights_approx(general_para, weights):
    N = general_para.pairs_amount
    layouts_amount = np.shape(weights)[0]
    assert np.shape(weights) == (layouts_amount, N)
    sorted_indices = np.argsort(weights, axis=1)
    weights_normalized = weights / np.linalg.norm(weights,axis=1,keepdims=True) # normalize to l2 norm 1
    # initialize variables
    binary_weights = np.zeros([layouts_amount, N])
    max_dot_product = np.zeros(layouts_amount)
    # use greedy to activate one at a time
    for i in range(N-1, -1, -1):
        binary_weights[np.arange(layouts_amount), sorted_indices[:,i]] = 1
        binary_weights_normalized = binary_weights/np.linalg.norm(binary_weights,axis=1,keepdims=True)
        current_dot_product = np.einsum('ij,ij->i', weights_normalized, binary_weights_normalized)
        binary_weights[np.arange(layouts_amount), sorted_indices[:,i]] = (current_dot_product >= max_dot_product).astype(int)
        max_dot_product = np.maximum(max_dot_product, current_dot_product)
    return binary_weights

def get_pair_indices(general_para, locations):
    N = general_para.pairs_amount
    assert np.shape(locations)==(N,2), "[get_pair_indices] input locations argument with wrong shape: {}".format(np.shape(locations))
    x_amount, y_amount = general_para.grid_amount
    locations_indices = np.floor(locations / np.array([general_para.cell_length, general_para.cell_length])).astype(int)
    # deal with stations located at upper boundaries
    locations_indices[locations_indices[:,0]>=x_amount, 0] = x_amount-1
    locations_indices[locations_indices[:,1]>=y_amount, 1] = y_amount-1
    assert np.shape(locations_indices)==(N,2), "[get_pair_indices] generated location indices with wrong shape: {}".format(locations_indices)
    return locations_indices

def append_indices_array(general_para, indices):
    N = general_para.n_links
    grid_amount = general_para.grid_amount
    n_layouts = np.shape(indices)[0]
    # Firstly, append at the end the indices that each link appears in the layout, parallelly over all layouts
    pad = np.tile(np.expand_dims(np.expand_dims(np.arange(n_layouts), -1), -1), [1, N, 1])
    indices_extract = np.concatenate([pad,indices], axis=2)
    assert np.shape(indices_extract) == (n_layouts, N, 3), "Wrong shape: {}".format(np.shape(indices_extract))
    # Secondly, append at the end the hash distinctive indices for summation among all links later
    hash_step = np.max(grid_amount) + 1
    hash_indices = np.dot(indices_extract, np.array([[hash_step**2], [hash_step],[1]]))
    indices_hash = np.concatenate([indices_extract, hash_indices], axis=-1)
    indices_hash = np.reshape(indices_hash, [n_layouts*N, 4])
    return indices_extract, indices_hash

# Processing test layout inputs into inputs for the conv net model
def process_layouts_inputs(general_para, layouts):
    N = general_para.n_links
    n_layouts = np.shape(layouts)[0]
    assert np.shape(n_layouts) == (n_layouts, N, 4)
    data_dict = dict()
    data_dict['layouts'] = layouts
    data_dict['pair_dists'] = np.linalg.norm(layouts[:, :, 0:2] - layouts[:, :, 2:4], axis=2)
    # compute station indices within the grid
    tx_indices = []
    rx_indices = []
    for i in range(n_layouts):
        txs = layouts[i, :, 0:2]
        rxs = layouts[i, :, 2:4]
        tx_indices.append(get_pair_indices(general_para, txs))
        rx_indices.append(get_pair_indices(general_para, rxs))
    data_dict['tx_indices'] = np.array(tx_indices)
    data_dict['rx_indices'] = np.array(rx_indices)
    # Assuming the convolutional filter size is 63X63
    conv_filter_center = np.floor(np.array([63, 63]) / 2).astype(int)
    data_dict['pair_tx_convfilter_indices'] = data_dict['tx_indices'] - data_dict['rx_indices'] + conv_filter_center
    data_dict['pair_rx_convfilter_indices'] = data_dict['rx_indices'] - data_dict['tx_indices'] + conv_filter_center
    # Add additional indices required by the neural network model to each tx/rx grid index
    data_dict['tx_indices_ext'], data_dict['tx_indices_hash'] = append_indices_array(general_para, data_dict['tx_indices'])
    data_dict['rx_indices_ext'], data_dict['rx_indices_hash'] = append_indices_array(general_para, data_dict['rx_indices'])
    return data_dict

# gains_diagonal & gains_nondiagonal: layouts_amount X N (X N)
def greedy_sumrate(general_para, gains_diagonal, gains_nondiagonal):
    layouts_amount = np.shape(gains_diagonal)[0]; N = general_para.pairs_amount
    sorted_links_indices = np.argsort(gains_diagonal, axis=-1)
    # Variables to update
    previous_sum_rates = np.zeros(layouts_amount)
    allocs = np.zeros([layouts_amount, N])
    current_interferences = np.zeros([layouts_amount, N])
    start_time = time.time()
    for i in range(N - 1, -1, -1):  # iterate from highest indices since strongest links are sorted last
        allocs[np.arange(layouts_amount), sorted_links_indices[:, i]] = 1
        current_signals = allocs * gains_diagonal  # O(N); shape: [layouts X N]
        current_interferences += gains_nondiagonal[np.arange(layouts_amount), :, sorted_links_indices[:, i]]  # O(1); shape: [layouts X N]
        current_SINRs = current_signals / (current_interferences + general_para.output_noise_power / general_para.tx_power)  # O(N); shape: [layouts X N]
        current_rates = general_para.bandwidth * np.log2(1 + current_SINRs / general_para.SNR_gap)  # O(N); shape: [layouts X N] (This is computed precisely as rate values for capping)
        # Perform capping
        # current_rates, _ = rate_capping(general_para, current_rates) # O(N); shape: [layouts X N]
        current_sum_rates = np.sum(current_rates, axis=-1)  # O(N); shape: [layouts]
        # schedule the ith shortest pair for samples that have sum rate improved
        allocs[np.arange(layouts_amount), sorted_links_indices[:, i]] = (current_sum_rates > previous_sum_rates).astype(int)
        # remove the interference corresponding to links got turned off
        current_interferences -= gains_nondiagonal[np.arange(layouts_amount), :, sorted_links_indices[:, i]] * np.expand_dims((current_sum_rates <= previous_sum_rates).astype(int), axis=-1)
        previous_sum_rates = np.maximum(current_sum_rates, previous_sum_rates)
    run_time = time.time() - start_time
    return allocs, run_time

# dists: samples X N X N
def covariances_generate(dists):
    layouts = np.shape(dists)[0]; N = np.shape(dists)[1]
    assert np.shape(dists) == (layouts, N, N)
    cov_mats = np.zeros([layouts,N,N])
    cov_mats[dists <= np.percentile(dists, 10, axis=(1,2), keepdims=True)] = -1
    cov_mats[:,np.eye(N)==1] = 1
    # as long as there is one of the two crosslink distances below threshold, mark the two links inversely correlated
    cov_mats = np.minimum(cov_mats, np.transpose(cov_mats, axes=(0,2,1)))
    assert (cov_mats == np.transpose(cov_mats, axes=(0,2,1))).all()
    return cov_mats

def cov_vectors_sampling(dists):
    layouts = np.shape(dists)[0]; N = np.shape(dists)[1]
    assert np.shape(dists) == (layouts, N, N), "Wrong Shape: {}".format(np.shape(dists))
    cov_mats = covariances_generate(dists)
    # sampling from Gaussian, then quantize to obtain scheduling
    vectors = []
    mean_vector = np.ones([N]) * 0.5
    start_time = time.time()
    for i in range(layouts):
        vector = np.random.multivariate_normal(mean=mean_vector, cov=cov_mats[i], check_valid='ignore')
        vectors.append(vector)
    vectors = np.array(vectors)
    vectors = (vectors >= 0.5).astype(int)  # quantize to scheduling outputs
    run_time = time.time() - start_time
    assert np.shape(vectors)==(layouts, N), "Wrong Shape: {}".format(np.shape(vectors))
    return vectors, run_time

# Input: shape: Tuple
def check_data_shape(data, shape):
    assert np.shape(data)==shape, "Wrong Shape: {}".format(np.shape(data))
    return