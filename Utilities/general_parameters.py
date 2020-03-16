# Storing all general parameters about environmental settings

import numpy as np

class parameters():
    def __init__(self, pairs_amount=50):
        # general training setting
        self.pairs_amount = pairs_amount
        self.field_length = 500 # originally was 1000 X 1000 for FP paper assumption
        self.train_data_info = {"layouts": 500000,
                                "slots_per_layout": 1,
                                "folder": "Train/"}
        self.valid_data_info = {"layouts": 1000,
                                "slots_per_layout": 1,
                                "folder": "Valid/"}
        self.test_data_info = {"layouts": 20,
                               "slots_per_layout": 500,
                               "folder": "Test/"}
        self.data_info = {"train": self.train_data_info,
                          "valid": self.valid_data_info,
                          "test": self.test_data_info}
        # self.base_dir = "C:/Users/willc1/Documents/Github/Master_Communication/" # subject to change when running from different machines
        self.base_dir = "/home/will/Master_Communication/" # subject to change when running from different machines
        # self.base_dir = "/home/ubuntu/Master_Communication/" # subject to change when running from different machines
        # FPLinQ setting
        self.FP_iter_amount = 100
        self.quantize_levels = 2  # power control setting
        self.alpha = 0.95  # for weights updates
        # for files saved after generation
        self.data_folder = self.base_dir+"Data_Samples/"
        self.locs_file = "locs_{}_meters_{}_pairs.npy".format(self.field_length, self.pairs_amount)
        self.dists_file = "dists_{}_meters_{}_pairs.npy".format(self.field_length, self.pairs_amount)
        self.weights_file = "weights_{}_meters_{}_pairs.npy".format(self.field_length, self.pairs_amount) # seldomly used
        self.G_file = "gains_{}_meters_{}_pairs.npy".format(self.field_length, self.pairs_amount)
        self.G_file_fastFading = "gains_fade_{}_meters_{}_pairs.npy".format(self.field_length, self.pairs_amount)
        self.FP_alloc_file = "FP_allocs_{}_meters_{}_pairs.npy".format(self.field_length, self.pairs_amount)
        self.FP_alloc_file_fastFading = "FP_allocs_fade_{}_meters_{}_pairs.npy".format(self.field_length, self.pairs_amount)
        self.Optimal_alloc_file = "Optimal_allocs_{}_meters_{}_pairs.npy".format(self.field_length, self.pairs_amount)
        self.file_names = {"locs": self.locs_file,
            "dists": self.dists_file,
            "gains": self.G_file,
            "gains_fade": self.G_file_fastFading,
            "FP": self.FP_alloc_file,
            "FP_fade": self.FP_alloc_file_fastFading
        }
        # for log file
        self.log_folder = self.base_dir+"Logs/"
        self.log_file = "log.txt"
        # specific channel setting
        self.shortest_dist = 0.5
        self.longest_dist = 3
        self.bandwidth = 5e6
        self.carrier_f = 2.4e9
        self.tx_height = 1.5
        self.rx_height = 1.5
        self.antenna_gain_decibel = 2.5
        self.tx_power_milli_decibel = 40
        self.tx_power = np.power(10, (self.tx_power_milli_decibel-30)/10)
        self.noise_density_milli_decibel = -169
        self.input_noise_power = np.power(10, ((self.noise_density_milli_decibel-30)/10)) * self.bandwidth
        self.output_noise_power = self.input_noise_power
        self.SNR_gap_dB = 6
        self.SNR_gap = np.power(10, self.SNR_gap_dB/10)
        self.max_rate_allowed = 15 * self.bandwidth # with 15bps/Hz being the fundamental upperbound
        # Machine Learning training setting
        self.amount_per_batch = 1000
        self.epoches_amount = 100000
        self.stddev_init = 0.01 # truncated norm initialization standard deviation
        self.learning_rate = 1e-4 # used with Adam optimizer
        # occupancy grid setting
        self.cell_length = 5
        self.grid_amount_each_side = np.round(self.field_length/self.cell_length).astype(int)
        self.grid_amount = np.array([self.grid_amount_each_side, self.grid_amount_each_side])
