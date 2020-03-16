# storing all model specific parameters

import numpy as np

class parameters():
    def __init__(self, model_type_arg):
        self.model_type = model_type_arg
        # for base folder
        self.model_dirs_bank = {"fc_net_v0": "Fully_Connected_Model_V0/Trained_FC_V0_Models/",
                                "fc_net_v1": "Fully_Connected_Model_V1/Trained_FC_V1_Models/",
                                "fc_net_v2": "Fully_Connected_Model_V2/Trained_FC_V2_Models/",
                                "fc_net_v3": "Fully_Connected_Model_V3/Trained_Models/",
                                "fc_net_toy_sch": "Fully_Connected_Model_Toy/Trained_Models/",
                                "fc_net_weights": "Fully_Connected_Model_Weights/Trained_FC_Weights_Models/",
                                "conv_net_v1": "Conv_Net_Model_V1/Trained_Conv_V1_Models/",
                                "conv_net_v2": "Conv_Net_Model_V2/Trained_Conv_V2_Models/",
                                "conv_net_v3": "Conv_Net_Model_V3/Trained_Conv_V3_Models/",
                                "conv_net_v3_unsup": "Conv_Net_Model_V3/Trained_Conv_V3_Models_Unsupervised/",
                                "conv_net_v4": "Conv_Net_Model_V4/Trained_Conv_V4_Models/",
                                "conv_net_v5": "Conv_Net_Model_V5/Trained_Conv_V5_Models/",
                                "conv_net_v5_weights": "Conv_Net_Model_V5/Trained_Conv_V5_Weights_Models/",
                                "conv_net_v6": "Conv_Net_Model_V6/Trained_Conv_V6_Models/",
                                "conv_net_v7_sch": "Conv_Net_Model_V7/Trained_Conv_V7_Models_scheduling/",
                                "conv_net_v7_pc": "Conv_Net_Model_V7/Trained_Conv_V7_Models_powercontrol/",
                                "conv_net_v7_unsupervised": "Conv_Net_Model_V7/Trained_Conv_V7_Unsupervised_Models/",
                                "conv_net_v7_hybrid": "Conv_Net_Model_V7/Trained_Conv_V7_Hybrid_Models/",
                                "conv_net_v7_simplified": "Conv_Net_Model_V7/Trained_Conv_V7_Simplified_Models/",
                                "conv_net_v7_simplest": "Conv_Net_Model_V7/Trained_Conv_V7_Simplest_Models/",
                                "conv_net_v8": "Conv_Net_Model_V8/Trained_Conv_V8_Models/",
                                "conv_net_v8_fp": "Conv_Net_Model_V8/Trained_Conv_V8_FP_Models/",
                                "conv_net_v9": "Conv_Net_Model_V9/Trained_Conv_V9_Models/",
                                "conv_net_snr": "Conv_Net_Model_SNR/Trained_Conv_SNR_Models/",
                                "conv_net_coes": "Conv_Net_Model_SNR/Trained_Conv_Coes_Models/",
                                "conv_net_activeratio": "Conv_Net_Model_ActiveRatio/Trained_Conv_ActiveRatio_Models/",
                                "conv_net_v10": "Conv_Net_Model_V10/Trained_Models/",
                                "conv_net_sumrate_v1": "Conv_Net_Model_SumRate_V1/Trained_Models/",
                                "conv_net_sumrate_v2": "Conv_Net_Model_SumRate_V2/Trained_Models/",
                                "conv_net_sumrate_v3": "Conv_Net_Model_SumRate_V3/Trained_Models/",
                                "conv_net_sumrate_v4": "Conv_Net_Model_SumRate_V4/Trained_Models/",
                                "conv_net_sumrate_v4_subsets": "Conv_Net_Model_SumRate_V4/Trained_Models_Subsets/",
                                "conv_net_sumrate_v5": "Conv_Net_Model_SumRate_V5/Trained_Models/",
                                "conv_net_sumrate_v6": "Conv_Net_Model_SumRate_V6/Trained_Models/",
                                "conv_net_sumrate_v6_subsets": "Conv_Net_Model_SumRate_V6/Trained_Models_Subsets/",
                                "conv_net_sumrate_v7": "Conv_Net_Model_SumRate_V7/Trained_Models/",
                                "conv_net_weighted_sumrate_v4": "Conv_Net_Model_Weighted_SumRate_V4/Trained_Models/",
                                "conv_net_sumrate_v8": "Conv_Net_Model_SumRate_V8/Trained_Models/",
                                "conv_net_sumrate_v9": "Conv_Net_Model_SumRate_V9/Trained_Models/",
                                "conv_net_sumrate_v10": "Conv_Net_Model_SumRate_V10/Trained_Models/",
                                "active_prob_net_v1": "Active_Prob_Net_V1/Trained_V1_Models/",
                                "active_prob_net_v2": "Active_Prob_Net_V2/Trained_V2_Models/",
                                "dl_wmmse": "DL_on_WMMSE_Work/Trained_DL_Models/",
                                "pure_importance_weights": "Pure_Importance_Weights_Net/Trained_Models/",
                                "pure_importance_weights_unsupervised": "Pure_Importance_Weights_Net/Trained_Models_Unsupervised/"
                                }
        self.model_types = self.model_dirs_bank.keys()
        assert self.model_type in self.model_types, "Invalid argument {} for model type!".format(self.model_type)
        self.model_dir = self.model_dirs_bank[self.model_type]
        self.model_loc = self.model_dir + "{}_model.ckpt".format(self.model_type)
        # Conv filter setting, length of square should be odd number
        self.conv_filter_size_full = None
        self.conv_filter_size_medium = None
        self.conv_filter_size_small = None
        if (self.model_type == "conv_net_sumrate_v4"):
            self.conv_filter_size_full = 33
            self.conv_filter_size_medium = 15
            self.conv_filter_size_small = 5
        if (self.model_type == "conv_net_sumrate_v8"):
            self.conv_filter_size_full = 63
            self.conv_filter_size_small = 31
        if (self.model_type == "conv_net_sumrate_v9"):
            self.conv_filter_size_full = 63
        if (self.model_type == "conv_net_sumrate_v10"):
            self.conv_filter_size_full = 63
