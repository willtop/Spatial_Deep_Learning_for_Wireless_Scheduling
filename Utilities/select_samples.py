# Select subset of samples with specified indexing
# Assume large samples set is stored in the file with suffix "_large"

import numpy as np
import sys
import parameters

if(__name__=="__main__"):
    if(len(sys.argv) != 5):
        print("Usage: select_samples.py [train|test] model_type starting_index ending_index")
        exit(-1)
    # find corresponding training/testing samples file
    para = parameters.fc_parameters(sys.argv[2])
    N = para.pairs_amount
    if(sys.argv[1] == "train"):
        subfolder = para.train_folder
    elif(sys.argv[1] == "test"):
        subfolder = para.test_folder
    else:
        print("Invalid argument {}".format(sys.argv[1]))
        exit(-1)
    start_ind = int(sys.argv[3]); end_ind = int(sys.argv[4])
    print("Start loading source files with large set of samples...")
    coefficients_src = np.load("../"+subfolder+"channel_coefficients_{}_pairs_large.npy".format(N))
    locations_src = np.load("../"+subfolder+"locations_{}_pairs_large.npy".format(N))
    FP_allocs_src = np.load("../"+subfolder+"FP_allocs_{}_pairs_large.npy".format(N))
    assert np.shape(coefficients_src)[0] == np.shape(locations_src)[0] == np.shape(FP_allocs_src)[0], "amounts of samples included in each original file are not aligning!"
    print("Source files include {} samples...".format(np.shape(coefficients_src)[0]))
    print("Start selecting samples and saving them...")
    coefficients = coefficients_src[start_ind: end_ind]
    locations = locations_src[start_ind: end_ind]
    FP_allocs = FP_allocs_src[start_ind: end_ind]
    assert np.shape(coefficients)[0] == np.shape(locations)[0] == np.shape(FP_allocs)[0], "amounts of samples included in each extracted file are not aligning!"
    print("Saving {} selected samples...".format(np.shape(coefficients)[0]))
    assert np.shape(coefficients)[1:] == (N, N), "selected coefficients with wrong shape: {}".format(np.shape(coefficients))
    assert np.shape(locations)[1:] == (N, 4), "selected locations with wrong shape: {}".format(np.shape(locations))
    assert np.shape(FP_allocs)[1:] == (N, 1), "selected FP allocations with wrong shape: {}".format(np.shape(FP_allocs))
    np.save("../"+subfolder+para.h_file, coefficients)
    np.save("../"+subfolder+para.loc_file, locations)
    np.save("../"+subfolder+para.FP_alloc_file, FP_allocs)
    print("Files successfully saved at: {}!".format(subfolder))
    print("Script Finished Successfully!")

