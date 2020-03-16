# Script for generating and storing training pairs with FPLinQ results
import numpy as np
import general_parameters
import utils
import os
import time
import sys

# Generate both the training&validation samples and testing samples
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--N', help='Amount of D2D Links to Generate', default=50)
    parser.add_argument('--train', help='Whether generate training data', default=False)
    parser.add_argument('--valid', help='Whether generate validation or testing data', default=False)
    parser.add_argument('--randPairdist', help='Whether generate with random pairdists', default=False)
    parser.add_argument('--fastFading', help='Whether generate with fast fading in CSI', default=False)
    args = parser.parse_args()
    data_type_list_arg = []
    if (args.train):
        print("[generator.py] Generating training data...")
        data_type_list_arg.append('train')
    if (args.valid):
        print("[generator.py] Generating validation data...")
        data_type_list_arg.append('valid')
    if (data_type_list_arg == []):
        print("[generator.py] Not choosing to generate either training or validation data. Exiting...")
        exit(0)

    if (args.randPairdist):
        rand_pairdist_arg = True
    else:
        rand_pairdist_arg = False
    if (args.fastFading):
        fast_fading_arg = True
    else:
        fast_fading_arg = False

    N = int(args.N)
    general_para = general_parameters.parameters(N)
    utils.generate(general_para, data_type_list=data_type_list_arg, rand_pairdist=rand_pairdist_arg, fast_fading=fast_fading_arg)
    print("[generator.py] Script Finished Successfully!")
