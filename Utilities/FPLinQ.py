# Implementation of FPLinQ algorithm including iterative optimization

import numpy as np
import time

# Optimize FP one single layout at a time
def FP_optimize(general_para, g, g_diag, g_nondiag, weights, iter_amount):
    weights = np.reshape(weights, [-1, 1])
    g_diag = np.reshape(g_diag, [-1, 1])
    N = np.shape(g)[0]
    x = np.ones([N, 1]) # All one initialization due to quantization need
    tx_power = general_para.tx_power
    output_noise_power = general_para.output_noise_power
    tx_powers = np.ones([N, 1]) * tx_power  # assume same power for each transmitter
    start_time = time.time()
    for i in range(iter_amount):
        # Compute z
        p_x_prod = x * tx_powers
        z_denominator = np.dot(g_nondiag, p_x_prod) + output_noise_power  # N X 1 result
        z_numerator = g_diag * p_x_prod  # N X 1 result
        z = z_numerator / z_denominator
        # compute y
        y_denominator = np.dot(g, p_x_prod) + output_noise_power  # N X 1 result
        y_numerator = np.sqrt(z_numerator * weights * (z + 1))  # N X 1 result
        y = y_numerator / y_denominator
        # compute x (continuous)
        x_denominator = np.dot(g.T, np.power(y, 2)) * tx_powers  # N X 1 result
        x_numerator = y * np.sqrt(weights * (z + 1) * g_diag * tx_powers)  # N X 1 result
        x_new = np.power(x_numerator / x_denominator, 2)
        x_new[x_new > 1] = 1  # thresholding at upperbound 1
        x = x_new
    x_final = (np.sqrt(x) >= 0.5).astype(int)
    FP_duration = time.time() - start_time
    x_final = np.squeeze(x_final) # (N,) shaped vector
    return x_final, FP_duration
