import math
import numpy as np
def _cal_freq_list(freq_init, frequency_num, max_radius, min_radius):
    if freq_init == "random":
        # the frequence we use for each block, alpha in ICLR paper
        # freq_list shape: (frequency_num)
        freq_list = np.random.random(size=[frequency_num]) * max_radius
    elif freq_init == "geometric":
        # freq_list = []
        # for cur_freq in range(frequency_num):
        #     base = 1.0/(np.power(max_radius, cur_freq*1.0/(frequency_num-1)))
        #     freq_list.append(base)

        # freq_list = np.asarray(freq_list)

        log_timescale_increment = math.log(float(max_radius) / float(min_radius)) / (
            frequency_num * 1.0 - 1
        )

        timescales = min_radius * np.exp(
            np.arange(frequency_num).astype(float) * log_timescale_increment
        )

        freq_list = 1.0 / timescales
    elif freq_init == "nerf":
        """
        compute according to NeRF position encoding, 
        Equation 4 in https://arxiv.org/pdf/2003.08934.pdf 
        2^{0}*pi, ..., 2^{L-1}*pi
        """
        #
        freq_list = np.pi * np.exp2(np.arange(frequency_num).astype(float))

    return freq_list