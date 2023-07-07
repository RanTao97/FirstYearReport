# EMVA1288_M2_postprocess is to process measurements of EMVA1288 dark current

import numpy as np
import matplotlib.pyplot as plt

import EMVA1288_utils
import EMVA1288_M2_utils

PATH = "20230529_LUCID_ATP013S_230101680_TEC10"  # path to save data, has the name of measurement date

if __name__ == '__main__':
    """Import EMVA M2 Measurement Variables"""
    meas_var = np.load(f"{PATH}/M2Data/measurement.npz")
    pixel_format_high_num = meas_var['pixel_format_high_num']
    gain_a = meas_var['gain_a']
    gain_d = meas_var['gain_d']
    black_level = meas_var['black_level']

    t_exp_dist = meas_var['t_exp_dist']
    meas_var.close()

    """Import EMVA M1 Results"""
    meas_results = np.load(f"{PATH}/M1Data/results.npz")
    K = meas_results['K']
    meas_results.close()

    """Read Dark Images at Each Exposure Time"""
    muy_dark = np.zeros_like(t_exp_dist)
    sigmay_dark2 = np.zeros_like(t_exp_dist)
    for idx, t_exp in enumerate(t_exp_dist):
        d0 = np.load(f"{PATH}/M2Data/Exp{idx}_d0.npy").astype(float)
        d1 = np.load(f"{PATH}/M2Data/Exp{idx}_d1.npy").astype(float)
        muy_dark[idx] = EMVA1288_utils.calc_muy(d0, d1)
        sigmay_dark2[idx] = EMVA1288_utils.calc_sigmay2(d0, d1)

    """Calculate muc_mean and Plot Dark Current from Mean"""
    muc_mean = EMVA1288_M2_utils.calc_muc_mean(muy_dark, t_exp_dist, K)

    """Calculate muc_var and Plot Dark Current from Variance"""
    muc_var = EMVA1288_M2_utils.calc_muc_var(sigmay_dark2, t_exp_dist, K)

    """Print Report"""
    print(f"EMVA1288 M2 Report:")
    print(f"Measurement Variables: \n"
          f"Bit Depth = {pixel_format_high_num}\tAnalog Gain = {gain_a}\tDigital Gain = {gain_d}\tBlack Level = {black_level} (DN)\n")
    print(f"Dark Current: \n"
          f"muc_mean = {muc_mean:.5f} (e-/s)\tmuc_var = {muc_var:.5f} (e-/s)")

    """Save Results"""
    np.savez(f"{PATH}/M2Data/results.npz",
             pixel_format_high_num=pixel_format_high_num,
             gain_a=gain_a, gain_d=gain_d, black_level=black_level,
             t_exp_dist=t_exp_dist,
             K=K,
             muc_mean=muc_mean, muc_var=muc_var)

    plt.show()

