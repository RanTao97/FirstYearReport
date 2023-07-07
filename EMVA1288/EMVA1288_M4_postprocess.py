# EMVA1288_M4_postprocess is to process measurements of EMVA1288 spectral sensitivity

import numpy as np
import matplotlib.pyplot as plt

import EMVA1288_utils
import EMVA1288_M4_utils

PATH = "20230518_LUCID_ATP013S_230101680"  # path to save data, has the name of measurement date
# path to manufactures' QE .txt
MNF_QE_PATH = "D:/OneDrive - University of Cambridge/SWIR Camera Lists/SONY SenSWIR QE.txt"

if __name__ == '__main__':
    """Read Manufacture's QE Data"""
    # put None if data is not available
    wave_MNF = np.loadtxt(MNF_QE_PATH, usecols=0)
    QE_MNF = np.loadtxt(MNF_QE_PATH, usecols=1)

    """Import EMVA M4 Measurement Variables"""
    meas_var = np.load(f"{PATH}/M4Data/measurement.npz")
    No_WAVELENGTH = meas_var['No_WAVELENGTH']
    WAVELENGTH_dist = meas_var['WAVELENGTH_dist']

    pixel_format_high = meas_var['pixel_format_high']
    pixel_format_high_num = meas_var['pixel_format_high_num']
    gain_a = meas_var['gain_a']
    gain_d = meas_var['gain_d']
    black_level = meas_var['black_level']

    t_exp_dist = meas_var['t_exp_dist']
    mup_dist = meas_var['mup_dist']
    meas_var.close()

    """Import EMVA M1 Results"""
    meas_results = np.load(f"{PATH}/M1Data/results.npz")
    K = meas_results['K']
    meas_results.close()

    """Read Bright and Dark Images at Each Wavelength"""
    muy = np.zeros_like(WAVELENGTH_dist)
    muy_dark = np.zeros_like(WAVELENGTH_dist)
    for idx in range(No_WAVELENGTH):
        b0 = np.load(f"{PATH}/M4Data/Wave{idx}_b0.npy")
        b1 = np.load(f"{PATH}/M4Data/Wave{idx}_b1.npy")
        d0 = np.load(f"{PATH}/M4Data/Wave{idx}_d0.npy")
        d1 = np.load(f"{PATH}/M4Data/Wave{idx}_d1.npy")
        muy[idx] = EMVA1288_utils.calc_muy(b0, b1)
        muy_dark[idx] = EMVA1288_utils.calc_muy(d0, d1)

    print(muy - muy_dark)
    print(mup_dist)

    """Calculate and Plot Spectral Sensitivity"""
    QE_dist = EMVA1288_M4_utils.calc_QE(muy, muy_dark, K, mup_dist)
    EMVA1288_M4_utils.plot_QE(WAVELENGTH_dist, QE_dist, wave_MNF, QE_MNF)

    """Print Report"""
    print(f"EMVA1288 M4 Report:")
    print(f"Measurement Variables: \n"
          f"Bit Depth = {pixel_format_high_num}\tAnalog Gain = {gain_a}\tDigital Gain = {gain_d}\tBlack Level = {black_level} (DN)\n")
    print(f"Spectral Sensitivity: \n")
    for idx in range(No_WAVELENGTH):
        print(f"{idx}: at Wavelength {WAVELENGTH_dist[idx]} (nm), QE is {QE_dist[idx] * 100:.5f} (%)")

    """Save Results"""
    np.savez(f"{PATH}/M4Data/results.npz",
             pixel_format_high=pixel_format_high, pixel_format_high_num=pixel_format_high_num,
             gain_a=gain_a, gain_d=gain_d, black_level=black_level,
             WAVELENGTH_dist=WAVELENGTH_dist, QE_dist=QE_dist)

    plt.show()
