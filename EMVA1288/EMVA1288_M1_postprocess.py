# EMVA1288_M1_postprocess is to process measurements of EMVA1288 (1) sensitivity, (2) temporal noise, (3) linearity

import numpy as np
import matplotlib.pyplot as plt
import pickle

import EMVA1288_utils
import EMVA1288_M1_utils

PATH = "20230521_LUCID_ATP013S_230101680_LED490nm"  # path to save data, has the name of measurement date

if __name__ == '__main__':
    """Import EMVA M1 Measurement Variables"""
    meas_var = np.load(f"{PATH}/M1Data/measurement.npz")
    E = meas_var['E']
    A = meas_var['A']
    WAVELENGTH = meas_var['WAVELENGTH']

    M = meas_var['M']
    N = meas_var['N']

    pixel_format_high = meas_var['pixel_format_high']
    pixel_format_high_num = meas_var['pixel_format_high_num']
    gain_a = meas_var['gain_a']
    gain_d = meas_var['gain_d']
    black_level = meas_var['black_level']

    t_exp_sat = meas_var['t_exp_sat']
    mup_sat = meas_var['mup_sat']

    t_exp_dist = meas_var['t_exp_dist']
    H_dist = meas_var['H_dist']
    mup_dist = meas_var['mup_dist']

    print(f'Wavelength = {WAVELENGTH}, E = {E}, black_level = {black_level}, t_exp_sat = {t_exp_sat}')

    meas_var.close()

    """Read Bright and Dark Images at Each Exposure Time"""
    muy = np.zeros_like(t_exp_dist)
    sigmay2 = np.zeros_like(t_exp_dist)
    muy_dark = np.zeros_like(t_exp_dist)
    sigmay_dark2 = np.zeros_like(t_exp_dist)
    for idx, t_exp in enumerate(t_exp_dist):
        b0 = np.load(f"{PATH}/M1Data/Exp{idx}_b0.npy").astype(float)
        b1 = np.load(f"{PATH}/M1Data/Exp{idx}_b1.npy").astype(float)
        d0 = np.load(f"{PATH}/M1Data/Exp{idx}_d0.npy").astype(float)
        d1 = np.load(f"{PATH}/M1Data/Exp{idx}_d1.npy").astype(float)
        muy[idx] = EMVA1288_utils.calc_muy(b0, b1)
        sigmay2[idx] = EMVA1288_utils.calc_sigmay2(b0, b1)
        muy_dark[idx] = EMVA1288_utils.calc_muy(d0, d1)
        sigmay_dark2[idx] = EMVA1288_utils.calc_sigmay2(d0, d1)

    """Calculate R and Plot Sensitivity"""
    R = EMVA1288_M1_utils.calc_R(mup_dist, muy, muy_dark)

    """Calculate K and Plot Photon Transfer"""
    K = EMVA1288_M1_utils.calc_K(muy, muy_dark, sigmay2, sigmay_dark2)

    """Calculate QE"""
    QE = EMVA1288_M1_utils.calc_QE(R, K)

    """Calculate sigmay_dark and sigma_d"""
    sigmay_dark = np.sqrt(sigmay_dark2[0])
    sigmad = EMVA1288_M1_utils.calc_sigmad(sigmay_dark**2, K)

    """Calculate mup_min and mue_min"""
    mup_min = EMVA1288_M1_utils.calc_mup_min(QE, sigmad**2, K)
    mue_min = EMVA1288_M1_utils.calc_mue_min(QE, mup_min)

    """Calculate mue_sat"""
    mue_sat = EMVA1288_M1_utils.calc_mue_sat(QE, mup_sat)

    """Calculate SNR_max"""
    SNR_max = EMVA1288_M1_utils.calc_SNR_max(mue_sat)

    """Plot SNR"""
    SNR_plot_ax = EMVA1288_M1_utils.plot_SNR(mup_dist, muy, muy_dark, np.sqrt(sigmay2), QE, sigmad**2, K)

    """Calculate DR"""
    DR = EMVA1288_M1_utils.calc_DR(mup_sat, mup_min)

    """Calculate LE and Plot Linearity"""
    LE = EMVA1288_M1_utils.calc_LE(muy, muy_dark, H_dist)

    """Print Report"""
    print(f"EMVA1288 M1 Report:")
    print(f"Measurement Variables: \n"
          f"E = {E} (uW/cm^2)\tA = {A} (um^2)\tWavelength = {WAVELENGTH} (nm) \n"
          f"Bit Depth = {pixel_format_high_num}\tAnalog Gain = {gain_a}\tDigital Gain = {gain_d}\tBlack Level = {black_level} (DN)\n")
    print(f"Quantum Efficiency: \n"
          f"QE = {QE * 100:.5f} (%)")
    print(f"Overall System Gain: \n"
          f"K = {K:.5f} (DN/e-)\t1/K = {1/K:.5f} (e-/DN)")
    print(f"Temporal Dark Noise: \n"
          f"sigma_d = {sigmad:.5f} (e-)\tsigmay_dark = {sigmay_dark:.5f} (DN)")
    print(f"SNR: \n"
          f"SNR_max = {SNR_max:.5f}\t{20 * np.log10(SNR_max):.5f} (dB)\t1/SNR_max = {1 / SNR_max * 100:.5f} (%)")
    print(f"Absolute Sensitivity Threshold: \n"
          f"mue_min = {mue_min:.5f} (e-)\tmue_min_area = {mue_min / A:.5f} (e-/um^2)")
    print(f"Saturation Capacity: \n"
          f"mue_sat = {mue_sat:.5f} (e-)\tmue_sat_area = {mue_sat / A:.5f} (e-/um^2)")
    print(f"Dynamic Range: \n"
          f"DR = {DR:.5f} (dB)")
    print(f"Linearity Error: \n"
          f"LE = {LE:.5f} (%)")

    """Save Results"""
    np.savez(f"{PATH}/M1Data/results.npz",
             E=E, A=A, WAVELENGTH=WAVELENGTH,
             pixel_format_high=pixel_format_high, pixel_format_high_num=pixel_format_high_num,
             gain_a=gain_a, gain_d=gain_d, black_level=black_level,
             t_exp_sat=t_exp_sat, mup_sat=mup_sat,
             t_exp_dist=t_exp_dist, H_dist=H_dist, mup_dist=mup_dist,
             R=R, K=K, QE=QE,
             sigmay_dark=sigmay_dark, sigmad=sigmad,
             mup_min=mup_min, mue_min=mue_min, mue_sat=mue_sat,
             SNR_max=SNR_max, DR=DR, LE=LE,
             muy=muy)

    """Save SNR Plot Object for Later Use in EMVA M3"""
    with open(f"{PATH}/M1Data/SNR_plot_ax.pkl", 'wb') as fid:
        pickle.dump(SNR_plot_ax, fid)

    plt.show()
