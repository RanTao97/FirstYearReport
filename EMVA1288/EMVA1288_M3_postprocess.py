# EMVA1288_M3_postprocess is to process measurements of EMVA1288 spatial nonuniformity and defect pixels

import numpy as np
import matplotlib.pyplot as plt
import pickle

import EMVA1288_utils
import EMVA1288_M3_utils
import EMVA1288_M3_dark_utils

PATH = "20230602_LUCID_ATP013S_230101680_LED1200nm"  # path to save data, has the name of measurement date

# set figure size and font size
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.titlesize'] = 'x-large'
plt.rcParams['axes.labelsize'] = 'large'
plt.rcParams['xtick.labelsize'] = 'large'
plt.rcParams['ytick.labelsize'] = 'large'
plt.rcParams['legend.fontsize'] = 'large'

if __name__ == '__main__':
    """Import EMVA M3 Measurement Variables"""
    meas_var = np.load(f"{PATH}/M3Data/measurement.npz")

    M = meas_var['M']
    N = meas_var['N']

    pixel_format_high = meas_var['pixel_format_high']
    pixel_format_high_num = meas_var['pixel_format_high_num']
    gain_a = meas_var['gain_a']
    gain_d = meas_var['gain_d']
    black_level = meas_var['black_level']

    t_exp_50 = meas_var['t_exp_50']
    IMG_SEQ = meas_var['IMG_SEQ']
    meas_var.close()

    """Import EMVA M1 Results"""
    meas_results = np.load(f"{PATH}/M1Data/results.npz")
    mup_dist = meas_results['mup_dist']
    K = meas_results['K']
    QE = meas_results['QE']
    sigmad = meas_results['sigmad']
    meas_results.close()
    print(f"From M1 Results: K={K:.5f}, QE={QE:.5f}, sigmad={sigmad:.5f}")

    """Import EMVA M1 SNR Plot Object"""
    with open(f"{PATH}/M1Data/SNR_plot_ax.pkl", 'rb') as fid:
        SNR_plot_ax = pickle.load(fid)

    """Read 50% Saturation and Dark Images"""
    y50_stack = np.zeros((M, N, IMG_SEQ))
    ydark_stack = np.zeros((M, N, IMG_SEQ))
    for i in range(IMG_SEQ):
        y50_stack[:, :, i] = np.load(f"{PATH}/M3Data/Exp50_{i}.npy").astype(float)
        ydark_stack[:, :, i] = np.load(f"{PATH}/M3Data/Dark_{i}.npy").astype(float)

    """Calculate HPF Image Stack y50_HPF_stack, ydark_HPF_stack"""
    y50_HPF_stack = EMVA1288_M3_utils.calc_HPF(y50_stack)
    ydark_HPF_stack = EMVA1288_M3_utils.calc_HPF(ydark_stack)

    """Calculate sy_502, sy_dark2"""
    sy_502 = EMVA1288_M3_utils.calc_sy_stack2(y50_HPF_stack)
    sy_dark2 = EMVA1288_M3_utils.calc_sy_stack2(ydark_HPF_stack)

    # Todo
    # sy_dark2_dark = EMVA1288_M3_dark_utils.calc_sy_stack2(ydark_stack)

    """Calculate scol_502, srow_502, spix_502, scol_dark2, srow_dark2, spix_dark2"""
    scol_502, srow_502, spix_502 = EMVA1288_M3_utils.calc_sy_components(y50_HPF_stack)
    scol_dark2, srow_dark2, spix_dark2 = EMVA1288_M3_utils.calc_sy_components(ydark_HPF_stack)

    # Todo
    # scol_dark2_dark, srow_dark2_dark, spix_dark2_dark = EMVA1288_M3_dark_utils.calc_sy_components(ydark_stack)

    print(EMVA1288_M3_utils.calc_sigmay_stack2(y50_HPF_stack), EMVA1288_M3_utils.calc_sigmay_stack2(ydark_HPF_stack))
    print(sy_502, scol_502, srow_502, spix_502)
    print(sy_dark2, scol_dark2, srow_dark2, spix_dark2)
    # print(sy_dark2_dark, scol_dark2_dark, srow_dark2_dark, spix_dark2_dark)

    plt.figure("HPF Mean 50%")
    HPF_Mean_50 = EMVA1288_M3_utils.calc_y_mean_img(y50_HPF_stack)
    print(np.mean(HPF_Mean_50))
    plt.imshow(HPF_Mean_50, vmin=-500, vmax=500)
    print(np.var(HPF_Mean_50))
    plt.colorbar()

    plt.figure("HPF Dark 50%")
    HPF_Mean_Dark = EMVA1288_M3_utils.calc_y_mean_img(ydark_HPF_stack)
    print(np.mean(HPF_Mean_Dark))
    plt.imshow(HPF_Mean_Dark, vmin=-500, vmax=500)
    print(np.var(HPF_Mean_Dark))
    plt.colorbar()

    """Calculate DSNU, PRNU"""
    DSNU = EMVA1288_M3_utils.calc_DSNU(np.sqrt(sy_dark2), K)
    DSNU_col = EMVA1288_M3_utils.calc_DSNU(np.sqrt(scol_dark2), K)
    DSNU_row = EMVA1288_M3_utils.calc_DSNU(np.sqrt(srow_dark2), K)
    DSNU_pix = EMVA1288_M3_utils.calc_DSNU(np.sqrt(spix_dark2), K)
    PRNU = EMVA1288_M3_utils.calc_PRNU(sy_502, sy_dark2, np.mean(y50_stack), np.mean(ydark_stack))
    PRNU_col = EMVA1288_M3_utils.calc_PRNU(scol_502, scol_dark2, np.mean(y50_stack), np.mean(ydark_stack))
    PRNU_row = EMVA1288_M3_utils.calc_PRNU(srow_502, srow_dark2, np.mean(y50_stack), np.mean(ydark_stack))
    PRNU_pix = EMVA1288_M3_utils.calc_PRNU(spix_502, spix_dark2, np.mean(y50_stack), np.mean(ydark_stack))

    # Todo
    # DSNU_dark = EMVA1288_M3_dark_utils.calc_DSNU(np.sqrt(sy_dark2_dark), K)
    # DSNU_col_dark = EMVA1288_M3_dark_utils.calc_DSNU(np.sqrt(scol_dark2_dark), K)
    # DSNU_row_dark = EMVA1288_M3_dark_utils.calc_DSNU(np.sqrt(srow_dark2_dark), K)
    # DSNU_pix_dark = EMVA1288_M3_dark_utils.calc_DSNU(np.sqrt(spix_dark2_dark), K)

    # """Calculate and Plot Total SNR"""
    EMVA1288_M3_utils.plot_SNR_tot(QE, mup_dist, sigmad**2, K, DSNU, PRNU, SNR_plot_ax)

    # """Calculate and Plot DSNU, PRNU Spectrograms"""
    # EMVA1288_M3_utils.calc_spec_DSNU(ydark_stack, ydark_HPF_stack)
    # EMVA1288_M3_utils.calc_spec_PRNU(y50_stack, ydark_stack, y50_HPF_stack, ydark_HPF_stack, PRNU)

    # # Todo
    # EMVA1288_M3_dark_utils.calc_spec_DSNU(ydark_stack)
    #
    """Calculate and Plot Horizontal, Vertical Profiles"""
    EMVA1288_M3_utils.calc_profile_DSNU(ydark_stack)
    EMVA1288_M3_utils.calc_profile_PRNU(y50_stack, ydark_stack)

    """Characterise and Plot Defect Pixels"""
    EMVA1288_M3_utils.calc_defect_pix_DSNU(ydark_stack)
    EMVA1288_M3_utils.calc_defect_pix_PRNU(y50_stack, ydark_stack)

    # Todo
    # EMVA1288_M3_dark_utils.calc_defect_pix_DSNU(ydark_stack)

    """Print Report"""
    print(f"EMVA1288 M3 Report:")
    print(f"Measurement Variables: \n"
          f"Bit Depth = {pixel_format_high_num}\tAnalog Gain = {gain_a}\tDigital Gain = {gain_d}\tBlack Level = {black_level} (DN)\n"
          f"at 50% Saturation: Exposure Time = {t_exp_50} us")
    print(f"Spatial Nonuniformities: \n"
          f"DSNU = {DSNU:.5f} (e-)\tDSNU_col = {DSNU_col:.5f} (e-)\tDSNU_row = {DSNU_row:.5f} (e-)\tDSNU_pix = {DSNU_pix:.5f} (e-)\n"
          f"PRNU = {PRNU * 100:.5f} (%)\tPRNU_col = {PRNU_col * 100:.5f} (%)\tPRNU_row = {PRNU_row * 100:.5f} (%)\tPRNU_pix = {PRNU_pix * 100:.5f} (%)")

    # # Todo
    # print(f"\n\nDark: Spatial Nonuniformities: \n"
    #       f"DSNU_dark = {DSNU_dark} (e-)\tDSNU_col = {DSNU_col_dark} (e-)\tDSNU_row = {DSNU_row_dark} (e-)\tDSNU_pix = {DSNU_pix_dark} (e-)\n")

    """Save Results"""
    np.savez(f"{PATH}/M3Data/results.npz",
             pixel_format_high=pixel_format_high, pixel_format_high_num=pixel_format_high_num,
             gain_a=gain_a, gain_d=gain_d, black_level=black_level, t_exp_50=t_exp_50,
             DSNU=DSNU, DSNU_col=DSNU_col, DSNU_row=DSNU_row, DSNU_pix=DSNU_pix,
             PRNU=PRNU, PRNU_col=PRNU_col, PRNU_row=PRNU_row, PRNU_pix=PRNU_pix)

    plt.show()


