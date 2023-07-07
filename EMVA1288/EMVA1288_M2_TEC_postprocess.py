# EMVA1288_M2_TEC_postprocess is to process measurements of EMVA1288 dark current, TEC dependent

import numpy as np
import matplotlib.pyplot as plt

import EMVA1288_utils

# update linear regression to estimate the uncertainty in slopes
from scipy import stats
from scipy.stats import t
tinv = lambda p, df: abs(t.ppf(p/2, df))
std_K = 0.002797

# set figure size and font size
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.titlesize'] = 'x-large'
plt.rcParams['axes.labelsize'] = 'large'
plt.rcParams['xtick.labelsize'] = 'large'
plt.rcParams['ytick.labelsize'] = 'large'
plt.rcParams['legend.fontsize'] = 'large'

PATH_TEC15 = "20230529_LUCID_ATP013S_230101680_TEC15"  # path to save data, has the name of measurement date
PATH_TEC10 = "20230529_LUCID_ATP013S_230101680_TEC10"
PATH_TEC20 = "20230529_LUCID_ATP013S_230101680_TEC20"

def calc_muc_mean(muy_dark, t_exp, K, TEC, ax_mean):
    """
    helper function to calculate dark current from mean
    :param muy_dark: an array of 6 muy_dark
    :param t_exp: an array of 6 t_exp [us]
    :param K: overall system gain
    :param TEC: TEC temperature
    :param ax_mean: plot handler for "Dark Current from Mean"
    :return: muc_mean [e-/s], ax_mean
    """
    # linear regression
    p = np.polyfit(x=t_exp, y=muy_dark, deg=1)
    y_fit = np.polyval(p, t_exp)
    muc_mean = p[0] / K * 1e6

    """20230702: Update Linear Regression to Estimate Uncertainty in Slope"""
    x = t_exp
    y = muy_dark
    res = stats.linregress(x, y)
    ts = tinv(0.05, len(x) - 2)
    b = res.slope
    std_b = ts * res.stderr
    print(f"slope (95%): {res.slope:.6f} +/- {ts * res.stderr:.6f}")
    print(f"Alternative muc_mean = {res.slope / K * 1e6}")
    print(f"Std: {np.sqrt((10**12) * (std_b**2) / (K**2) + (10**12) * (b**2) * (std_K**2) / (K**4))}")

    ax_mean.scatter(t_exp / 1e3, muy_dark, label=f"Data TEC = {TEC} $^\circ$C")
    ax_mean.plot(t_exp / 1e3, y_fit, linestyle=':', label=f"Fit TEC = {TEC} $^\circ$C")
    ax_mean.legend()
    ax_mean.set_xlabel(r'$t_{exp} \; \mathrm{(ms)}$')
    ax_mean.set_ylabel(r'$\mu_{y.dark} \; \mathrm{(DN)}$')
    ax_mean.set_title("Dark Current from Mean")

    return muc_mean, ax_mean


def calc_muc_var(sigmay_dark2, t_exp, K, TEC, ax_var):
    """
    helper function to calculate dark current from variance
    :param sigmay_dark2: an array of 6 sigmay_dark2
    :param t_exp: an array of 6 t_exp [us]
    :param K: overall system gain
    :param TEC: TEC temperature
    :param ax_var: plot handler for "Dark Current from Variance"
    :return: muc_var [e-/s], ax_var
    """
    # linear regression
    p = np.polyfit(x=t_exp, y=sigmay_dark2, deg=1)
    y_fit = np.polyval(p, t_exp)
    muc_var = p[0] / (K**2) * 1e6

    """20230702: Update Linear Regression to Estimate Uncertainty in Slope"""
    x = t_exp
    y = sigmay_dark2
    res = stats.linregress(x, y)
    ts = tinv(0.05, len(x) - 2)
    b = res.slope
    std_b = ts * res.stderr
    print(f"slope (95%): {res.slope:.6f} +/- {ts * res.stderr:.6f}")
    print(f"Alternative muc_var = {res.slope / (K**2) * 1e6}")
    print(f"Std: {np.sqrt((10 ** 12) * (std_b ** 2) / (K ** 4) + 4 * (10 ** 12) * (b ** 2) * (std_K ** 2) / (K ** 6))}")

    ax_var.scatter(t_exp / 1e3, sigmay_dark2, label=f"Data TEC = {TEC} $^\circ$C")
    ax_var.plot(t_exp / 1e3, y_fit, linestyle=':', label=f"Fit TEC = {TEC} $^\circ$C")
    ax_var.legend()
    ax_var.set_xlabel(r'$t_{exp} \; \mathrm{(ms)}$')
    ax_var.set_ylabel(r'$\sigma_{y.dark}^2 \; \mathrm{(DN^2)}$')
    ax_var.set_title("Dark Current from Variance")

    return muc_var, ax_var

if __name__ == '__main__':
    """Import EMVA M2 Measurement Variables"""
    PATH_list = [PATH_TEC10, PATH_TEC15, PATH_TEC20]
    TEC_list = [10, 15, 20]

    pixel_format_high_num = np.zeros(len(PATH_list))
    gain_a = np.zeros(len(PATH_list))
    gain_d = np.zeros(len(PATH_list))
    black_level = np.zeros(len(PATH_list))
    t_exp_dist = np.zeros(((len(PATH_list)), 11))
    T_sensor_dist = np.zeros(((len(PATH_list)), 11))
    T_TEC_dist = np.zeros(((len(PATH_list)), 11))
    for path_idx, path in enumerate(PATH_list):
        meas_var = np.load(f"{path}/M2Data/measurement.npz")
        pixel_format_high_num[path_idx] = meas_var['pixel_format_high_num']
        gain_a[path_idx] = meas_var['gain_a']
        gain_d[path_idx] = meas_var['gain_d']
        black_level[path_idx] = meas_var['black_level']
        t_exp_dist[path_idx, :] = meas_var['t_exp_dist']
        T_sensor_dist[path_idx, :] = meas_var['T_sensor_dist']
        T_TEC_dist[path_idx, :] = meas_var['T_TEC_dist']
        meas_var.close()

    """Import EMVA M1 Results"""
    meas_results = np.load(f"{PATH_list[0]}/M1Data/results.npz")
    K = meas_results['K']
    meas_results.close()

    """Read Dark Images at Each Exposure Time, and Calculate Dark Current"""
    muc_mean = np.zeros(len(PATH_list))
    muc_var = np.zeros(len(PATH_list))
    fig_mean, ax_mean = plt.subplots(num="Dark Current from Mean")  # plot for muc_mean
    fig_var, ax_var = plt.subplots(num="Dark Current from Variance")  # plot for muc_var
    for path_idx, path in enumerate(PATH_list):
        t_exp_dist_idx = t_exp_dist[path_idx, :]
        muy_dark = np.zeros_like(t_exp_dist_idx)
        sigmay_dark2 = np.zeros_like(t_exp_dist_idx)
        for idx, t_exp in enumerate(t_exp_dist_idx):
            d0 = np.load(f"{path}/M2Data/Exp{idx}_d0.npy").astype(float)
            d1 = np.load(f"{path}/M2Data/Exp{idx}_d1.npy").astype(float)
            muy_dark[idx] = EMVA1288_utils.calc_muy(d0, d1)
            sigmay_dark2[idx] = EMVA1288_utils.calc_sigmay2(d0, d1)

        """Calculate muc_mean and Plot Dark Current from Mean"""
        muc_mean[path_idx], ax_mean = calc_muc_mean(muy_dark, t_exp_dist_idx, K, TEC_list[path_idx], ax_mean)

        """Calculate muc_var and Plot Dark Current from Variance"""
        muc_var[path_idx], ax_var = calc_muc_var(sigmay_dark2, t_exp_dist_idx, K, TEC_list[path_idx], ax_var)

    """Plot Temperature Variation"""
    fig_sensor, ax_sensor = plt.subplots(num="Sensor Temperature")  # plot for Sensor Temperature
    fig_TEC, ax_TEC = plt.subplots(num="TEC Temperature")  # plot for TEC Temperature
    for path_idx, path in enumerate(PATH_list):
        ax_sensor.plot(T_sensor_dist[path_idx, :], label=f"TEC Set Point = {TEC_list[path_idx]} $^\circ$C")
        ax_TEC.plot(T_TEC_dist[path_idx, :], label=f"TEC Set Point = {TEC_list[path_idx]} $^\circ$C")
    ax_sensor.set_title("Sensor Temperature")
    ax_sensor.set_xlabel("Time Stamp")
    ax_sensor.set_ylabel("Temperature ($^\circ$C)")
    ax_sensor.set_ylim(35, 45)
    ax_sensor.legend()
    ax_TEC.set_title("TEC Temperature")
    ax_TEC.set_xlabel("Time Stamp")
    ax_TEC.set_ylabel("Temperature ($^\circ$C)")
    ax_TEC.set_ylim(5, 25)
    ax_TEC.legend()

    """Print Report"""
    print(f"EMVA1288 M2 Report:")
    for path_idx, path in enumerate(PATH_list):
        print(f"Measurement Variables for path {path}: \n"
              f"Bit Depth = {pixel_format_high_num[path_idx]}\t"
              f"Analog Gain = {gain_a[path_idx]}\tDigital Gain = {gain_d[path_idx]}\t"
              f"Black Level = {black_level[path_idx]} (DN)\n")
        print(f"Dark Current: \n"
              f"muc_mean = {muc_mean[path_idx]:.5f} (e-/s)\tmuc_var = {muc_var[path_idx]:.5f} (e-/s)\n\n")

    plt.show()

