# EMVA1288_M2_utils contains all functions required to compute EMVA1288 dark current

import numpy as np
import matplotlib.pyplot as plt

import constants

def find_t_exp_vals(t_exp_min, t_exp_max):
    """
    helper function to find t_exp distribution 7.1
    find 6 equally spaced t_exp
    N.B. SUBJECTED TO ROUNDING BY REAL CAMERA SETTING, ALWAYS CHECK THE t_exp BY QUERYING THE CAMERA
    :param t_exp_min: min t_exp available from camera
    :param t_exp_max: max t_exp available from camera
    :return: an array of 6 equally spaced t_exp
    """
    # Todo: 2023/05/29 set 6->11 points
    return np.linspace(start=t_exp_min, stop=t_exp_max, num=11, endpoint=True)


def calc_muc_mean(muy_dark, t_exp, K):
    """
    helper function to calculate dark current from mean
    :param muy_dark: an array of 6 muy_dark
    :param t_exp: an array of 6 t_exp [us]
    :param K: overall system gain
    :return: muc_mean [e-/s]
    """
    # linear regression
    p = np.polyfit(x=t_exp, y=muy_dark, deg=1)
    y_fit = np.polyval(p, t_exp)
    muc_mean = p[0] / K * 1e6

    plt.figure("Dark Current from Mean")
    plt.scatter(t_exp / 1e3, muy_dark, label="Data")
    plt.plot(t_exp / 1e3, y_fit, linestyle=':', label="Fit")
    plt.legend()
    plt.xlabel(r'$t_{exp} \; \mathrm{(ms)}$')
    plt.ylabel(r'$\mu_{y.dark} \; \mathrm{(DN)}$')
    plt.title("Dark Current from Mean")

    return muc_mean


def calc_muc_var(sigmay_dark2, t_exp, K):
    """
    helper function to calculate dark current from variance
    :param sigmay_dark2: an array of 6 sigmay_dark2
    :param t_exp: an array of 6 t_exp [us]
    :param K: overall system gain
    :return: muc_var [e-/s]
    """
    # linear regression
    p = np.polyfit(x=t_exp, y=sigmay_dark2, deg=1)
    y_fit = np.polyval(p, t_exp)
    muc_var = p[0] / (K**2) * 1e6

    plt.figure("Dark Current from Variance")
    plt.scatter(t_exp / 1e3, sigmay_dark2, label="Data")
    plt.plot(t_exp / 1e3, y_fit, linestyle=':', label="Fit")
    plt.legend()
    plt.xlabel(r'$t_{exp} \; \mathrm{(ms)}$')
    plt.ylabel(r'$\sigma_{y.dark}^2 \; \mathrm{(DN^2)}$')
    plt.title("Dark Current from Variance")

    return muc_var

