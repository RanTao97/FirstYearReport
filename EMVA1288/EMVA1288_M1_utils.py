# EMVA1288_M1_utils contains all functions required to compute EMVA1288 (1) sensitivity, (2) temporal noise, (3) linearity

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep

import constants

def check_con_dark(img, y):
    """
    helper function to check measurement condition 6.5 Offest:
    if <= 0.5% of pixels underflow, i.e., have value zero or one
    :param img: the dark image
    :param y: the camera min value y
    :return: 0 if <= 0.5%, 1 if > 0.5%
    """
    # Todo: N.B. the threshold is 1 rather than 0 in EMVA standard
    percentage = (np.count_nonzero(img <= y) / img.size)
    if percentage <= 0.5/100:
        return 0
    else:
        return 1


def check_con_sat(img0, img1, y):
    """
    helper function to check measurement condition 6.6 Saturation:
    if 0.1-0.2% of the total number of pixels show the camera maximum value y,
    take 2 images without any averaging
    :param img0: the saturation image 0
    :param img1: the saturation image 1
    :param y: the camera maximum value y
    :return: -1 if < 0.1% 0 if between 0.1-0.2%, 1 if > 0.2%
    """
    percentage = (np.count_nonzero(img0 >= y) + np.count_nonzero(img1 >= y)) / (img0.size + img1.size)
    if percentage < 0.1/100:
        return -1
    elif percentage > 0.2/100:
        return 1
    else:
        return 0


def find_t_exp_vals(t_exp_min, t_exp_sat):
    """
    helper function to find t_exp distribution (measurement condition 6.5 Distribution of Radiant Exposure Values),
    vary t_exp (exposure time) to adjust radiant exposure
    find 50 equally spaced t_exp
    N.B. SUBJECTED TO ROUNDING BY REAL CAMERA SETTING, ALWAYS CHECK THE t_exp BY QUERYING THE CAMERA
    :param t_exp_min: min t_exp available from camera
    :param t_exp_sat: defined by measurement condition 6.6 Saturation
    :return: an array of 50 equally spaced t_exp
    """
    return np.linspace(start=t_exp_min, stop=t_exp_sat, num=50, endpoint=True)


def calc_R(mup, muy, muy_dark):
    """
    helper function to calculate R, Eqn. (49)
    :param mup: an array of 50 mup
    :param muy: an array of 50 muy
    :param muy_dark: an array of 50 muy_dark
    :return: R
    """
    y = muy - muy_dark

    # fit: 0-70% saturation range
    lower_limit_idx = 0
    upper_limit_idx = np.argmax(y > 0.7 * y[-1])

    # linear regression
    p = np.polyfit(x=mup[lower_limit_idx:upper_limit_idx], y=y[lower_limit_idx:upper_limit_idx], deg=1)
    y_fit = np.polyval(p, mup)
    R = p[0]

    # plot sensitivity
    plt.figure("Sensitivity")
    plt.plot(mup, y, label="Data")
    plt.plot(mup, y_fit, linestyle=':', label="Fit")
    plt.scatter(mup[lower_limit_idx], y[lower_limit_idx], color='k', label="Fit Range")
    plt.scatter(mup[upper_limit_idx-1], y[upper_limit_idx-1], color='k')
    plt.legend()
    plt.xlabel(r'$\mu_p \; \mathrm{(<\sharp photons>/pixel)}$')
    plt.ylabel(r'$\mu_y - \mu_{y.dark} \; \mathrm{(DN)}$')
    plt.title("Sensitivity")

    return R


def calc_K(muy, muy_dark, sigmay2, sigmay_dark2):
    """
    helper function to calculate K, Eqn. (50), and plot Photon Transfer
    :param muy: an array of 50 muy
    :param muy_dark: an array of 50 muy_dark
    :param sigmay2: an array of 50 sigmay2
    :param sigmay_dark2: an array of 50 sigmay_dark2
    :return: K
    """
    x = muy - muy_dark
    y = sigmay2 - sigmay_dark2

    # fit: 0-70% saturation range
    lower_limit_idx = 0
    upper_limit_idx = np.argmax(x > 0.7 * x[-1])

    # linear regression
    p = np.polyfit(x=x[lower_limit_idx:upper_limit_idx], y=y[lower_limit_idx:upper_limit_idx], deg=1)
    y_fit = np.polyval(p, x)
    K = p[0]

    # check linearity
    # cubic spline
    tck = splrep(x=x[lower_limit_idx:upper_limit_idx], y=y[lower_limit_idx:upper_limit_idx], k=3)
    # check local K
    K_local = splev(x=x[lower_limit_idx:upper_limit_idx], tck=tck, der=1)
    # check linearity condition , Eqn. (51) modified
    if np.mean(np.abs(K_local - K)/K) >= 0.03:
        print("The photon transfer curve is not sufficiently linear, "
              "the evaluation of the measurements must be performed according to the general model")
        print(f"The deviation of local K is {np.mean(np.abs(K_local - K)/K)} > the required 0.03")

    # plot photon transfer curve
    plt.figure("Photon Transfer")
    plt.plot(x, y, label="Data")
    plt.plot(x, y_fit, linestyle=':', label="Fit")
    plt.scatter(x[lower_limit_idx], y[lower_limit_idx], color='k', label="Fit Range")
    plt.scatter(x[upper_limit_idx - 1], y[upper_limit_idx - 1], color='k')
    plt.legend()
    plt.xlabel(r'$\mu_y - \mu_{y.dark} \; \mathrm{(DN)}$')
    plt.ylabel(r'$\sigma_y^2 - \sigma_{y.dark}^2 \; \mathrm{(DN^2)}$')
    plt.title("Photon Transfer")

    return K


def calc_QE(R, K):
    """
    helper function to calculate QE (Quantum Efficiency), Eqn. (52)
    :param R: responsivity R
    :param K: overall system gain K
    :return: QE
    """
    return R/K


def calc_sigmad(sigmay_dark2, K):
    """
    helper function to calculate the temporal dark noise sigmad, Eqn. (53) & (54)
    :param sigmay_dark2: sigmay_dark2 measured at the lowest exposure time
    :param K: overall system gain K
    :return: sigmad
    """
    if sigmay_dark2 >= 0.24:
        return np.sqrt(sigmay_dark2 - constants.sigmaq2) / K
    else:
        return 0.40 / K


def calc_mup_min(QE, sigmad2, K):
    """
    helper function to calculate the absolute sensitivity threshold mup_min when SNR=1, Eqn. (26)
    :param QE: quantum efficiency QE
    :param sigmad2: temporal dark noise sigmad2
    :param K: overall system gain K
    :return: mup_min
    """
    return 1 / QE * (np.sqrt(sigmad2 + constants.sigmaq2/(K**2) + 0.25) + 0.5)


def calc_mue_min(QE, mup_min):
    """
    helper function to calculate the absolute sensitivity threshold mue_min when SNR=1, Eqn. (27)
    :param QE: quantum efficiency QE
    :param mup_min: absolute sensitivity threshold mup_min
    :return: mue_min
    """
    return QE * mup_min


def calc_mue_sat(QE, mup_sat):
    """
    helper function to calculate the saturation capacity mue_sat, Eqn. (24)
    :param QE: quantum efficiency QE
    :param mup_sat: saturation capacity mup_sat
    :return: mue_sat
    """
    return QE * mup_sat


def calc_SNR_data(muy, muy_dark, sigmay):
    """
    helper function to calculate measured SNR, Eqn. (20)
    :param muy: an array of 50 muy
    :param muy_dark: an array of 50 muy_dark
    :param sigmay: an array of 50 sigmay
    :return: SNR_data
    """
    return (muy - muy_dark) / sigmay


def calc_SNR_fit(QE, mup, sigmad2, K):
    """
    helper function to calculate modelled SNR, Eqn. (21)
    :param QE: quantum efficiency QE
    :param mup: an array of 50 mup
    :param sigmad2: temporal dark noise sigmad2
    :param K: overall gain K
    :return: SNR_fit
    """
    return QE * mup / np.sqrt(sigmad2 + constants.sigmaq2/(K**2) + QE * mup)


def calc_SNR_lim(mup):
    """
    helper function to calculate ideal SNR, Eqn. (23)
    :param mup: an array of 50 mup
    :return: SNR_lim
    """
    return np.sqrt(mup)


def calc_SNR_max(mue_sat):
    """
    helper function to calculate the maximum achievable SNR_max, Eqn. (55)
    :param mue_sat: absolute sensitivity threshold mue_sat
    :return: SNR_max
    """
    return np.sqrt(mue_sat)


def plot_SNR(mup, muy, muy_dark, sigmay, QE, sigmad2, K):
    """
    helper function to plot SNR
    :param mup: an array of 50 mup
    :param muy: an array of 50 muy
    :param muy_dark: an array of 50 muy_dark
    :param sigmay: an array of 50 sigmay
    :param QE: quantum efficiency QE
    :param sigmad2: temporal dark noise sigmad2
    :param K: overall gain K
    :return: figure handle ax
    """
    SNR_data = calc_SNR_data(muy, muy_dark, sigmay)
    SNR_fit = calc_SNR_fit(QE, mup, sigmad2, K)
    SNR_lim = calc_SNR_lim(mup)

    # plot SNR
    fig, ax = plt.subplots(num="SNR")
    ax.scatter(mup, 20 * np.log10(SNR_data), label="Data")
    ax.plot(mup, 20 * np.log10(SNR_fit), label="Fit")
    ax.plot(mup, 20 * np.log10(SNR_lim), label="Theor. Limit")
    ax.set_xscale("log", base=10)
    ax.set_xlabel(r'$\mu_p \; \mathrm{(<\sharp photons>/pixel)}$')
    ax.set_ylabel("SNR (dB)")
    ax.set_title("SNR")
    ax.legend()

    return ax


def calc_DR(mup_sat, mup_min):
    """
    helper function to calculate Dynamic Range (DR), Eqn. (28)
    :param mup_sat: saturation capacity mup_sat
    :param mup_min: absolute sensitivity threshold mup_min
    :return: DR [dB]
    """
    return 20 * np.log10(mup_sat / mup_min)


def calc_LE(muy, muy_dark, H):
    """
    helper function to calculate Linearity Error LE, Eqn. (63)
    :param muy: an array of 50 muy
    :param muy_dark: an array of 50 muy_dark
    :param H: an array of 50 radiant exposure values
    :return: LE [%]
    """
    y = muy - muy_dark

    # fit: 5-95% saturation range
    lower_limit_idx = np.argmax(y > 0.05 * y[-1])
    upper_limit_idx = np.argmax(y > 0.95 * y[-1])

    # linear regression
    p = np.polyfit(x=H[lower_limit_idx:upper_limit_idx], y=y[lower_limit_idx:upper_limit_idx],
                   deg=1, w=1/y[lower_limit_idx:upper_limit_idx])
    y_fit = np.polyval(p, H)

    # plot linearity
    plt.figure("Linearity")
    plt.plot(H, y, label="Data")
    plt.plot(H, y_fit, linestyle=':', label="Fit")
    plt.scatter(H[lower_limit_idx], y[lower_limit_idx], color='k', label="Fit Range")
    plt.scatter(H[upper_limit_idx - 1], y[upper_limit_idx - 1], color='k')
    plt.legend()
    plt.xlabel(r'$\mathrm{H \; (pJ/cm^2)}$')
    plt.ylabel(r'$\mu_y - \mu_{y.dark} \; \mathrm{(DN)}$')
    plt.title("Linearity")

    # calculate relative deviation Eqn. (62)
    delta = 100 * (y - y_fit) / y_fit

    # calculate LE
    LE = np.mean(delta[lower_limit_idx:upper_limit_idx])

    return LE


