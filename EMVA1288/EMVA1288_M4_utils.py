# EMVA1288_M3_utils contains all functions required to compute EMVA1288 spectral sensitivity

import numpy as np
import matplotlib.pyplot as plt

import constants

def find_t_exp(mup, A, E, wavelength):
    """
    find t_exp at the measured wavelength so that it has the same mup (50% saturation) as that at the main operating wavelength
    N.B. SUBJECTED TO ROUNDING BY REAL CAMERA SETTING, ALWAYS CHECK THE t_exp BY QUERYING THE CAMERA
    :param mup: an array of 50 mup measured at the main operating wavelength, 6.4 Calibration of Irradiation
    :param A: pixel area [um^2]
    :param E: irradiance [uW/cm^2] at the measured wavelength
    :param wavelength: illumination wavelength [nm]
    :return: t_exp [us]
    """
    mup_50 = np.median(mup)
    return mup_50 / (A * E * wavelength * constants.coeff_3)


def calc_QE(muy, muy_dark, K, mup):
    """
    helper function to calculate quantum efficiency QE at the measured wavelengths, Eqn. (82)
    :param muy: an array of muy at measured wavelengths
    :param muy_dark: an array of muy_dark at measured wavelengths
    :param K: overall system gain
    :param mup: an array of mup at measured wavelengths
    :return: QE_dist: an array QE at the measured wavelengths [-]
    """
    return (muy - muy_dark) / (K * mup)


def plot_QE(wave_meas, QE_meas, wave_MNF, QE_MNF):
    """
    helper function to plot spectral sensitivity, QE v.s. wavelength
    :param wave_meas: an array of measured wavelength [nm]
    :param QE_meas: an array QE at the measured wavelengths [-]
    :param wave_MNF: an array of wavelengths extracted from manufacture's datasheet, is None if not available
    :param QE_MNF: an array of QE extracted from manufacture's datasheet, is None if not available
    :return: nothing
    """
    plt.figure("Spectral Sensitivity")
    plt.scatter(wave_meas, QE_meas * 100 / 5.29, label="Meas")
    if (wave_MNF is not None) and (QE_MNF is not None):
        plt.plot(wave_MNF, QE_MNF * 100, label="MNF", c='k')
        plt.legend()
    plt.xlabel("Wavelength (nm)")
    plt.ylabel(r'$\mathrm{Quantum \; Efficiency} \; \eta \; \mathrm{(\%)}$')
    plt.title("Spectral Sensitivity")
    return

