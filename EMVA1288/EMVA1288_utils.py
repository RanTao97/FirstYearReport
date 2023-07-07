# EMVA1288_utils implements some basic equation in EMVA1288 standards
# These equations are used by EMVA1288_M1_utils, EMVA1288_M2_utils, EMVA1288_M3_utils

import numpy as np

import constants

def calc_H(E, t_exp):
    """
    helper function to calculate radiant exposure H, Eqn. (1)
    :param E: irradiance [uW/cm^2]
    :param t_exp: exposure time [us]
    :return: H [pJ/cm^2]
    """
    return E * t_exp

def calc_mup(A, E, t_exp, wavelength):
    """
    helper function to calculate mup, Eqn. (3)
    :param A: pixel area [um^2]
    :param E: irradiance [uW/cm^2]
    :param t_exp: exposure time [us]
    :param wavelength: illumination wavelength [nm]
    :return: mup [-]
    """
    return A * E * t_exp * wavelength * constants.coeff_3


def calc_muy(img0, img1):
    """
    helper function to calculate muy, Eqn. (16)
    :param img0: image 0
    :param img1: image 1
    :return: muy
    """
    return 0.5 * (np.mean(img0) + np.mean(img1))


def calc_muy_stack(img):
    """
    helper function to calculate muy_stack, Eqn. (16)
    :param img: image stack, of size MxNxL
    :return: muy_stack
    """
    return np.mean(np.mean(img, axis=(0, 1)))


def calc_sigmay2(img0, img1):
    """
    helper function to calculate sigma_y^2, Eqn. (18)
    :param img0: image 0
    :param img1: image 1
    :return: sigmay2
    """
    mu0 = np.mean(img0)
    mu1 = np.mean(img1)
    sigmay2 = 0.5 * (np.mean((img0 - img1)**2) - (mu0 - mu1)**2)
    return sigmay2
