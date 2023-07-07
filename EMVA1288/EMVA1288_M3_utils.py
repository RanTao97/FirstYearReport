# EMVA1288_M3_utils contains all functions required to compute EMVA1288 (1) spatial nonuniformity and (2) defect pixels

import numpy as np
from scipy.ndimage import uniform_filter, convolve
from scipy.stats import norm
import matplotlib.pyplot as plt

import constants

def find_t_exp_50(t_exp, muy):
    """
    helper function to find the t_exp for 50% saturation
    N.B. SUBJECTED TO ROUNDING BY REAL CAMERA SETTING, ALWAYS CHECK THE t_exp BY QUERYING THE CAMERA
    :param t_exp: an array of 50 t_exp determined from measurement condition 6.5.5
    :param muy: the corresponding muy taken at each t_exp in the sensitivity measurement
    :return: t_exp_50 for 50% saturation
    """
    return np.interp(0.5*(np.amin(muy) + np.amax(muy)), muy, t_exp)


def calc_y_mean_img(y):
    """
    helper function to calculate <y>, Eqn. (33)
    :param y: an array of 100 images, of size MxNx100, i.e., L=100
    :return: y_mean_img <y>
    """
    return np.mean(y, axis=2)


def calc_HPF(y):
    """
    helper function to calculate HPF y, according to 8.1 Correction for Uneven Illumination
    :param y: an array of 100 images, of size MxNx100, i.e., L=100
            or a 2D array for just one image, of size MxN
    :return: y_HPF, HPF images, of size MxNx100, i.e., L=100
    """
    y_HPF = np.zeros_like(y)
    binomial_filter = 1/16 * np.array([[1, 2, 1],
                                       [2, 4, 2],
                                       [1, 2, 1]])
    if len(y.shape) == 3:  # image stack
        for l in range(y.shape[2]):
            img = y[:, :, l]
            # smooth with a 7x7 box filter
            img = uniform_filter(img, size=7, mode='reflect')
            # smooth with a 11x11 box filter
            img = uniform_filter(img, size=11, mode='reflect')
            # smooth with 3x3 binomial filter
            img = convolve(img, weights=binomial_filter, mode='reflect')
            # subtract the smoothed image to get the HPF image
            y_HPF[:, :, l] = y[:, :, l] - img
    if len(y.shape) == 2:  # just one 2D image
        # smooth with a 7x7 box filter
        img = uniform_filter(y, size=7, mode='reflect')
        # smooth with a 11x11 box filter
        img = uniform_filter(img, size=11, mode='reflect')
        # smooth with 3x3 binomial filter
        img = convolve(img, weights=binomial_filter, mode='reflect')
        y_HPF = y - img
    return y_HPF


def calc_sigmay_stack2(y_HPF):
    """
    helper function to calculate temporal noise sigmay_stack2, # Eqn. (65)
    :param y_HPF: HPF images, of size MxNx100, i.e., L=100
    :return: sigmay_stack2
    """
    sigmas2 = np.var(y_HPF, axis=2, ddof=1)
    sigmay_stack2 = np.mean(sigmas2)
    return sigmay_stack2


def calc_sy_meas2(y_HPF):
    """
    helper function to calculate sy_meas2 Eqn. 33-34
    :param y_HPF: HPF images, of size MxNx100, i.e., L=100
    :return: sy_meas2, measured spatial variance
    """
    y_mean_img = calc_y_mean_img(y_HPF)  # Eqn. (33)
    muy = np.mean(y_mean_img)  # Eqn. (34)
    sy_meas2 = np.mean((y_mean_img - muy) ** 2)  # Eqn. (35)
    return sy_meas2


def calc_sy_stack2(y_HPF):
    """
    helper function to calculate spatial variance Eqn. (33)-(36) & (65)
    :param y_HPF: HPF images, of size MxNx100, i.e., L=100
    :return: sy_stack2, spatial variance
    """
    sy_meas2 = calc_sy_meas2(y_HPF)
    sigmay_stack2 = calc_sigmay_stack2(y_HPF)
    sy_stack2 = sy_meas2 - sigmay_stack2 / (y_HPF.shape[2])  # Eqn. (36)
    return sy_stack2


def calc_DSNU(sy_dark, K):
    """
    helper function to calculate DSNU1288, Eqn. (66)
    :param sy_dark: spatial std of dark image
    :param K: overall system gain K
    :return: DSNU1288 [e-]
    """
    return sy_dark / K


def calc_PRNU(sy_502, sy_dark2, muy_50, muy_dark):
    """
    helper function to calculate PRNU1288, Eqn.(67)
    :param sy_502: spatial variance at 50 saturation
    :param sy_dark2: spatial variance aof dark image
    :param muy_50: mean of not HPF images
    :param muy_dark: mean of not HPF images
    :return: PRNU1288 [-]
    """
    return np.sqrt(sy_502 - sy_dark2) / (muy_50 - muy_dark)


def calc_sy_components(y_HPF):
    """
    helper function to calculate column, row and pixel spatial variance Eqn. (37)-(42)
    :param y_HPF: HPF images, of size MxNx100, i.e., L=100
    :return: sy_col2, sy_row2, sy_pix2
    """
    (M, N, L) = y_HPF.shape
    sy_2 = calc_sy_stack2(y_HPF)  # for Eqn. (37)
    y_mean_img = calc_y_mean_img(y_HPF)  # Eqn. (33)
    muy_col = np.mean(y_mean_img, axis=1)
    muy_row = np.mean(y_mean_img, axis=0)
    muy = np.mean(y_mean_img)  # Eqn. (34)
    sigmay_stack2 = calc_sigmay_stack2(y_HPF)
    sy_cav2 = np.mean((muy_row - muy)**2) - sigmay_stack2 / (L * M)  # Eqn. (41)
    sy_rav2 = np.mean((muy_col - muy)**2) - sigmay_stack2 / (L * N)  # Eqn. (41)

    print(f'sy_components: {sy_cav2}, {sy_rav2}')
    plt.figure()
    plt.plot(muy_col)
    plt.figure()
    plt.plot(muy_row)
    # Eqn. (42)
    A = np.array([[1, 0, 1/M],
                  [0, 1, 1/N],
                  [1, 1, 1]])
    b = np.array([sy_cav2, sy_rav2, sy_2])
    x = np.linalg.solve(A, b)
    sy_col2 = x[0]
    sy_row2 = x[1]
    sy_pix2 = x[2]
    return sy_col2, sy_row2, sy_pix2


def calc_SNR_tot(QE, mup, sigmad2, K, DSNU, PRNU):
    """
    helper function to calculate SNR_tot, Eqn. (69)
    :param QE: quantum efficiency
    :param mup: an array of 50 mup
    :param sigmad2: temporal dark noise sigmad2
    :param K: overall gain K
    :param DSNU: DSNU
    :param PRNU: PRNU
    :return: SNR_tot
    """
    return QE * mup / np.sqrt(sigmad2 + constants.sigmaq2/(K**2) + QE * mup + DSNU**2 + (PRNU * QE * mup)**2)


def plot_SNR_tot(QE, mup, sigmad2, K, DSNU, PRNU, ax):
    """
    helper function to plot SNR_tot overlaid on the existing SNR plot ax
    :param QE: quantum efficiency
    :param mup: an array of 50 mup
    :param sigmad2: temporal dark noise sigmad2
    :param K: overall gain K
    :param DSNU: DSNU
    :param PRNU: PRNU
    :param ax: existing figure handle ax
    :return: nothing
    """
    SNR_tot = calc_SNR_tot(QE, mup, sigmad2, K, DSNU, PRNU)
    ax.plot(mup, 20 * np.log10(SNR_tot), c='k', linestyle=':', label="Fit Tot.")
    ax.set_xscale("log", base=10)
    ax.set_xlabel(r'$\mu_p \mathrm{(<\sharp photons>/pixel)}$')
    ax.set_ylabel("SNR (dB)")
    ax.set_title("SNR")
    ax.legend()
    return


def calc_spec_DSNU(y_dark, y_dark_HPF):
    """
    helper function to calculate spectrogram DSNU
    N.B.: spectrograms are computed from the HPF <ydark>, <ydark> is the mean of NON HPF images
    :param y_dark: 100 NON HPF dark images
    :param y_dark_HPF: 100 HPF dark images
    :return: nothing
    """
    sy_meas = np.sqrt(calc_sy_meas2(y_dark_HPF))
    sy_dark = np.sqrt(calc_sy_stack2(y_dark_HPF))
    sigmay_dark_stack = np.sqrt(calc_sigmay_stack2(y_dark))
    (M, N, L) = y_dark.shape
    y = calc_HPF(calc_y_mean_img(y_dark))  # HPF on <ydark>
    y = y - np.mean(y)
    # FT on row
    y_hat_row = np.fft.rfft(y, axis=1, norm="ortho")
    # horizontal power spectrum
    p_row = np.mean(np.abs(y_hat_row)**2, axis=0)
    # show horizontal spectrogram DSNU
    plt.figure("Horizontal Spectrogram DSNU")
    plt.plot(np.arange(p_row.size) / N, np.sqrt(p_row), label="Data")
    plt.axhline(sy_meas, c='k', linestyle=':', label=r'$s_{y.measured.dark}$')
    plt.axhline(sy_dark, c='r', linestyle=':', label=r'$DSNU_{1288.DN}$')
    plt.axhline(sigmay_dark_stack, c='g', linestyle=':', label=r'$\sigma_{y.stack.dark}$')
    plt.yscale("log")
    plt.xlabel("cycles (periods/pixel)")
    plt.ylabel("Standard Deviation and Relative Presence of Each Cycle (DN)")
    plt.title("Horizontal Spectrogram DSNU")
    plt.legend()

    # FT on column
    y_hat_col = np.fft.rfft(y, axis=0, norm="ortho")
    # vertical power spectrum
    p_col = np.mean(np.abs(y_hat_col)**2, axis=1)
    # show vertical spectrogram DSNU
    plt.figure("Vertical Spectrogram DSNU")
    plt.plot(np.arange(p_col.size) / M, np.sqrt(p_col), label="Data")
    plt.axhline(sy_meas, c='k', linestyle=':', label=r'$s_{y.measured.dark}$')
    plt.axhline(sy_dark, c='r', linestyle=':', label=r'$DSNU_{1288.DN}$')
    plt.axhline(sigmay_dark_stack, c='g', linestyle=':', label=r'$\sigma_{y.stack.dark}$')
    plt.yscale("log")
    plt.xlabel("cycles (periods/pixel)")
    plt.ylabel("Standard Deviation and Relative Presence of Each Cycle (DN)")
    plt.title("Vertical Spectrogram DSNU")
    plt.legend()
    return


def calc_spec_PRNU(y_50, y_dark, y_50_HPF, y_dark_HPF, PRNU):
    """
    helper function to calculate spectrogram PRNU
    N.B.: spectrograms are computed from the HPF <y50> - <ydark>,
          <y50>, <ydark> are the mean of NON HPF images
    :param y_50: 100 NON HPF images at 50% saturation
    :param y_dark: 100 NON HPF dark images
    :param y_50_HPF: 100 HPF images at 50% saturation
    :param y_dark_HPF: 100 HPF dark images
    :param PRNU: PRNU
    :return: nothing
    """
    sigmay_stack = np.sqrt(calc_sigmay_stack2(y_50))
    sy_meas2 = calc_sy_meas2(y_50_HPF - y_dark_HPF)
    muy_50 = np.mean(y_50)
    muy_dark = np.mean(y_dark)
    (M, N, L) = y_50.shape
    y = calc_HPF(calc_y_mean_img(y_50) - calc_y_mean_img(y_dark))  # HPF on <y50> - <ydark>
    y = y - np.mean(y)
    # FT on row
    y_hat_row = np.fft.rfft(y, axis=1, norm="ortho")
    # horizontal power spectrum
    p_row = np.mean(np.abs(y_hat_row)**2, axis=0)
    # show horizontal spectrogram PRNU
    plt.figure("Horizontal Spectrogram PRNU")
    plt.plot(np.arange(p_row.size) / N, np.sqrt(p_row) / (muy_50 - muy_dark) * 100, label="Data")
    plt.axhline(np.sqrt(sy_meas2) / (muy_50 - muy_dark) * 100, c='k', linestyle=':', label=r'$s_{y.measured.stack.\%}$')
    plt.axhline(PRNU * 100, c='r', linestyle=':', label=r'$PRNU_{1288}$')
    plt.axhline(sigmay_stack / (muy_50 - muy_dark) * 100, c='g', linestyle=':', label=r'$\sigma_{y.stack.\%}$')
    plt.yscale("log")
    plt.xlabel("cycles (periods/pixel)")
    plt.ylabel("Standard Deviation and Relative Presence of Each Cycle (%)")
    plt.title("Horizontal Spectrogram PRNU")
    plt.legend()

    # FT on column
    y_hat_col = np.fft.rfft(y, axis=0, norm="ortho")
    # vertical power spectrum
    p_col = np.mean(np.abs(y_hat_col)**2, axis=1)
    # show vertical spectrogram PRNU
    plt.figure("Vertical Spectrogram PRNU")
    plt.plot(np.arange(p_col.size) / M, np.sqrt(p_col) / (muy_50 - muy_dark) * 100, label="Data")
    plt.axhline(np.sqrt(sy_meas2) / (muy_50 - muy_dark) * 100, c='k', linestyle=':', label=r'$s_{y.measured.stack.\%}$')
    plt.axhline(PRNU * 100, c='r', linestyle=':', label=r'$PRNU_{1288}$')
    plt.axhline(sigmay_stack / (muy_50 - muy_dark) * 100, c='g', linestyle=':', label=r'$\sigma_{y.stack.\%}$')
    plt.yscale("log")
    plt.xlabel("cycles (periods/pixel)")
    plt.ylabel("Standard Deviation and Relative Presence of Each Cycle (%)")
    plt.title("Vertical Spectrogram PRNU")
    plt.legend()
    return


def calc_profile_DSNU(y_dark):
    """
    helper function to calculate DSNU profiles, 8.7 Horizontal and Vertical Profiles
    :param y_dark: 100 NON HPF dark images
    :return: nothing
    """
    (M, N, L) = y_dark.shape
    y = calc_y_mean_img(y_dark)

    # horizontal profile
    mid_row = y[int(M/2), :]
    mean_row = np.mean(y, axis=0)
    max_row = np.amax(y, axis=0)
    min_row = np.amin(y, axis=0)
    # plot horizontal profile
    plt.figure("Horizontal Profile DSNU")
    plt.plot(np.arange(N), mid_row, label="Mid")
    plt.plot(np.arange(N), mean_row, label="Mean")
    plt.plot(np.arange(N), max_row, label="Max")
    plt.plot(np.arange(N), min_row, label="Min")
    plt.xlabel("Horizontal Position (pixel)")
    plt.ylabel("Grey Value (DN)")
    plt.title("Horizontal Profile DSNU")
    # Todo: 20230621, change lower limit from 0.9 to 0.8, upper limit from 1.1 to 1.2
    plt.ylim(0.8*np.mean(min_row), 1.2*np.mean(max_row))
    plt.legend()

    # vertical profile
    mid_col = y[:, int(N / 2)]
    mean_col = np.mean(y, axis=1)
    max_col = np.amax(y, axis=1)
    min_col = np.amin(y, axis=1)
    # plot vertical profile
    plt.figure("Vertical Profile DSNU")
    plt.plot(mid_col, np.arange(M), label="Mid")
    plt.plot(mean_col, np.arange(M), label="Mean")
    plt.plot(max_col, np.arange(M), label="Max")
    plt.plot(min_col, np.arange(M), label="Min")
    plt.ylabel("Vertical Position (pixel)")
    plt.xlabel("Grey Value (DN)")
    plt.title("Vertical Profile DSNU")
    # Todo: 20230621, change lower limit from 0.9 to 0.8, upper limit from 1.1 to 1.2
    plt.xlim(0.8 * np.mean(min_col), 1.2 * np.mean(max_col))
    # flip y axis for vertical profile
    bottom, top = plt.ylim()
    plt.ylim(top, bottom)
    plt.legend()
    return

def calc_profile_PRNU(y_50, y_dark):
    """
    helper function to calculate PRNU profiles, 8.7 Horizontal and Vertical Profiles
    :param y_50: 100 NON HPF images at 50% saturation
    :param y_dark: 100 NON HPF dark images
    :return: nothing
    """
    (M, N, L) = y_50.shape
    y = calc_y_mean_img(y_50) - calc_y_mean_img(y_dark)

    # horizontal profile
    mid_row = y[int(M/2), :]
    mean_row = np.mean(y, axis=0)
    max_row = np.amax(y, axis=0)
    min_row = np.amin(y, axis=0)
    # plot horizontal profile
    plt.figure("Horizontal Profile PRNU")
    plt.plot(np.arange(N), mid_row, label="Mid")
    plt.plot(np.arange(N), mean_row, label="Mean")
    plt.plot(np.arange(N), max_row, label="Max")
    plt.plot(np.arange(N), min_row, label="Min")
    plt.xlabel("Horizontal Position (pixel)")
    plt.ylabel("Grey Value (DN)")
    plt.title("Horizontal Profile PRNU")
    plt.ylim(0.9*np.mean(min_row), 1.1*np.mean(max_row))
    plt.legend()

    # vertical profile
    mid_col = y[:, int(N / 2)]
    mean_col = np.mean(y, axis=1)
    max_col = np.amax(y, axis=1)
    min_col = np.amin(y, axis=1)
    # plot vertical profile
    plt.figure("Vertical Profile PRNU")
    plt.plot(mid_col, np.arange(M), label="Mid")
    plt.plot(mean_col, np.arange(M), label="Mean")
    plt.plot(max_col, np.arange(M), label="Max")
    plt.plot(min_col, np.arange(M), label="Min")
    plt.ylabel("Vertical Position (pixel)")
    plt.xlabel("Grey Value (DN)")
    plt.title("Vertical Profile PRNU")
    plt.xlim(0.9 * np.mean(min_col), 1.1 * np.mean(max_col))
    # flip y axis for vertical profile
    bottom, top = plt.ylim()
    plt.ylim(top, bottom)
    plt.legend()
    return


def calc_defect_pix_DSNU(y_dark):
    """
    helper function to characterize defect pixel DSNU, 8.8 Defect Pixel Characterisation
    N.B.: histograms are computed from the HPF <ydark>, <ydark> is the mean of NON HPF images
    :param y_dark: 100 NON HPF dark images
    :return: nothing
    """
    (M, N, L) = y_dark.shape
    y = calc_HPF(calc_y_mean_img(y_dark))  # HPF on <ydark>
    y = y - np.mean(y)  # deviation

    # histogram
    y_min = np.amin(y)
    y_max = np.amax(y)

    I = np.floor(L*(y_max - y_min) / 256) + 1
    Q = L*(y_max - y_min)/I + 1
    hist, bin_edges = np.histogram(y, bins=int(Q))
    loc, scale = norm.fit(y)
    pdf = norm.pdf(bin_edges, loc, scale) * M * N
    plt.figure("Log Histogram DSNU")
    plt.stairs(hist, bin_edges, fill=True, label="Data")
    plt.plot(bin_edges, pdf * (bin_edges[1] - bin_edges[0]), linestyle=':', label="Fit")
    plt.yscale("log")
    plt.ylim(bottom=0.5, top=max(np.amax(hist), np.amax(pdf)))
    plt.xlabel("Deviation from Dark Value (DN)")
    plt.ylabel("Number of Pixel/Bin")
    plt.title("Log Histogram DSNU")
    plt.legend()

    # accumulated histogram
    y = np.abs(y)
    y_min = np.amin(y)
    y_max = np.amax(y)
    I = np.floor(L * (y_max - y_min) / 256) + 1
    Q = L * (y_max - y_min) / I + 1
    hist, bin_edges = np.histogram(y, bins=int(Q))
    hist_cum = np.cumsum(hist[::-1])[::-1]
    plt.figure("Accum. Log Histogram DSNU")
    plt.stairs(hist_cum, bin_edges, fill=True)
    plt.yscale("log")
    plt.ylim(bottom=0.5)
    plt.xlabel("Abs Deviation from Dark Value (DN)")
    plt.ylabel("Number of Pixel/Bin")
    plt.title("Accum. Log Histogram DSNU")
    return


def calc_defect_pix_PRNU(y_50, y_dark):
    """
    helper function to characterize defect pixel PRNU, 8.8 Defect Pixel Characterisation
    N.B.: histograms are computed from the HPF <y50> - <ydark>,
          <y50>, <ydark> are the mean of NON HPF images
    :param y_50: 100 NON HPF images at 50% saturation
    :param y_dark: 100 NON HPF dark images
    :return: nothing
    """
    (M, N, L) = y_50.shape
    y = calc_HPF(calc_y_mean_img(y_50) - calc_y_mean_img(y_dark))  # HPF on <y50> - <ydark>
    y = y - np.mean(y)  # deviation

    # histogram
    y_min = np.amin(y)
    y_max = np.amax(y)
    I = np.floor(L*(y_max - y_min) / 256) + 1
    Q = L*(y_max - y_min)/I + 1
    hist, bin_edges = np.histogram(y, bins=int(Q))
    loc, scale = norm.fit(y)
    pdf = norm.pdf(bin_edges, loc, scale) * M * N
    plt.figure("Log Histogram PRNU")
    plt.stairs(hist, bin_edges, fill=True, label="Data")
    plt.plot(bin_edges, pdf * (bin_edges[1] - bin_edges[0]), linestyle=':', label="Fit")
    plt.yscale("log")
    plt.ylim(bottom=0.5, top=max(np.amax(hist), np.amax(pdf)))
    plt.xlabel("Deviation from Mean Value (DN)")
    plt.ylabel("Number of Pixel/Bin")
    plt.title("Log Histogram PRNU")
    plt.legend()

    # accumulated histogram
    y = np.abs(y)
    y_min = np.amin(y)
    y_max = np.amax(y)
    I = np.floor(L * (y_max - y_min) / 256) + 1
    Q = L * (y_max - y_min) / I + 1
    hist, bin_edges = np.histogram(y, bins=int(Q))
    hist_cum = np.cumsum(hist[::-1])[::-1]
    plt.figure("Accum. Log Histogram PRNU")
    plt.stairs(hist_cum, bin_edges, fill=True)
    plt.yscale("log")
    plt.ylim(bottom=0.5)
    plt.xlabel("Abs Deviation from Mean Value (DN)")
    plt.ylabel("Number of Pixel/Bin")
    plt.title("Accum. Log Histogram PRNU")
    return









