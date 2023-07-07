# plot_LUCID_QE is to plot the relative QE curve of LUCID Atlas Camera

import numpy as np
import matplotlib.pyplot as plt

PATH_490 = "20230521_LUCID_ATP013S_230101680_LED490nm/M1Data/results.npz"
PATH_660 = "20230521_LUCID_ATP013S_230101680_LED660nm/M1Data/results.npz"
PATH_780 = "20230521_LUCID_ATP013S_230101680_LED780nm/M1Data/results.npz"
PATH_1050 = "20230526_LUCID_ATP013S_230101680_LED1050nm/M1Data/results.npz"
PATH_1200 = "20230602_LUCID_ATP013S_230101680_LED1200nm/M1Data/results.npz"
PATH_1550 = "20230602_LUCID_ATP013S_230101680_LED1550nm/M1Data/results.npz"

PATH_SONY_QE = "D:/OneDrive - University of Cambridge/SWIR Camera Lists/SONY SenSWIR QE.txt"
PATH_LUCID_QE = "D:/OneDrive - University of Cambridge/LUCID Atlas Camera/LUCID Atlas QE.txt"

# set figure size and font size
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.titlesize'] = 'x-large'
plt.rcParams['axes.labelsize'] = 'large'
plt.rcParams['xtick.labelsize'] = 'large'
plt.rcParams['ytick.labelsize'] = 'large'
plt.rcParams['legend.fontsize'] = 'large'

if __name__ == '__main__':
    """Read Measured QE"""
    wavelength_meas = np.zeros(6)
    QE_meas = np.zeros(6)
    for path_idx, path in enumerate([PATH_490, PATH_660, PATH_780, PATH_1050, PATH_1200, PATH_1550]):
        results = np.load(path)
        wavelength_meas[path_idx] = results['WAVELENGTH']
        QE_meas[path_idx] = results['QE']
        print(f"At {results['WAVELENGTH']:.5f} nm, E = {results['E']}: K = {results['K']:.5f}, QE = {results['QE'] * 100:.5f}%")

    """Read SONY and LUCID QE"""
    wavelength_SONY = np.loadtxt(PATH_SONY_QE, usecols=0)
    QE_SONY = np.loadtxt(PATH_SONY_QE, usecols=1)
    wavelength_LUCID = np.loadtxt(PATH_LUCID_QE, usecols=0)
    QE_LUCID = np.loadtxt(PATH_LUCID_QE, usecols=1)

    """Plot Relative QE Curve"""
    plt.figure("Relative QE")
    plt.plot(wavelength_SONY, QE_SONY * 100, label="SONY")
    plt.plot(wavelength_LUCID, QE_LUCID / np.amax(QE_LUCID) * 100, label="LUCID", marker=".")
    plt.scatter(wavelength_meas, QE_meas / np.amax(QE_meas) * 100, label="Measured", c="black", marker="x")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Relative QE (%)")
    plt.legend()
    plt.show()