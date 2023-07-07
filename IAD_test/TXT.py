import numpy as np
from os import path
import matplotlib.pyplot as plt

import constants

class TXT:
    """
    this is a .txt class
    contain info from .txt file as IAD output
    """
    def __init__(self, file, _RXT_path):
        """
        to initialise TXT class
        :param file: .txt file path
        :param _RXT_path: the corresponding RXT file
        """
        if not path.isfile(file):
            raise Exception(f"Cannot find file")

        self.path = file
        self.wavelength = np.loadtxt(file, comments='#', usecols=0)
        self.M_R_meas = np.loadtxt(file, comments='#', usecols=1)
        self.M_R_fit = np.loadtxt(file, comments='#', usecols=2)
        self.M_T_meas = np.loadtxt(file, comments='#', usecols=3)
        self.M_T_fit = np.loadtxt(file, comments='#', usecols=4)
        self.mu_a_est = np.loadtxt(file, comments='#', usecols=5)
        self.mu_sp_est = np.loadtxt(file, comments='#', usecols=6)
        self.err_code = np.loadtxt(file, dtype=np.dtype(str), skiprows=44, comments=None, usecols=9)

        # read true mu_a, mu_sp from .rxt
        if path.isfile(_RXT_path):
            self.mu_a_true = np.loadtxt(_RXT_path, skiprows=31, comments=None, usecols=4)
            self.mu_sp_true = np.loadtxt(_RXT_path, skiprows=31, comments=None, usecols=5)


    def find_pnt(self, pnt_M_R, pnt_M_T):
        """
        this is to find the point given its M_R, M_T coordinates
        :param pnt_M_R: M_R of the point
        :param pnt_M_T: M_T of the point
        :return: nothing
        """
        pnt_idx = np.argmin((self.M_R_meas - pnt_M_R)**2+(self.M_T_meas - pnt_M_T)**2)

        print(f"{pnt_idx+2}\t"  # start from 2
              f"M_R_meas: {self.M_R_meas[pnt_idx]}\tM_R_fit: {self.M_R_fit[pnt_idx]}\t"
              f"M_T_meas: {self.M_T_meas[pnt_idx]}\tM_T_fit: {self.M_T_fit[pnt_idx]}\t"
              f"mu_a_est: {self.mu_a_est[pnt_idx]}\tmu_a_true: {self.mu_a_true[pnt_idx]}\t"
              f"mu_sp_est: {self.mu_sp_est[pnt_idx]}\tmu_sp_true: {self.mu_sp_true[pnt_idx]}\t"
              f"err_code: {self.err_code[pnt_idx]}")
        print(f"err_mu_a: {self.calc_err_mu_a()[pnt_idx]}%\terr_mu_sp: {self.calc_err_mu_sp()[pnt_idx]}%")

    def calc_err_mu_a(self):
        """
        this is to calculate % relative errors in mu_a
        :return: err_mu_a
        """
        err_mu_a = (self.mu_a_est - self.mu_a_true) / (self.mu_a_true + 1e-6) * 100
        return err_mu_a

    def calc_err_mu_sp(self):
        """
        this is to calculate % relative errors in mu_sp
        :return: err_mu_sp
        """
        err_mu_sp = (self.mu_sp_est - self.mu_sp_true) / (self.mu_sp_true + 1e-6) * 100
        return err_mu_sp

    def plot_err_code(self):
        """
        this is to plot error code
        *  ==> Success          (blue)          0-9 ==> Monte Carlo Iteration
        R  ==> M_R is too big   (green)         r  ==> M_R is too small         (light blue)
        T  ==> M_T is too big   (pink)          t  ==> M_T is too small         (orange)
        U  ==> M_U is too big   (none)          u  ==> M_U is too small         (none)
        !  ==> M_R + M_T > 1    (none)          +  ==> Did not converge         (red)
        :return:
        """

        plt.figure()

        self.overlay_practical_range(plt)

        plt.scatter(self.M_R_meas[self.err_code == "*"], self.M_T_meas[self.err_code == "*"],
                    label="*", c="tab:blue", s=3)
        plt.scatter(self.M_R_meas[self.err_code == "R"], self.M_T_meas[self.err_code == "R"],
                    label="R", c="tab:green", s=3)
        plt.scatter(self.M_R_meas[self.err_code == "r"], self.M_T_meas[self.err_code == "r"],
                    label="r", c="tab:cyan", s=3)
        plt.scatter(self.M_R_meas[self.err_code == "T"], self.M_T_meas[self.err_code == "T"],
                    label="T", c="tab:pink", s=3)
        plt.scatter(self.M_R_meas[self.err_code == "t"], self.M_T_meas[self.err_code == "t"],
                    label="t", c="tab:orange", s=3)
        plt.scatter(self.M_R_meas[self.err_code == "+"], self.M_T_meas[self.err_code == "+"],
                    label="+", c="tab:red", s=3)
        plt.plot(np.arange(0, 1.1, 0.1), 1 - np.arange(0, 1.1, 0.1), 'k', alpha=0.5)
        plt.xlabel(r'$M_R$'+" (a.u.)")
        plt.ylabel(r'$M_T$'+" (a.u.)")
        
        plt.legend()

        return

    def plot_err_mu_a_sp(self):
        """
        this is to plot relative errors in mu_a, mu_sp separately
        :return: nothing
        """
        err_mu_a = self.calc_err_mu_a()
        err_mu_sp = self.calc_err_mu_sp()

        # 2 subplots, each for mu_a, mu_sp
        _, (ax1, ax2) = plt.subplots(1, 2, sharex='all', sharey='all')

        # mu_a
        ax1.set_title("Relative % Errors in " + r'$\mu_a$')
        ax1.set_xlabel(r'$M_R$'+" (a.u.)")
        ax1.set_ylabel(r'$M_T$'+" (a.u.)")
        # plot successful points
        # ToDo: self.overlay_practical_range(ax1)
        fig = ax1.scatter(self.M_R_meas[self.err_code == "*"], self.M_T_meas[self.err_code == "*"],
                          c=err_mu_a[self.err_code == "*"], vmin=-10, vmax=10, s=1,
                          label="Converged")
        plt.colorbar(mappable=fig, ax=ax1)
        # plot unsuccessful points in red
        ax1.scatter(self.M_R_meas[self.err_code != "*"], self.M_T_meas[self.err_code != "*"],
                    c='r', s=1, label="Failed")
        ax1.legend()

        # mu_sp
        ax2.set_title("Relative % Errors in " + r'$\mu_s^{\prime}$')
        ax2.set_xlabel(r'$M_R$'+" (a.u.)")
        ax2.set_ylabel(r'$M_T$'+" (a.u.)")
        # plot successful points
        # ToDo: self.overlay_practical_range(ax2)
        fig = ax2.scatter(self.M_R_meas[self.err_code == "*"], self.M_T_meas[self.err_code == "*"],
                          c=err_mu_sp[self.err_code == "*"], vmin=-5, vmax=5, s=1,
                          label="Converged")
        plt.colorbar(mappable=fig, ax=ax2)
        # plot unsuccessful points in red
        ax2.scatter(self.M_R_meas[self.err_code != "*"], self.M_T_meas[self.err_code != "*"],
                    c='r', s=1, label="Failed")
        ax2.legend()
        return

    def plot_err_mu(self):
        """
        this is to plot the average of relative errors in mu_a, mu_sp
        :return: nothing
        """
        # the error is defined by the larger error between mu_a and mu_sp
        err_mu = np.maximum(np.abs(self.calc_err_mu_a()), np.abs(self.calc_err_mu_sp()))
        plt.figure()
        plt.title(r'$Relative \/ \% \/ Errors \/ in \/ \mu_a \/ and \/\mu_{s}^{\prime}$')
        plt.xlabel(r'$M_R$')
        plt.ylabel(r'$M_T$')
        # plot successful points
        plt.scatter(self.M_R_meas[self.err_code == "*"], self.M_T_meas[self.err_code == "*"],
                    c=err_mu[self.err_code == "*"], s=3)
        plt.colorbar()
        # plot unsuccessful points in red
        plt.scatter(self.M_R_meas[self.err_code != "*"], self.M_T_meas[self.err_code != "*"],
                    c='r', s=3)
        # plot reference
        plt.plot(np.arange(0, 1.1, 0.1), 1 - np.arange(0, 1.1, 0.1), 'k', alpha=0.5)
        return

    def overlay_practical_range(self, hdl):
        """
        this is to overlay the practical range of mu_a: <= 0.05 [mm-1]
                                                  mu_sp: 0.5~2 [mm-1]
        ToDo: need sorted M_R, M_T by mu_a, mu_sp first, which is the case in MC simulation,
              so need to do nothing now
        :param: hdl, either plt or ax
        :return: nothing
        """
        err_tol = 1e-5

        # get the mu_a boundary
        mu_a_1 = constants.MU_A[np.argmin(np.abs(constants.MU_A - 0.0))]
        practical_range_1 = (self.mu_a_true >= mu_a_1-err_tol) & (self.mu_a_true <= mu_a_1+err_tol) &\
                            (self.mu_sp_true >= 0.5) & (self.mu_sp_true <= 2)

        mu_a_2 = constants.MU_A[np.argmin(np.abs(constants.MU_A - 0.05))]
        practical_range_2 = (self.mu_a_true >= mu_a_2-err_tol) & (self.mu_a_true <= mu_a_2+err_tol) & \
                            (self.mu_sp_true >= 0.5) & (self.mu_sp_true <= 2)

        # get the mu_sp boundary
        mu_sp_1 = constants.MU_SP[np.argmin(np.abs(constants.MU_SP - 0.5))]
        practical_range_3 = (self.mu_sp_true >= mu_sp_1 - err_tol) & (self.mu_sp_true <= mu_sp_1 + err_tol) & \
                            (self.mu_a_true >= mu_a_1) & (self.mu_a_true <= mu_a_2)

        mu_sp_2 = constants.MU_SP[np.argmin(np.abs(constants.MU_SP - 2))]
        practical_range_4 = (self.mu_sp_true >= mu_sp_2 - err_tol) & (self.mu_sp_true <= mu_sp_2 + err_tol) & \
                            (self.mu_a_true >= mu_a_1) & (self.mu_a_true <= mu_a_2)

        hdl.fill(np.concatenate((self.M_R_meas[practical_range_1], self.M_R_meas[practical_range_4],
                                 self.M_R_meas[practical_range_2][::-1], self.M_R_meas[practical_range_3][::-1]),
                                axis=0),
                 np.concatenate((self.M_T_meas[practical_range_1], self.M_T_meas[practical_range_4],
                                 self.M_T_meas[practical_range_2][::-1], self.M_T_meas[practical_range_3][::-1]),
                                axis=0),
                 label=f"Practical Range:\n"+
                       r'$\mu_a: $'+f"{mu_a_1:.2f}-{mu_a_2:.2f} "+r'$\mathrm{mm^{-1}}$' + "\n"
                       r'$\mu_s^{\prime} : $'+f"{mu_sp_1:.2f}-{mu_sp_2:.2f} "+r'$\mathrm{mm^{-1}}$',
                 color="lightgrey")

        return

    def check_err_trend(self):
        """
        this is to check the trend of error given at certain mu_sp against increasing mu_a
        ToDo: need sorted M_R, M_T by mu_a, mu_sp first, which is the case in MC simulation,
              so need to do nothing now
        :return:
        """
        err_tol = 1e-5

        # find mu_sp boundaries: 0.5-2 mm-1
        mu_sp_1 = np.argmin(np.abs(constants.MU_SP - 0.5))
        mu_sp_2 = np.argmin(np.abs(constants.MU_SP - 2))

        # calculate errors
        err_mu_a = self.calc_err_mu_a()
        err_mu_sp = self.calc_err_mu_sp()

        # find points within boundaries
        # mu_a boundaries: 0-0.05 mm-1
        idx = (self.mu_sp_true >= constants.MU_SP[mu_sp_1] - err_tol) & \
              (self.mu_sp_true <= constants.MU_SP[mu_sp_2] + err_tol) & \
              (self.mu_a_true > 0) & (self.mu_a_true <= 0.05)

        # reshape to mu_a x mu_sp
        err_mu_a_check = err_mu_a[idx].reshape((-1, (mu_sp_2-mu_sp_1+1)))
        err_mu_sp_check = err_mu_sp[idx].reshape((-1, (mu_sp_2-mu_sp_1+1)))

        # summary as a function of mu_a
        err_mu_a_mean = np.mean(np.abs(err_mu_a_check), axis=1)
        err_mu_a_std = np.std(np.abs(err_mu_a_check), axis=1)

        err_mu_sp_mean = np.mean(np.abs(err_mu_sp_check), axis=1)
        err_mu_sp_std = np.std(np.abs(err_mu_sp_check), axis=1)


        # plot results
        _, (ax1, ax2) = plt.subplots(1, 2)

        ax1.plot(constants.MU_A[(constants.MU_A > 0) & (constants.MU_A <= 0.05)], np.abs(err_mu_a_check), marker='o')

        ax2.plot(constants.MU_A[(constants.MU_A > 0) & (constants.MU_A <= 0.05)], np.abs(err_mu_sp_check), marker='o')

        plt.figure()
        plt.plot(constants.MU_A[(constants.MU_A > 0) & (constants.MU_A <= 0.05)], err_mu_a_mean,
                 color="tab:blue", label="% Error in " + r'$\mu_a$' + " (Mean)")
        plt.fill_between(constants.MU_A[(constants.MU_A > 0) & (constants.MU_A <= 0.05)],
                         err_mu_a_mean-err_mu_a_std, err_mu_a_mean+err_mu_a_std,
                         alpha=0.5, color="tab:blue", edgecolor="None",
                         label="% Error in " + r'$\mu_a$' + " (Std)")

        plt.plot(constants.MU_A[(constants.MU_A > 0) & (constants.MU_A <= 0.05)], err_mu_sp_mean,
                 color="tab:orange", label="% Error in " + r'$\mu_s^{\prime}$' + " (Mean)")
        plt.fill_between(constants.MU_A[(constants.MU_A > 0) & (constants.MU_A <= 0.05)],
                         err_mu_sp_mean - err_mu_sp_std, err_mu_sp_mean + err_mu_sp_std,
                         alpha=0.5, color="tab:orange", edgecolor="None",
                         label="% Error in " + r'$\mu_s^{\prime}$' + " (Std)")
        plt.xlabel(r'$\mu_a\/(\mathrm{mm^{-1}})$')
        plt.ylabel("% Error")
        plt.legend()





