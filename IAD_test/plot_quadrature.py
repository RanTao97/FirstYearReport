# This is to plot the effects of the number of quadrature points

import numpy as np
import matplotlib.pyplot as plt

# set figure size and font size
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.titlesize'] = 'x-large'
plt.rcParams['axes.labelsize'] = 'large'
plt.rcParams['xtick.labelsize'] = 'large'
plt.rcParams['ytick.labelsize'] = 'large'
plt.rcParams['legend.fontsize'] = 'large'

if __name__ == '__main__':
    # case 1: mu_a = 0.0000		mu_sp = 0.0500
    # case 2: mu_a = 0.2000		mu_sp = 0.0500
    q_points = np.array([12, 24, 36])
    mu_a_true = np.array([0.0000, 0.2000])
    mu_sp_true = np.array([0.0500, 0.0500])
    mu_a_q = np.array([[7.662e-003, 2.045e-001],  # q12
                       [4.562e-003, 2.017e-001],  # q24
                       [4.180e-003, 2.009e-001]])  # q36
    mu_sp_q = np.array([[4.159e-002, 4.513e-002],  # q12
                        [4.556e-002, 4.836e-002],  # q24
                        [4.601e-002, 4.914e-002]])  # q36

    mu_a_diff = mu_a_q - mu_a_true
    mu_sp_diff = mu_sp_q - mu_sp_true

    # plot results
    plt.figure("mu_a")
    plt.plot(q_points, mu_a_q[:, 0], label="Case 1 Est", c="tab:blue", marker='.')
    plt.axhline(mu_a_true[0], label="Case 1 True", c="tab:blue", linestyle=":")
    plt.plot(q_points, mu_a_q[:, 1], label="Case 2 Est", c="tab:orange", marker='.')
    plt.axhline(mu_a_true[1], label="Case 2 True", c="tab:orange", linestyle=":")
    plt.xticks(q_points)
    plt.legend()
    plt.figure("mu_sp")
    plt.plot(q_points, mu_sp_q[:, 0], label="Case 1 Est", c="tab:blue", marker='.')
    plt.axhline(mu_sp_true[0], label="Case 1 True", c="tab:blue", linestyle=":")
    plt.plot(q_points, mu_sp_q[:, 1], label="Case 2 Est", c="tab:orange", marker='.')
    plt.axhline(mu_sp_true[1], label="Case 2 True", c="tab:orange", linestyle=":")
    plt.xticks(q_points)
    plt.legend()

    plt.figure("mu_a error")
    plt.plot(q_points, mu_a_q[:, 0] - mu_a_true[0],
             label=r"$\mathrm{Case \; 1:} \mu_a = 0 \; \mathrm{mm^{-1}}, \; \mu_s^{\prime} = 0.05 \; \mathrm{mm^{-1}}$",
             c="tab:blue", marker='.')
    plt.plot(q_points, mu_a_q[:, 1] - mu_a_true[1],
             label=r"$\mathrm{Case \; 2:} \mu_a = 0.2 \; \mathrm{mm^{-1}}, \; \mu_s^{\prime} = 0.05 \; \mathrm{mm^{-1}}$",
             c="tab:orange", marker='.')
    plt.axhline(0, c="k", linestyle=":")
    plt.xticks(q_points)
    plt.xlabel("Number of Quadrature Points")
    plt.ylabel(r'$\mu_a \; \mathrm{Error \; (mm^{-1})}$')
    plt.legend()

    plt.figure("mu_sp error")
    plt.plot(q_points, mu_sp_q[:, 0] - mu_sp_true[0],
             label=r"$\mathrm{Case \; 1:} \mu_a = 0 \; \mathrm{mm^{-1}}, \; \mu_s^{\prime} = 0.05 \; \mathrm{mm^{-1}}$",
             c="tab:blue", marker='.')
    plt.plot(q_points, mu_sp_q[:, 1] - mu_sp_true[1],
             label=r"$\mathrm{Case \; 2:} \mu_a = 0.2 \; \mathrm{mm^{-1}}, \; \mu_s^{\prime} = 0.05 \; \mathrm{mm^{-1}}$",
             c="tab:orange", marker='.')
    plt.axhline(0, c="k", linestyle=":")
    plt.xticks(q_points)
    plt.xlabel("Number of Quadrature Points")
    plt.ylabel(r'$\mu_s^{\prime} \; \mathrm{Error \; (mm^{-1})}$')
    plt.legend()

    plt.show()

