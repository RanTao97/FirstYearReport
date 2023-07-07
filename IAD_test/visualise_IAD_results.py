# this is to visualise IAD results

import numpy as np
import matplotlib.pyplot as plt
from TXT import TXT

# set figure size and font size
plt.rcParams['figure.autolayout'] = True
plt.rcParams['axes.titlesize'] = 'x-large'
plt.rcParams['axes.labelsize'] = 'large'
plt.rcParams['xtick.labelsize'] = 'large'
plt.rcParams['ytick.labelsize'] = 'large'
plt.rcParams['legend.fontsize'] = 'large'

mu_sp_threshold = 0.2  # the threshold below which mu_a has large errors

if __name__ == '__main__':
    IAD_results = TXT("Data/IAD_test.txt", "Data/IAD_test.rxt")

    plt.figure("mu_a Correlation")
    plt.scatter(IAD_results.mu_a_true[IAD_results.mu_sp_true < mu_sp_threshold],
                IAD_results.mu_a_est[IAD_results.mu_sp_true < mu_sp_threshold],
                marker='.', label=r'$\mathrm{True} \; \mu_s^{\prime} < %.1f \; \mathrm{mm^{-1}}$' % mu_sp_threshold)
    plt.scatter(IAD_results.mu_a_true[IAD_results.mu_sp_true >= mu_sp_threshold],
                IAD_results.mu_a_est[IAD_results.mu_sp_true >= mu_sp_threshold],
                marker='.', label=r'$\mathrm{True} \; \mu_s^{\prime} \geq %.1f \; \mathrm{mm^{-1}}$' % mu_sp_threshold)
    plt.plot(IAD_results.mu_a_true, IAD_results.mu_a_true, c='k')
    plt.xlabel(r'$\mathrm{True} \; \mu_a \; \mathrm{(mm^{-1})}$')
    plt.ylabel(r'$\mathrm{IAD} \; \mu_a \; \mathrm{(mm^{-1})}$')
    plt.legend()

    plt.figure("mu_a Difference")
    plt.scatter(IAD_results.mu_a_true, IAD_results.mu_sp_true, c=(IAD_results.mu_a_est - IAD_results.mu_a_true),
                marker='.')
    plt.axhline(1, c='tab:red', linestyle=':')
    plt.axvline(0.01, c='w', linestyle=':')
    plt.xlabel(r'$\mathrm{True} \; \mu_a \; \mathrm{(mm^{-1})}$')
    plt.ylabel(r'$\mathrm{True} \; \mu_s^{\prime} \; \mathrm{(mm^{-1})}$')
    plt.colorbar()

    plt.figure("mu_a % Difference")
    mu_a_diff = (IAD_results.mu_a_est - IAD_results.mu_a_true) / (IAD_results.mu_a_true + 1e-15) * 100
    plt.scatter(IAD_results.mu_a_true[IAD_results.mu_sp_true < mu_sp_threshold],
                mu_a_diff[IAD_results.mu_sp_true < mu_sp_threshold],
                marker='.', label=r'$\mathrm{True} \; \mu_s^{\prime} < %.1f \; \mathrm{mm^{-1}}$' % mu_sp_threshold)
    plt.scatter(IAD_results.mu_a_true[IAD_results.mu_sp_true >= mu_sp_threshold],
                mu_a_diff[IAD_results.mu_sp_true >= mu_sp_threshold],
                marker='.',
                label=r'$\mathrm{True} \; \mu_s^{\prime} \geq %.1f \; \mathrm{mm^{-1}}$' % mu_sp_threshold)
    plt.axvline(0.01, c='k', linestyle=':', label=r'$\mathrm{Typical} \; \mu_a = 0.01 \; \mathrm{mm^{-1}}$')
    plt.xlabel(r'$\mathrm{True} \; \mu_a \; \mathrm{(mm^{-1})}$')
    plt.ylabel("Difference (%)")
    plt.ylim(-50, 50)
    plt.yticks(np.arange(-50, 55, 10))
    plt.legend()

    plt.figure("mu_sp Correlation")
    plt.scatter(IAD_results.mu_sp_true[IAD_results.mu_sp_true < mu_sp_threshold],
                IAD_results.mu_sp_est[IAD_results.mu_sp_true < mu_sp_threshold],
                marker='.', label=r'$\mathrm{True} \; \mu_s^{\prime} < %.1f \; \mathrm{mm^{-1}}$' % mu_sp_threshold)
    plt.scatter(IAD_results.mu_sp_true[IAD_results.mu_sp_true >= mu_sp_threshold],
                IAD_results.mu_sp_est[IAD_results.mu_sp_true >= mu_sp_threshold],
                marker='.', label=r'$\mathrm{True} \; \mu_s^{\prime} \geq %.1f \; \mathrm{mm^{-1}}$' % mu_sp_threshold)
    plt.plot(IAD_results.mu_sp_true, IAD_results.mu_sp_true, c='k')
    plt.xlabel(r'$\mathrm{True} \; \mu_s^{\prime} \; \mathrm{(mm^{-1})}$')
    plt.ylabel(r'$\mathrm{IAD} \; \mu_s^{\prime} \; \mathrm{(mm^{-1})}$')
    plt.legend()

    plt.figure("mu_sp Difference")
    plt.scatter(IAD_results.mu_a_true, IAD_results.mu_sp_true, c=(IAD_results.mu_sp_est - IAD_results.mu_sp_true),
                marker='.')
    plt.axhline(1, c='tab:red', linestyle=':')
    plt.axvline(0.01, c='w', linestyle=':')
    plt.xlabel(r'$\mathrm{True} \; \mu_a \; \mathrm{(mm^{-1})}$')
    plt.ylabel(r'$\mathrm{True} \; \mu_s^{\prime} \; \mathrm{(mm^{-1})}$')
    plt.colorbar()

    plt.figure("mu_sp % Difference")
    mu_sp_diff = (IAD_results.mu_sp_est - IAD_results.mu_sp_true) / IAD_results.mu_sp_true * 100
    plt.scatter(IAD_results.mu_sp_true[IAD_results.mu_sp_true < mu_sp_threshold],
                mu_sp_diff[IAD_results.mu_sp_true < mu_sp_threshold],
                marker='.', label=r'$\mathrm{True} \; \mu_s^{\prime} < %.1f \; \mathrm{mm^{-1}}$' % mu_sp_threshold)
    plt.scatter(IAD_results.mu_sp_true[IAD_results.mu_sp_true >= mu_sp_threshold],
                mu_sp_diff[IAD_results.mu_sp_true >= mu_sp_threshold],
                marker='.',
                label=r'$\mathrm{True} \; \mu_s^{\prime} \geq %.1f \; \mathrm{mm^{-1}}$' % mu_sp_threshold)
    plt.axvline(1, c='k', linestyle=':', label=r'$\mathrm{Typical} \; \mu_s^{\prime} = 1 \; \mathrm{mm^{-1}}$')
    plt.xlabel(r'$\mathrm{True} \; \mu_s^{\prime} \; \mathrm{(mm^{-1})}$')
    plt.ylabel("Difference (%)")
    plt.ylim(-20, 10)
    plt.legend()

    plt.show()