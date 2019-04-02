#!/usr/bin/python
from matplotlib import pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d
import numpy as np
import pymaster as nmt
import os

DEFAULT_COLOR_CYCLE = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Diagonal check
def check_Covariance_diagonal_terms(CovSims, CovTh, labelsTh, principal=True):
    f, ax = plt.subplots(2, 1, sharex=True, figsize=(4, 7))

    c = ['blue', 'orange', 'green', 'purple']
    lsth = ['--', '-.', ':']

    Ysim = []
    X = []
    for i in range(4):
        Ysim.append(np.abs(np.diag(CovSims, i)))
        X.append(range(len(Ysim[-1])))
        ax[0].plot(X[-1], Ysim[-1], c=c[i], label='Simulated' if not i else '')

        if principal:
            break

    for i_cov, covth in enumerate(CovTh):
        for i in range(4):
            Yth = np.abs(np.diag(covth, i))

            ax[0].plot(X[i], Yth, c=c[i], ls=lsth[i_cov], label=labelsTh[i_cov] if not i else '')

            ax[1].plot(X[i], Yth/Ysim[i] - 1, c=c[i], ls=lsth[i_cov], label='Diagonal {}'.format(i) if not i_cov else '')

            if principal:
                break

    ax[0].set_ylabel('abs(Diag terms covariance matrix)')
    ax[0].set_yscale('log')
    ax[0].legend(loc=0)

    ax[1].set_ylabel('Rel.dev. wrt. sim. cov.')
    ax[1].set_xlabel('Element')
    ax[1].legend(loc=0)

    plt.subplots_adjust(hspace=0)
    plt.show()
    plt.close()

# Chi2 check
def check_chi2_distributions(cls, CovSims, CovTh, hartlap=False):
    Ns, Nl = np.shape(cls)
    if hartlap:
        # arXiv:1601.05786
        factor_sim = (Ns - Nl - 2)/(Ns - 1)
    else:
        factor_sim = 1

    chi2_sim, chi2_th = get_chi2(cls, [factor_sim * np.linalg.inv(CovSims),
                                        np.linalg.inv(CovTh)])
    lmax = len(CovSims)
    plot_chi2([chi2_sim, chi2_th], ['Simulations Cov', 'Theoretical Cov'], lmax=lmax)

    print('KS between sim. and th. distributions: ', stats.ks_2samp(chi2_sim, chi2_th))
    print('KS between sim. and chi2 distributions: ', stats.kstest(chi2_sim, 'chi2', args=([lmax])))
    print('KS between th. and chi2 distributions: ', stats.kstest(chi2_th, 'chi2', args=([lmax])))
    print('KS between sampled_chi2 and chi2 distributions: ', stats.kstest(stats.chi2.rvs(lmax, size=len(chi2_sim)), 'chi2', args=([lmax])))
    return chi2_sim, chi2_th

def chi2_cl(cl, clmeans, InvCov):
    return (cl-clmeans).dot(InvCov).dot(cl - clmeans)

def get_chi2(cls, array_of_invcovs):
    chi2_list = [[] for i in range(len(array_of_invcovs))]

    cl_means = np.mean(cls, axis=0)

    for cli in cls:
        for i, InvCov in enumerate(array_of_invcovs):
            chi2_list[i].append(chi2_cl(cli, cl_means, InvCov))

    return np.array(chi2_list)

def plot_chi2(chi2s, labels, lmax, bins=30):
    _, x, _ = plt.hist(chi2s, bins=60, histtype='step', density=True, label=labels)

    plt.plot(x[:-1], stats.chi2.pdf(x[:-1], lmax), ls='--', label=r'$\chi^2$ pdf')

    plt.xlabel(r'$\chi^2$')
    plt.ylabel('pdf')

    plt.legend(loc=0)
    plt.show()
    plt.close()

# Correlation matrices difference check
def correlation_matrix(covariance):
    c = np.diag(covariance)
    return covariance / np.sqrt(c * c[:, None])

def get_correlation_from_covariance(array_of_covs):
    array_of_corrs = []
    for Cov in array_of_covs:
        array_of_corrs.append(correlation_matrix(Cov))

    return array_of_corrs

def plot_correlation_difference(lbins, CovSims, CovTh):
    CorrSims, CorrTh = get_correlation_from_covariance([CovSims, CovTh])
    plt.imshow(CorrSims - CorrTh)  #, vmin=-2, vmax=2)
    c = plt.colorbar()
    c.set_label('CorrSims - CorrTh')
    plt.xlabel('l')
    plt.ylabel('l')

    lbins = lbins.astype(int)
    ticks = np.linspace(0, len(lbins)-1, 7, dtype=int)

    plt.xticks(ticks, lbins[ticks])
    plt.yticks(ticks, lbins[ticks])

    plt.show()
    plt.close()

    return CorrSims, CorrTh

# Create CovarianceWorkspace
def generate_covariance_workspace00(run_path, fa1, fa2, flat=False):
    if flat:
        w00 = nmt.NmtWorkspaceFlat()
        cw00 = nmt.NmtCovarianceWorkspaceFlat()
    else:
        w00 = nmt.NmtWorkspace()
        cw00 = nmt.NmtCovarianceWorkspace()

    w00.read_from(run_path + "_w00.dat")

    cw00_file = run_path + "_cw00.dat"
    if not os.path.isfile(cw00_file):
        if flat:
            raise ValueError('flat not implemented yet')
            # cw00.compute_coupling_coefficients(w00, w00)
        else:
            cw00.compute_coupling_coefficients(fa1, fa2)
        cw00.write_to(cw00_file)
    else:
        cw00.read_from(cw00_file)

    return w00, cw00

# Plot reldev. eigenvalues
def plot_reldev_eigv(CovSims, CovTh, labels, yscale=['log', 'linear']):
    f, ax = plt.subplots(2, 1, sharex=True, figsize=(4, 8))

    Eval_Sims, Evec_Sims = diagonalize(CovSims)
    X = np.arange(len(Eval_Sims))
    ax[0].plot(X, Eval_Sims, label='Simulations')
    ax[1].plot(X, 0 * X)

    for i, covth in enumerate(CovTh):
        Eval_Th, Evec_Th = diagonalize(covth)

        ax[0].plot(X, Eval_Th, label=labels[i])
        ax[1].plot(X, Eval_Th/Eval_Sims - 1)

    ax[0].set_xlabel('Eigenvalue')
    ax[0].legend(loc=0)

    ax[1].set_xlabel('# dimmension')
    ax[1].set_ylabel(r'$\tilde e_{i}^{An} / \tilde e_{i}^{Sims} - 1$')

    ax[0].set_yscale(yscale[0])
    ax[1].set_yscale(yscale[1])

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    plt.close()

def diagonalize(matrix):
    Eval, Evec = np.linalg.eigh(matrix)

    idx = np.abs(Eval).argsort()[::-1]
    Eval = Eval[idx]
    Evec = Evec[idx]

    return Eval, Evec

# Plot rows of covariance matrix
def plot_rows_cov_matrix(lbins, CovSims, CovTh, normalization, labels, index=20, dx=5, peaks=4):
    c = DEFAULT_COLOR_CYCLE

    f, ax = plt.subplots(2, 1, sharex=True, figsize=(4, 8))

    X = lbins[index - dx:index + dx + 1]

    indexi = index
    Xi = X
    for _ in range(peaks):
        Xi = lbins[indexi - dx:indexi + dx + 1]
        Y = CovSims[indexi, indexi - dx:indexi + dx + 1]/normalization
        ax[0].plot(Xi, Y, c=c[0], label='Simulations' if not _ else '')
        ax[1].plot(Xi, 0 * Xi, c=c[0])
        indexi += 2 * dx

    for i, covth in enumerate(CovTh):
        indexi = index
        Xi = X
        for _ in range(peaks):
            Xi = lbins[indexi - dx:indexi + dx + 1]
            Yth = covth[indexi, indexi - dx:indexi + dx + 1]/normalization
            ax[0].plot(Xi, Yth, c=c[i+1], label=labels[i] if not _ else '')
            ax[1].plot(Xi, Yth/Y - 1, c=c[i+1])

            indexi += 2 * dx

    ax[0].legend(loc=0)

    ax[0].set_ylabel("$Cov_{ll'}$ / monopole")
    ax[1].set_ylabel("Rel. dev. wrt. sims.")
    ax[1].set_xlabel("l'")

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    plt.close()


# Naive covaricance matrix
def naive_covariance(lTh, l0, lf, fsky, cla1b1, cla1b2, cla2b1, cla2b2, unbinned=True):
    lbins=0.5*(l0+lf)
    if unbinned:
        cla1b1 = interp1d(lTh, cla1b1)(lbins)
        cla1b2 = interp1d(lTh, cla1b2)(lbins)
        cla2b1 = interp1d(lTh, cla2b1)(lbins)
        cla2b2 = interp1d(lTh, cla2b2)(lbins)

    dl=lf-l0

    return np.diag((cla1b2 * cla2b1 + cla1b1 * cla2b2) / (fsky * (2 * lbins  + 1) * dl))

# Do all checks at once:

def do_all_checks(lbins, clsims, CovSims, CovTh, modes, hartlap=False, row_cov=True):
    print('Checks for {}'.format(modes))
    print('Diagonal covariance matrix')
    check_Covariance_diagonal_terms(CovSims, [CovTh], ['Analytical'], True)
    check_Covariance_diagonal_terms(CovSims, [CovTh], ['Analytical'], False)
    print('Chi2 distribution check')
    chi2_sim, chi2_th = check_chi2_distributions(clsims, CovSims, CovTh, hartlap)
    print()
    print('Difference between analytic and sims. correlation matrix')
    CorrSims, CorrTh = plot_correlation_difference(lbins, CovSims, CovTh)
    print('Eigenvalues vs l')
    plot_reldev_eigv(CovSims, [CovTh], ['Analytical'])
    if row_cov:
        print('Row of cov. matrix.')
        plot_rows_cov_matrix(lbins, CovSims, [CovTh], np.mean(clsims,
                                                              axis=0)[0]**2,
                             ['Analytical'])

    return (chi2_sim, chi2_th), (CorrSims, CorrTh)

def do_check_covariance_terms(lbins, CovSims, CovTh, labelsTh,
                              normalization_rows, principal=False,
                              yscale_eigv=['log', 'linear'], index_rows=20,
                              dx_rows=5):
    check_Covariance_diagonal_terms(CovSims, CovTh, labelsTh, principal)
    plot_reldev_eigv(CovSims, CovTh, labelsTh, yscale_eigv)
    plot_rows_cov_matrix(lbins, CovSims, CovTh, normalization_rows, labelsTh, index_rows, dx_rows)

