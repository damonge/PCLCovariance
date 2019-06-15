from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import os

# pylint: disable=C0103

DPI = 500
DEFAULT_COLOR_CYCLE = plt.rcParams['axes.prop_cycle'].by_key()['color']

##############################################################################
############################# All rows sph ###################################
##############################################################################

prefix = 'run_sph_2b'
run_path = os.path.join('./simulations_outputs/', prefix, prefix)
sims_suffix = '_cl_0001-20000.npz'

outdir = './general-plots/'

if not os.path.isdir(outdir):
    os.mkdir(outdir)

##############################################################################

ell = np.loadtxt(run_path + '_ells.txt')
ell_cl2bin_file = np.load('./data/cls_lss_2bins.npz')
ell_th, cl2bin, nls2bin = ell_cl2bin_file['ls'], ell_cl2bin_file['cls'], ell_cl2bin_file['nls']

clth = (cl2bin + nls2bin)[:3, :3]

clth = interp1d(ell_th, clth)(ell)
i, j = np.triu_indices(3)
clth_ar = clth[i, j]

lmax = (ell < 2*512).sum()
nlbins_orig = int(ell.shape[0])
nlbins = lmax
ell = ell[:lmax]

##############################################################################

CovSims_path = run_path + '_cov_b1' + sims_suffix
Csims = np.load(CovSims_path)['arr_0']

Csims = Csims.reshape((6, nlbins_orig, 6, nlbins_orig))
Csims = Csims[:, :lmax, :, :lmax]

CsimOld = Csims.reshape(6 * nlbins, 6 * nlbins)

##############################################################################
c0000 = np.load(run_path+'_c0000_b1.npz')['arr_0']
c0002 = np.load(run_path+'_c0002_b1.npz')['arr_0']
c0022 = np.load(run_path+'_c0022_b1.npz')['arr_0']
c0202 = np.load(run_path+'_c0202_b1.npz')['arr_0']
c0222 = np.load(run_path+'_c0222_b1.npz')['arr_0']
c2222 = np.load(run_path+'_c2222_b1.npz')['arr_0']

CovTh_TT = c0000
CovTh_TTTE, CovTh_TTTB = c0002[:, 0, :, [0, 1]]
CovTh_TTEE, CovTh_TTEB, CovTh_TTBE, CovTh_TTBB = c0022[:, 0, :, [0, 1, 2, 3]]
CovTh_TETE, CovTh_TETB = c0202[:, 0, :, [0, 1] ]
CovTh_TBTE, CovTh_TBTB = c0202[:, 1, :, [0, 1] ]
CovTh_TEEE, CovTh_TEEB, CovTh_TEBE, CovTh_TEBB =  c0222[:, 0, :, [0, 1, 2, 3] ]
CovTh_TBEE, CovTh_TBEB, CovTh_TBBE, CovTh_TBBB = c0222[:, 1, :, [0, 1, 2, 3] ]
CovTh_EEEE, CovTh_EEEB, CovTh_EEBE, CovTh_EEBB = c2222[:, 0, :, [0, 1, 2, 3] ]
CovTh_EBEE, CovTh_EBEB, CovTh_EBBE, CovTh_EBBB = c2222[:, 1, :, [0, 1, 2, 3] ]
CovTh_BEEE, CovTh_BEEB, CovTh_BEBE, CovTh_BEBB = c2222[:, 2, :, [0, 1, 2, 3] ]
CovTh_BBEE, CovTh_BBEB, CovTh_BBBE, CovTh_BBBB = c2222[:, 3, :, [0, 1, 2, 3] ]

CovTh_TE = CovTh_TETE
CovTh_TB = CovTh_TBTB
CovTh_EE = CovTh_EEEE
CovTh_EB = CovTh_EBEB
CovTh_BB = CovTh_BBBB

Cth_ar = np.array([CovTh_TT, CovTh_TTTE, CovTh_TTTB, CovTh_TTEE, CovTh_TTEB, CovTh_TTBB,
                             CovTh_TETE, CovTh_TETB, CovTh_TEEE, CovTh_TEEB, CovTh_TEBB,
                                         CovTh_TBTB, CovTh_TBEE, CovTh_TBEB, CovTh_TBBB,
                                                     CovTh_EEEE, CovTh_EEEB, CovTh_EEBB,
                                                                 CovTh_EBEB, CovTh_EBBB,
                                                                             CovTh_BBBB])

Cth = np.empty((6, nlbins, 6, nlbins))
i, j = np.triu_indices(6)
Cth[i, :,  j, :] = Cth_ar[:, :lmax, :lmax]
Cth[j, :,  i, :] = Cth_ar[:, :lmax, :lmax].swapaxes(1, 2)

##############################################################################

CthN = np.empty((6, nlbins, 6, nlbins))
CthN_ar = np.load(run_path + '_covNaive_ar_b1.npz')['arr_0']
i, j = np.triu_indices(6)
CthN[i, :,  j, :] = CthN_ar[:, :lmax, :lmax]
CthN[j, :,  i, :] = CthN_ar[:, :lmax, :lmax].swapaxes(1, 2)

##############################################################################

def ax_plot_row(ax, lbins, CovSims, CovTh, CovThN, normalization, index=20, dx=5, peaks=4):
    c = DEFAULT_COLOR_CYCLE


    indexi = index
    for _ in range(peaks):
        Xi = lbins[indexi - dx:indexi + dx + 1]
        Y = CovSims[indexi, indexi - dx:indexi + dx + 1]/normalization[indexi - dx:indexi + dx + 1]
        ax.plot(Xi, Y, c=c[0], label='Simulations' if not _ else '')
        indexi += 2 * dx

    indexi = index
    for _ in range(peaks):
        Xi = lbins[indexi - dx:indexi + dx + 1]
        Y = CovSims[indexi, indexi - dx:indexi + dx + 1]/normalization[indexi - dx:indexi + dx + 1]
        Yth = CovTh[indexi, indexi - dx:indexi + dx + 1]/normalization[indexi - dx:indexi + dx + 1]
        ax.plot(Xi, Yth, c=c[1], ls='--', label='NKA' if not _ else '')
        ax.plot(Xi, Yth - Y, c=c[2], ls=':', label='diff.' if not _ else '')

        indexi += 2 * dx

    indexi = index
    Xi = lbins[indexi : indexi + (peaks * 2 * dx) : 2 * dx]
    Yth = np.diag(CovThN)[indexi : indexi + (peaks * 2 * dx) : 2 * dx]/normalization[indexi : indexi + (peaks * 2 * dx) : 2 * dx]
    ax.scatter(Xi, Yth, marker="*", s=9, c='k', label='MC', zorder=3)

    # dl = lbins[1] - lbins[0]
    # Naive = 1. / (0.4 * (2 * lbins  + 1) * dl)
    # ax.plot(lbins[index - dx : index + (peaks * 2 * dx)], Naive[index - dx : index + (peaks * 2 * dx)])


##############################################################################

f, ax = plt.subplots(2, 3, figsize=(8, 4), sharex=True, sharey='row',
                     gridspec_kw={'wspace': 0, 'hspace': 0})

c = 0
labels = [r'$\delta \delta$', r'$\delta \gamma_{E}$', r'$\delta \gamma_{B}$',
                              r'$\gamma_E \gamma_{E}$', r'$\gamma_E \gamma_B$',
                                                        r'$\gamma_B \gamma_B$']


index_cl = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
index_plot = [0, 1, 3, 2, 4, 5]

l0 = ell - 1.5
lf = ell + 1.5


ax_ar = ax.reshape((6))
for i in range(6):
    cl_aa_i, cl_bb_i = index_cl[i]
    ax_plot_row(ax_ar[index_plot[i]], ell, Csims[i, :, i, :], Cth[i, :, i, :], CthN[i, :, i, :],
                normalization=(clth[cl_aa_i, cl_aa_i] * clth[cl_bb_i, cl_bb_i] + clth_ar[i]**2))

    ax_ar[index_plot[i]].text(0.78, 0.9, "({}, {})".format(labels[i], labels[i]),
          transform=ax_ar[index_plot[i]].transAxes,
          fontsize=9, horizontalalignment='center')


for i in range(2):
    ax[i, 0].set_ylabel("${\\rm Cov}_{\ell\ell'}\\,{\\rm (normalised)}$")

for i in range(3):
    ax[1, i].set_xlabel("$\ell'$")

ax[1, 1].legend(fontsize=9, loc='lower center', frameon=True, ncol=2)
plt.tight_layout()

fname = os.path.join(outdir, 'all_rows_sph_1bin.pdf')
plt.savefig(fname, dpi=DPI)
# plt.show()
plt.close()

##############################################################################

Csims = CsimOld
CorrSims = Csims/np.sqrt(np.diag(Csims)[:, None] * np.diag(Csims)[None, :])

Cth = Cth.reshape((6 * nlbins, 6 * nlbins))

CorrTh = Cth/np.sqrt(np.diag(Cth)[:, None] * np.diag(Cth)[None, :])

f, ax = plt.subplots(1, 1, figsize=(4, 4))

cb = ax.imshow(CorrTh - CorrSims, vmin=-0.02, vmax=0.02)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = f.colorbar(cb, cax=cax)
cbar.ax.tick_params(labelsize=8)
for i in range(1, 6):
    ax.plot([0, 6*nlbins-1], [i*nlbins, i*nlbins], 'k-', lw=0.5)
    ax.plot([i*nlbins, i*nlbins], [0, 6*nlbins-1], 'k-', lw=0.5)

ticks = [(i + 0.5) * nlbins for i in range(6)]
ax.set_xticks(ticks)
ax.set_xticklabels(labels, fontsize=8)
ax.set_yticks(ticks)
ax.set_yticklabels(labels, fontsize=8)

plt.tight_layout()
fname = os.path.join(outdir, 'sph_1bin_diff_corr.pdf')
plt.savefig(fname, dpi=DPI)
# plt.show()
plt.close()
