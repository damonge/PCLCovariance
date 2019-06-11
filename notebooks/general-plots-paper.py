#!/usr/bin/python
from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
import common as co
import flatmaps as fm
import healpy as hp
import numpy as np
import templates as tp
import matplotlib.pyplot as plt
import os

# pylint: disable=C0103

DPI = 500
FSIZE1 = (4, 3)
DEFAULT_COLOR_CYCLE = plt.rcParams['axes.prop_cycle'].by_key()['color']

outdir = './general-plots/'

if not os.path.isdir(outdir):
    os.mkdir(outdir)

##############################################################################
######################### Fig with z-bins ####################################
##############################################################################
def pz():
    fname = os.path.join(outdir, 'pz.pdf')

    nell = 30000

    def nofz(z, z0, sz, ndens):
        return np.exp(-0.5*((z-z0)/sz)**2)*ndens/np.sqrt(2*np.pi*sz**2)

    z = np.linspace(0, 2, 512)
    pz1 = nofz(z, 0.955, 0.13, 7.55)
    pz2 = nofz(z, 0.755, 0.13, 7.55)
    ndens1 = np.sum(pz1)*np.mean(z[1:]-z[:-1])*(180*60./np.pi)**2
    ndens2 = np.sum(pz2)*np.mean(z[1:]-z[:-1])*(180*60./np.pi)**2

    f, ax = plt.subplots(1, 1, figsize=FSIZE1)
    ax.plot(z, pz1, label='Bin 1')
    ax.plot(z, pz2, label='Bin 2')

    ax.set_ylabel(r'$P(z)$ [arcmin$^{-2}$]')
    ax.set_xlabel(r'$z$')

    ax.legend(loc=0, frameon=False)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(fname, dpi=DPI)
    plt.close()

##############################################################################
######################### Theoretical sph 2b Cls #############################
##############################################################################

def cls_2b():
    fname = os.path.join(outdir, 'cls-sph-2b.pdf')


    ell_cl2bin_file = np.load('./data/cls_lss_2bins.npz')
    ell, cl2bin, nls2bin = ell_cl2bin_file['ls'], ell_cl2bin_file['cls'], ell_cl2bin_file['nls']

    f, axs = plt.subplots(4, 4, figsize=(9, 6), gridspec_kw={'hspace': 0, 'wspace': 0}, sharex=True, sharey='row')

    labels = [r'$\delta_1$', r'$\gamma_{E,1}$', r'$\delta_2$', r'$\gamma_{E,2}$']
    ci = 0
    for i in range(cl2bin.shape[0]):
        if i in (2, 5):
            continue
        cj = ci
        for j in range(i, cl2bin.shape[0]):
            if j in (2, 5):
                continue
            axs[ci, cj].loglog(ell[ell >= 2], cl2bin[i, j, ell >= 2], label='Signal')
            # axs[ci, cj].loglog(ell[ell >= 2], cl2bin[i, j, ell >= 2] + nls2bin[i, j, ell >= 2], label='Signal + Noise', ls='--')
            axs[ci, cj].loglog(ell[ell >= 2], nls2bin[i, j, ell >= 2], label='Noise')
            axs[ci, cj].text(0.78, 0.9, "({}, {})".format(labels[ci], labels[cj]),
                             transform=axs[ci, cj].transAxes,
                             fontsize=9, horizontalalignment='center')
            cj += 1

        ci += 1

    axs[0, 0].legend(loc=0, frameon=False)

    i, j = np.triu_indices(axs.shape[0])
    for ax in axs[j, i].reshape((-1)):
        ax.set_visible(False)

    for i in range(axs.shape[0]):
        axs[i, i].set_xlabel('$\ell$')
        axs[i, i].set_ylabel(r'$C_\ell$')
        axs[i, i].set_visible(True)
        axs[i, i].xaxis.set_tick_params(labelbottom=True, labelsize=8)
        axs[i, i].yaxis.set_tick_params(labelleft=True, labelsize=8,
                                        labelrotation=45, pad=-3)
        # axs[0, i].set_title(labels[i])
        # axs[i, -1].text(1, 0.5, labels[i], fontsize=12,
        #                 transform=axs[i, -1].transAxes)


    plt.tight_layout()
    plt.savefig(fname, dpi=DPI)
    plt.close()

##############################################################################
########################## Compare 1bin sph Cls ##############################
##############################################################################

def compare_cls_1bin():
    fname = os.path.join(outdir, 'compare-sph-cls-used-1bin.pdf')

    ell_cl2bin_file = np.load('./data/cls_lss_2bins.npz')
    ell, cl2bin, nls2bin = ell_cl2bin_file['ls'], ell_cl2bin_file['cls'], ell_cl2bin_file['nls']

    clTh = np.loadtxt('./data/cls_lss.txt', unpack=True)
    lTh = clTh[0][:3*512]
    clTh_TT = (clTh[1] + clTh[5])[:3*512]
    clTh_EE = (clTh[2] + clTh[6])[:3*512]
    clTh_TE = (clTh[4] + clTh[-1])[:3*512]

    f, axs = plt.subplots(1, 3, figsize=(8, 3), gridspec_kw={'hspace': 0, 'wspace': 0}, sharex=True, sharey='row')

    cl2bins_1 = (cl2bin + nls2bin)[:3, :3]
    cl2bins_2 = (cl2bin + nls2bin)[3:, 3:]
    i, j = np.triu_indices(3)

    #### TT #####
    axs[0].loglog(lTh, clTh_TT, label='1-bin Fid.')
    axs[0].loglog(ell, cl2bins_1[0, 0], label='1st 2-bin Fid.', ls='--')
    axs[0].loglog(ell, cl2bins_2[0, 0], label='2nd 2-bin Fid.', ls='--')
    axs[0].set_ylabel(r'$C_\ell$')

    axs[0].text(0.5, 0.9, r'$C_\ell^{gg}$', transform=axs[0].transAxes)

    #### TE #####
    axs[1].loglog(lTh, clTh_TE, label='1-bin Fid.')
    axs[1].loglog(ell, cl2bins_1[0, 1], label='1st 2-bin Fid.', ls='--')
    axs[1].loglog(ell, cl2bins_2[0, 1], label='2nd 2-bin Fid.', ls='--')
    axs[1].text(0.5, 0.9, r'$C_\ell^{g\gamma}$', transform=axs[1].transAxes)
    #axs[1].set_ylabel(r'$C_\ell^{g\gamma}$')

    #### EE #####
    axs[2].loglog(lTh, clTh_EE, label='1-bin Fid.')
    axs[2].loglog(ell, cl2bins_1[1, 1], label='1st 2-bin Fid.', ls='--')
    axs[2].loglog(ell, cl2bins_2[1, 1], label='2nd 2-bin Fid.', ls='--')
    axs[2].text(0.5, 0.9, r'$C_\ell^{\gamma\gamma}$', transform=axs[2].transAxes)
    #axs[2].set_ylabel(r'$C_\ell^{\gamma\gamma}$')

    axs[0].legend(loc=0, frameon=False)

    for i in range(axs.shape[0]):
        axs[i].set_xlabel('$\ell$')

    plt.tight_layout()
    plt.savefig(fname, dpi=DPI)
    plt.close()

##############################################################################
############################# Mask sph1 ######################################
##############################################################################

def masks():
    fname = os.path.join(outdir, 'mask-lss1.pdf')

    fmask = "./data/mask_lss_sph1.fits"
    mask_lss = hp.ud_grade(hp.read_map(fmask, verbose=False), nside_out=512)
    mask_lss[mask_lss == 0] = hp.UNSEEN

    hp.mollview(mask_lss, title="", cbar=False, coord=['G', 'C'], notext=True)
    plt.savefig(fname, dpi=DPI)
    plt.close()

##############################################################################
############################# Mask sph1 ######################################
##############################################################################

    fname = os.path.join(outdir, 'mask-lss2.pdf')

    fmask = "./data/mask_lss_sph2.fits"
    mask_lss = hp.ud_grade(hp.read_map(fmask, verbose=False), nside_out=512)
    mask_lss[mask_lss == 0] = hp.UNSEEN

    hp.mollview(mask_lss, title="", coord=['G', 'C'], cbar=False, notext=True)
    plt.savefig(fname, dpi=DPI)
    plt.close()

##############################################################################
############################# Mask flat1 ######################################
##############################################################################

    fname = os.path.join(outdir, 'mask-lss_flat1.pdf')

    fmask = "./data/mask_lss_flat.fits"

    fmi, mask_hsc = fm.read_flat_map(fmask)
    mask_hsc[mask_hsc == 0] = hp.UNSEEN

    fmi.view_map(mask_hsc, addColorbar=False)

    plt.savefig(fname, dpi=DPI, bbox_iches='tight')
    plt.close()

##############################################################################
############################# Mask flat2 ######################################
##############################################################################

    # fname = os.path.join(outdir, 'mask-lss_flat2.pdf')

    # fmask = "./data/mask_lss_flat_2.fits"

    # fmi, mask_hsc = fm.read_flat_map(fmask)
    # mask_hsc[mask_hsc == 0] = hp.UNSEEN

    # fmi.view_map(mask_hsc)
    # plt.savefig(fname, dpi=DPI)
    # plt.close()

##############################################################################
############################# Foregrounds ####################################
##############################################################################

def foregrounds():
    ell_cl2bin_file = np.load('./data/cls_lss_2bins.npz')
    ell, cl2bin, nls2bin = ell_cl2bin_file['ls'], ell_cl2bin_file['cls'], ell_cl2bin_file['nls']
    cl_ss = tp.create_cl_templates(ell, cl2bin[0, 0] + nls2bin[0, 0], exp_range=(0, 0), N=1)
    cl_ls1 = tp.create_cl_templates(ell, cl2bin[0, 0] + nls2bin[0, 0], exp_range=(-3, -3), N=1)[0]
    cl_ls2 = tp.create_cl_templates(ell, cl2bin[0, 0] + nls2bin[0, 0], exp_range=(-1, -1), N=1)[0]

    fname = os.path.join(outdir, 'contaminants_cl.pdf')

    f, ax = plt.subplots(1, 1, figsize=FSIZE1)
    color = DEFAULT_COLOR_CYCLE

    sel = (ell >= 2) * (ell <= 1e4)

    ax.loglog(ell[sel], (cl2bin[0, 0] + nls2bin[0, 0])[sel], label='Signal')
    ax.loglog(ell[sel], cl_ls1[sel], c=color[1])
    ax.loglog(ell[sel], cl_ls2[sel], c=color[1], label='Large-scale cont.')
    ax.fill_between(ell[sel], cl_ls1[sel], cl_ls2[sel], facecolor=color[1], alpha=0.5)

    ax.loglog(ell[sel], cl_ss[0][sel], c=color[2], label='Small-scale cont.')

    ax.set_xlabel('$\ell$')
    ax.set_ylabel(r'$C_\ell^{\delta \delta}$')

    ax.legend(loc=0, fontsize=8, frameon=False)

    ax.set_ylim([1e-9, 1])

    plt.tight_layout()
    plt.savefig(fname, dpi=DPI)
    plt.close()

##############################################################################
############################ Chi2 foregrounds ################################
##############################################################################

def chi2_foregrounds():
    CovSims = np.load('../run_sph/run_sph_covTTTEE_short_clsims_0001-20000.npz')['arr_0']
    CovSims_fore = np.load('../run_sph_contaminants_ls100_ss100/run_sph_contaminants_ls100_ss100_covTTTEE_short_clsims_0001-20000.npz')['arr_0']
    CovTh = np.load('../run_sph_contaminants_ls100_ss100/run_sph_contaminants_ls100_ss100_covThTTTEEE_short.npz')['arr_0']

    clsims = np.load('../run_sph_contaminants_ls100_ss100/run_sph_contaminants_ls100_ss100_clsims_0001-20000.npz')
    clTT = np.reshape(clsims['cl00'], np.array(np.shape(clsims['cl00']))[[0, 2]])
    clTE = clsims['cl02'][:, 0, :]
    clEE = clsims['cl22'][:, 0, :]
    lbins = clsims['l']
    lmax = (clsims['l'] < 2*512).sum()

    clTTTEEE = np.concatenate([clTT[:, :lmax], clTE[:, :lmax], clEE[:, :lmax]], axis=1)

    chi2_list = co.get_chi2(clTTTEEE, [np.linalg.inv(CovSims), np.linalg.inv(CovSims_fore), np.linalg.inv(CovTh)])

    f, ax = plt.subplots(1, 1, figsize=FSIZE1)
    bins = np.linspace(np.min(chi2_list), np.max(chi2_list), 60)
    _, x, _ = ax.hist(chi2_list[0], bins=bins, histtype='step', density=True,
                      label='Sim.', ls='-')
    _, x, _ = ax.hist(chi2_list[1], bins=bins, histtype='step', density=True,
                      label='Sim. w/ cont.', ls='-.')
    _, x, _ = ax.hist(chi2_list[2], bins=bins, histtype='step', density=True,
                      label='NKA', ls='-')

    ax.plot(x[:-1], stats.chi2.pdf(x[:-1], clTTTEEE.shape[1]), ls='--', label=r'$\chi^2$ pdf')

    ax.set_xlabel(r'$\chi^2$')
    ax.set_ylabel('$10^3$ pdf')

    y_vals = ax.get_yticks()
    ax.set_yticklabels([str(x * 1000) for x in y_vals])

    ax.legend(loc='upper right', fontsize='8', frameon=False)
    plt.tight_layout()
    fname = os.path.join(outdir, 'contaminants_chi2.pdf')
    plt.savefig(fname, dpi=DPI)
    # plt.show()
    plt.close()

##############################################################################
############################# 2bins corr diff ################################
##############################################################################

def corr_diff_2bins():
    Cth = np.load('../run_sph_2b_same_mask/full_covariance/run_sph_2b_same_mask_covTh_TTTEEE_short_2bins_same_mask.npz')['arr_0']
    Csims = np.load('../run_sph_2b_same_mask/full_covariance/run_sph_2b_same_mask_covSims_TTTEEE_short_0001-20000.npz')['arr_0']

    ell = np.loadtxt('../run_sph_2b_same_mask/run_sph_2b_same_mask_ells.txt')

    CorrSims = Csims/np.sqrt(np.diag(Csims)[:, None] * np.diag(Csims)[None, :])
    CorrTh = Cth/np.sqrt(np.diag(Cth)[:, None] * np.diag(Cth)[None, :])


    labels = [r'$\delta_1$', r'$\gamma_{E,1}$', r'$\delta_2$', r'$\gamma_{E,2}$']
    labels_cov = []
    for i in range(4):
        for j in range(i, 4):
            labels_cov.append(labels[i] + labels[j])


    f, ax = plt.subplots(1, 1, figsize=(8, 8))
    nlbins = int(CorrSims.shape[0] / 10)
    cb = ax.imshow(CorrTh - CorrSims, vmin=-0.02, vmax=0.02)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    f.colorbar(cb, cax=cax)
    for i in range(1, 10):
        ax.plot([0, 10*nlbins-1], [i*nlbins, i*nlbins], 'k-', lw=0.5)
        ax.plot([i*nlbins, i*nlbins], [0, 10*nlbins-1], 'k-', lw=0.5)

    ticks = [(i + 0.5) * nlbins for i in range(10)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels_cov)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels_cov)

    fname = os.path.join(outdir, 'run_sph_2b_same_mask_NKA_diff_corr.pdf')
    plt.savefig(fname, dpi=DPI)
    # plt.show()
    plt.close()

    f, ax = plt.subplots(1, 1, figsize=(4, 4))
    mat = (CorrTh - CorrSims).reshape((10, nlbins, 10, nlbins))
    ixgrid = np.ix_([2, 3, 5, 6], [2, 3, 5, 6])
    mat = mat.swapaxes(1, 2)[ixgrid].swapaxes(1, 2).reshape((4 * nlbins, 4 * nlbins))
    cb = ax.imshow(mat, vmin=-0.02, vmax=0.02)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = f.colorbar(cb, cax=cax)
    cbar.ax.tick_params(labelsize=8)
    for i in range(1, 4):
        ax.plot([0, 4*nlbins-1], [i*nlbins, i*nlbins], 'k-', lw=0.5)
        ax.plot([i*nlbins, i*nlbins], [0, 4*nlbins-1], 'k-', lw=0.5)

    ticks = [(i + 0.5) * nlbins for i in range(4)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(np.array(labels_cov)[[2, 3, 5, 6]], fontsize=8)
    ax.set_yticks(ticks)
    ax.set_yticklabels(np.array(labels_cov)[[2, 3, 5, 6]], fontsize=8)

    plt.tight_layout()
    fname = os.path.join(outdir, 'run_sph_2b_same_mask_NKA_diff_corr_short.pdf')
    plt.savefig(fname, dpi=DPI)
    # plt.show()
    plt.close()


##############################################################################
####################### TTTT, TETE, EEEE chi2 plot ###########################
##############################################################################

def chi2_sph_TT_TE_EE():
    prefix = 'run_sph_2b'
    run_path = os.path.join('./simulations_outputs/', prefix, prefix)
    sims_suffix = '_cl_0001-20000.npz'

    ##############################################################################

    ell = np.loadtxt(run_path + '_ells.txt')
    lmax = (ell < 2*512).sum()
    nlbins = lmax

    clsims = np.load(run_path + '_b1' + sims_suffix)['arr_0']

    clTT = clsims[0, 0, :, :lmax]
    clEE = clsims[1, 1, :, :lmax]
    clTE = clsims[0, 1, :, :lmax]

    ##############################################################################

    CovSims_path = run_path + '_cov_b1' + sims_suffix
    Csims = np.load(CovSims_path)['arr_0']

    nlbins_orig = int(Csims.shape[0] / 6)
    Csims = Csims.reshape((6, nlbins_orig, 6, nlbins_orig))
    Csims = Csims[:, :lmax, :, :lmax]

    Csims_TT = Csims[0, :, 0, :]
    Csims_TE = Csims[1, :, 1, :]
    Csims_EE = Csims[3, :, 3, :]

    ##############################################################################
    c0000 = np.load(run_path+'_c0000_b1.npz')['arr_0']
    c0202 = np.load(run_path+'_c0202_b1.npz')['arr_0']
    c2222 = np.load(run_path+'_c2222_b1.npz')['arr_0']

    CovTh_TT = c0000[:lmax, :lmax]
    CovTh_TETE, CovTh_TETB = c0202[:, 0, :, [0, 1] ]
    CovTh_EEEE, CovTh_EEEB, CovTh_EEBE, CovTh_EEBB = c2222[:, 0, :, [0, 1, 2, 3] ]

    CovTh_TE = CovTh_TETE[:lmax, :lmax]
    CovTh_EE = CovTh_EEEE[:lmax, :lmax]

    ##############################################################################

    Cth0 = np.empty((6, nlbins, 6, nlbins))
    Cth0_ar = np.load(run_path + '_covSpin0_ar_b1.npz')['arr_0']
    i, j = np.triu_indices(6)
    Cth0[i, :,  j, :] = Cth0_ar[:, :lmax, :lmax]
    Cth0[j, :,  i, :] = Cth0_ar[:, :lmax, :lmax]

    Cth0_TT = Cth0[0, :, 0, :]
    Cth0_TE = Cth0[1, :, 1, :]
    Cth0_EE = Cth0[3, :, 3, :]

    ##############################################################################

    chi2_TT_ar = co.get_chi2(clTT, list(map(np.linalg.inv, [Csims_TT, CovTh_TT, Cth0_TT])))
    chi2_EE_ar = co.get_chi2(clEE, list(map(np.linalg.inv, [Csims_EE, CovTh_EE, Cth0_EE])))
    chi2_TE_ar = co.get_chi2(clTE, list(map(np.linalg.inv, [Csims_TE, CovTh_TE, Cth0_TE])))

    f, axs = plt.subplots(1, 3, figsize=(8, 3), sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0})

    label = [r"$(\delta \delta, \delta \delta)$", r"$(\delta \gamma_E, \delta \gamma_E)$",
             r"$(\gamma_E \gamma_E, \gamma_E \gamma_E)$"]
    i = 0
    for ax, chi2_list in zip(axs, [chi2_TT_ar, chi2_TE_ar, chi2_EE_ar]):
        bins = np.linspace(np.min(chi2_list), np.max(chi2_list), 60)
        _, x, _ = ax.hist(chi2_list[0], bins=bins, histtype='step', density=True,
                          label='Sims.')
        _, x, _ = ax.hist(chi2_list[1], bins=bins, histtype='step', density=True,
                          label='NKA')
        _, x, _ = ax.hist(chi2_list[2], bins=bins, histtype='step', density=True,
                          label='Spin-0')

        ax.plot(x[:-1], stats.chi2.pdf(x[:-1], lmax), ls='--', label=r'$\chi^2$ pdf')

        ax.text(0.8, 0.9, label[i], horizontalalignment='center',
                transform=ax.transAxes)
        ax.set_xlabel(r'$\chi^2$')

        i += 1

    axs[0].set_ylabel('pdf')

    # axs[0].legend(loc='center right', fontsize='9') # , frameon=False)
    axs[0].legend(loc='center right', fontsize='9', frameon=False)
    plt.tight_layout()
    fname = os.path.join(outdir, 'run_sph_2b_1stbin_chi2_TT_TE_EE.pdf')
    plt.savefig(fname, dpi=DPI)
    # plt.show()
    plt.close()

##############################################################################
##################### TTTT, TETE, EEEE chi2 plot flat ########################
##############################################################################

def chi2_flat_TT_TE_EE_TB_EB_BB():
    prefix = 'run'
    run_path = os.path.join('./simulations_outputs/', prefix, prefix)
    sims_suffix = '_clsims_0001-20000.npz'

    ##############################################################################
    clsims = np.load(run_path + sims_suffix)

    clTT = cl00 = np.reshape(clsims['cl00'], np.array(np.shape(clsims['cl00']))[[0, 2]])
    clTE = clsims['cl02'][:, 0, :]
    clTB = clsims['cl02'][:, 1, :]
    clEE = clsims['cl22'][:, 0, :]
    clEB = clsims['cl22'][:, 1, :]
    clBB = clsims['cl22'][:, 3, :]

    lmax = nlbins = len(clTT[0])
    lbins = clsims['l']

    ##############################################################################

    CovSims_path = run_path + '_cov' + sims_suffix
    Csims = np.load(CovSims_path)['arr_0']

    nlbins_orig = int(Csims.shape[0] / 6)
    Csims = Csims.reshape((6, nlbins_orig, 6, nlbins_orig))
    Csims = Csims[:, :lmax, :, :lmax]

    Csims_TT = Csims[0, :, 0, :]
    Csims_TE = Csims[1, :, 1, :]
    Csims_TB = Csims[2, :, 2, :]
    Csims_EE = Csims[3, :, 3, :]
    Csims_EB = Csims[4, :, 4, :]
    Csims_BB = Csims[5, :, 5, :]

    ##############################################################################
    c0000 = np.load(run_path+'_c0000.npz')['arr_0']
    c0202 = np.load(run_path+'_c0202.npz')['arr_0']
    c2222 = np.load(run_path+'_c2222.npz')['arr_0']

    CovTh_TT = c0000[:lmax, :lmax]
    CovTh_TETE, CovTh_TETB = c0202[:, 0, :, [0, 1] ]
    CovTh_EEEE, CovTh_EEEB, CovTh_EEBE, CovTh_EEBB = c2222[:, 0, :, [0, 1, 2, 3] ]
    CovTh_TBTE, CovTh_TBTB = c0202[:, 1, :, [0, 1] ]
    CovTh_EBEE, CovTh_EBEB, CovTh_EBBE, CovTh_EBBB = c2222[:, 1, :, [0, 1, 2, 3] ]
    CovTh_BBEE, CovTh_BBEB, CovTh_BBBE, CovTh_BBBB = c2222[:, 3, :, [0, 1, 2, 3] ]


    CovTh_TE = CovTh_TETE[:lmax, :lmax]
    CovTh_EE = CovTh_EEEE[:lmax, :lmax]

    CovTh_TB = CovTh_TBTB[:lmax, :lmax]
    CovTh_EB = CovTh_EBEB[:lmax, :lmax]
    CovTh_BB = CovTh_BBBB[:lmax, :lmax]

    ##############################################################################

    Cth0 = np.empty((6, nlbins, 6, nlbins))
    Cth0_ar = np.load(run_path + '_covSpin0_ar.npz')['arr_0']
    i, j = np.triu_indices(6)
    Cth0[i, :,  j, :] = Cth0_ar[:, :lmax, :lmax]
    Cth0[j, :,  i, :] = Cth0_ar[:, :lmax, :lmax]

    Cth0_TT = Cth0[0, :, 0, :]
    Cth0_TE = Cth0[1, :, 1, :]
    Cth0_TB = Cth0[2, :, 2, :]
    Cth0_EE = Cth0[3, :, 3, :]
    Cth0_EB = Cth0[4, :, 4, :]
    Cth0_BB = Cth0[5, :, 5, :]


    ##############################################################################

    chi2_TT_ar = co.get_chi2(clTT, list(map(np.linalg.inv, [Csims_TT, CovTh_TT, Cth0_TT])))
    chi2_EE_ar = co.get_chi2(clEE, list(map(np.linalg.inv, [Csims_EE, CovTh_EE, Cth0_EE])))
    chi2_TE_ar = co.get_chi2(clTE, list(map(np.linalg.inv, [Csims_TE, CovTh_TE, Cth0_TE])))
    chi2_TB_ar = co.get_chi2(clTB, list(map(np.linalg.inv, [Csims_TB, CovTh_TB, Cth0_TB])))
    chi2_EB_ar = co.get_chi2(clEB, list(map(np.linalg.inv, [Csims_EB, CovTh_EB, Cth0_EB])))
    chi2_BB_ar = co.get_chi2(clBB, list(map(np.linalg.inv, [Csims_BB, CovTh_BB, Cth0_BB])))

    f, axs = plt.subplots(2, 3, figsize=(8, 4), sharex=True, sharey=True,
                          gridspec_kw={'hspace': 0, 'wspace': 0})

    label = [r"$(\delta \delta, \delta \delta)$", r"$(\delta \gamma_E, \delta \gamma_E)$",
             r"$(\gamma_E \gamma_E, \gamma_E \gamma_E)$", r"$(\delta \gamma_B, \delta \gamma_B)$",
             r"$(\gamma_E \gamma_B, \gamma_E \gamma_B)$", r"$(\gamma_B \gamma_B, \gamma_B \gamma_B)$"]
    i = 0
    for ax, chi2_list in zip(axs.reshape((6)), [chi2_TT_ar, chi2_TE_ar, chi2_EE_ar, chi2_TB_ar, chi2_EB_ar, chi2_BB_ar]):
        bins = np.linspace(np.min(chi2_list), np.max(chi2_list), 60)
        _, x, _ = ax.hist(chi2_list[0], bins=bins, histtype='step', density=True,
                          label='Sims.')
        _, x, _ = ax.hist(chi2_list[1], bins=bins, histtype='step', density=True,
                          label='NKA')
        _, x, _ = ax.hist(chi2_list[2], bins=bins, histtype='step', density=True,
                          label='Spin-0')

        ax.plot(x[:-1], stats.chi2.pdf(x[:-1], lmax), ls='--', label=r'$\chi^2$ pdf')

        ax.text(0.8, 0.9, label[i], horizontalalignment='center',
                transform=ax.transAxes)
        ax.set_xlabel(r'$\chi^2$')

        i += 1

    axs[0, 0].set_ylabel('pdf')

    # axs[0].legend(loc='center right', fontsize='9') # , frameon=False)
    axs[0, 0].legend(loc='center right', fontsize='8', frameon=False)
    plt.tight_layout()
    fname = os.path.join(outdir, 'run_chi2_TT_TE_EE_TB_EB_BB.pdf')
    plt.savefig(fname, dpi=DPI)
    # plt.show()
    plt.close()

##############################################################################
############################ Chi2 TTTEEE Full ################################
##############################################################################

def chi2_NKA_TTTEEE_full():
    chi2_list = np.load('../run_sph_2b_same_mask/full_covariance/run_sph_2b_same_mask_chi2_sims_th_TTTEEE_short_Full.npz')['arr_0']

    ell = np.loadtxt('../run_sph_2b_same_mask/run_sph_2b_same_mask_ells.txt')
    lbins = ell
    lmax = (ell < 2*512).sum()

    f, ax = plt.subplots(1, 1, figsize=FSIZE1)
    bins = np.linspace(np.min(chi2_list), np.max(chi2_list), 60)
    _, x, _ = ax.hist(chi2_list[0], bins=bins, histtype='step', density=True,
                      label='Sim.', ls='-')
    _, x, _ = ax.hist(chi2_list[1], bins=bins, histtype='step', density=True,
                      label='NKA', ls='-')

    ax.plot(x[:-1], stats.chi2.pdf(x[:-1], lmax * 10), ls='--', label=r'$\chi^2$ pdf')

    ax.set_xlabel(r'$\chi^2$')
    ax.set_ylabel('$10^3$ pdf')

    y_vals = ax.get_yticks()
    ax.set_yticklabels([str(x * 1000) for x in y_vals])

    ax.legend(loc='upper right', fontsize='9', frameon=False)
    plt.tight_layout()
    fname = os.path.join(outdir, 'run_sph_2b_same_mask_NKA_TTTEEE_Full_chi2.pdf')
    plt.savefig(fname, dpi=DPI)
    # plt.show()
    plt.close()

def chi2_Spin0_NKA_TTTEEE_full():
    chi2_list = np.load('../run_sph_2b_same_mask/full_covariance/run_sph_2b_same_mask_chi2_sims_th_TTTEEE_short_Full.npz')['arr_0']
    chi2_spin0 = np.load('../run_sph_2b_same_mask/full_covariance_spin0/run_sph_2b_same_mask_chi2_sims_th_TTTEEE_short_Full_spin0.npz')['arr_0'][1]

    ell = np.loadtxt('../run_sph_2b_same_mask/run_sph_2b_same_mask_ells.txt')
    lbins = ell
    lmax = (ell < 2*512).sum()

    f, ax = plt.subplots(1, 1, figsize=FSIZE1)
    bins = np.linspace(np.min(chi2_list), np.max(chi2_list), 60)
    _, x, _ = ax.hist(chi2_list[0], bins=bins, histtype='step', density=True,
                      label='Sim.', ls='-')
    _, x, _ = ax.hist(chi2_list[1], bins=bins, histtype='step', density=True,
                      label='NKA', ls='-')
    _, x, _ = ax.hist(chi2_spin0, bins=bins, histtype='step', density=True,
                      label='Spin-0', ls='-')

    ax.plot(x[:-1], stats.chi2.pdf(x[:-1], lmax * 10), ls='--', label=r'$\chi^2$ pdf')

    ax.set_xlabel(r'$\chi^2$')
    ax.set_ylabel('$10^3$ pdf')

    y_vals = ax.get_yticks()
    ax.set_yticklabels([str(x * 1000) for x in y_vals])

    ax.legend(loc='upper right', fontsize='9', frameon=False)
    plt.tight_layout()
    fname = os.path.join(outdir, 'run_sph_2b_same_mask_Spin0_NKA_TTTEEE_Full_chi2.pdf')
    plt.savefig(fname, dpi=DPI)
    # plt.show()
    plt.close()

##############################################################################
############################## What to plot ##################################
##############################################################################

def eigv_sph_1bin():
    CovSims = np.load('../run_sph_2b/run_sph_2b_covTTTEE_b1_short_cl_0001-20000.npz')['arr_0']
    CovTh = np.load('../run_sph_2b/run_sph_2b_covThTTTEEE_b1_short.npz')['arr_0']

    f, ax = plt.subplots(2, 1, sharex=True, figsize=(4,4))

    Eval_Sims, Evec_Sims = co.diagonalize(CovSims)
    X = np.arange(len(Eval_Sims))
    ax[0].plot(X, Eval_Sims, label='Sim.')
    ax[1].plot(X, 0 * X)

    Eval_Th, Evec_Th = co.diagonalize(CovTh)

    ax[0].plot(X, Eval_Th, label='NKA')
    ax[1].plot(X, Eval_Th/Eval_Sims - 1)

    ax[0].set_ylabel('Eigenvalue')
    ax[0].legend(loc=0, frameon=False)

    ax[1].set_xlabel('Element')
    ax[1].set_ylabel(r'$\tilde e_{i}^{NKA} / \tilde e_{i}^{Sim.} - 1$')

    ax[0].set_yscale('log')
    ax[1].set_yscale('linear')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    fname = os.path.join(outdir, 'run_sph_2b_NKA_TTTEEE_reldev_eigval_1stbin.pdf')
    plt.savefig(fname, dpi=DPI)
    plt.show()
    plt.close()

if __name__ == '__main__':
    pz()
    masks()
    foregrounds()
    chi2_foregrounds()
    corr_diff_2bins()
    # compare_cls_1bin()
    cls_2b()
    chi2_sph_TT_TE_EE()
    chi2_flat_TT_TE_EE_TB_EB_BB()
    chi2_NKA_TTTEEE_full()
    eigv_sph_1bin()
    chi2_Spin0_NKA_TTTEEE_full()

