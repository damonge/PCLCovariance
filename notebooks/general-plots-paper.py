#!/usr/bin/python
import healpy as hp
import numpy as np
import templates as tp
import matplotlib.pyplot as plt
import os

# pylint: disable=C0103

DPI = 500
FSIZE1 = (4, 3)

outdir = './general-plots/'

if not os.path.isdir(outdir):
    os.mkdir(outdir)

##############################################################################
######################### Fig with z-bins ####################################
##############################################################################

fname = os.path.join(outdir, 'pz.png')

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

ax.set_ylabel('P(z)')
ax.set_xlabel('z')

ax.legend(loc=0)
plt.tight_layout()
plt.savefig(fname, dpi=DPI)
plt.close()

##############################################################################
######################### Theoretical sph 2b Cls #############################
##############################################################################

fname = os.path.join(outdir, 'cls-sph-2b.png')


ell_cl2bin_file = np.load('./data/cls_lss_2bins.npz')
ell, cl2bin, nls2bin = ell_cl2bin_file['ls'], ell_cl2bin_file['cls'], ell_cl2bin_file['nls']

f, axs = plt.subplots(4, 4, figsize=(8, 6), gridspec_kw={'hspace': 0, 'wspace': 0}, sharex=True, sharey='row')

ci = 0
for i in range(cl2bin.shape[0]):
    if i in (2, 5):
        continue
    cj = ci
    for j in range(i, cl2bin.shape[0]):
        if j in (2, 5):
            continue
        axs[ci, cj].loglog(ell, cl2bin[i, j], label='Th. w/o noise')
        axs[ci, cj].loglog(ell, cl2bin[i, j] + nls2bin[ci, cj], label='Th.', ls='--')
        axs[ci, cj].loglog(ell, nls2bin[i, j], label='Noise')
        cj += 1

    ci += 1

axs[0, 0].legend(loc=0)

i, j = np.triu_indices(axs.shape[0])
for ax in axs[j, i].reshape((-1)):
    ax.set_visible(False)

labels = ['T1', 'E1', 'T2', 'E2']
for i in range(axs.shape[0]):
    axs[i, i].set_xlabel('l')
    axs[i, i].set_ylabel(r'$C_l$')
    axs[i, i].set_visible(True)
    axs[i, i].xaxis.set_tick_params(labelbottom=True)
    axs[i, i].yaxis.set_tick_params(labelleft=True)
    axs[0, i].set_title(labels[i])
    axs[i, -1].text(1, 0.5, labels[i], rotation=270,
                    transform=axs[i, -1].transAxes)


plt.tight_layout()
plt.savefig(fname, dpi=DPI)
plt.close()

##############################################################################
########################## Compare 1bin sph Cls ##############################
##############################################################################

fname = os.path.join(outdir, 'compare-sph-cls-used-1bin.png')

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
axs[0].set_ylabel(r'$C_l$')

axs[0].text(0.5, 0.9, r'$C_l^{gg}$', transform=axs[0].transAxes)

#### TE #####
axs[1].loglog(lTh, clTh_TE, label='1-bin Fid.')
axs[1].loglog(ell, cl2bins_1[0, 1], label='1st 2-bin Fid.', ls='--')
axs[1].loglog(ell, cl2bins_2[0, 1], label='2nd 2-bin Fid.', ls='--')
axs[1].text(0.5, 0.9, r'$C_l^{g\gamma}$', transform=axs[1].transAxes)
#axs[1].set_ylabel(r'$C_l^{g\gamma}$')

#### EE #####
axs[2].loglog(lTh, clTh_EE, label='1-bin Fid.')
axs[2].loglog(ell, cl2bins_1[1, 1], label='1st 2-bin Fid.', ls='--')
axs[2].loglog(ell, cl2bins_2[1, 1], label='2nd 2-bin Fid.', ls='--')
axs[2].text(0.5, 0.9, r'$C_l^{\gamma\gamma}$', transform=axs[2].transAxes)
#axs[2].set_ylabel(r'$C_l^{\gamma\gamma}$')

axs[0].legend(loc=0)

for i in range(axs.shape[0]):
    axs[i].set_xlabel('l')

plt.tight_layout()
plt.savefig(fname, dpi=DPI)
plt.close()

##############################################################################
############################# Mask sph1 ######################################
##############################################################################

fname = os.path.join(outdir, 'mask-lss1.png')

fmask = "./data/mask_lss_sph1.fits"
mask_lss = hp.ud_grade(hp.read_map(fmask, verbose=False), nside_out=512)

hp.mollview(mask_lss)
plt.savefig(fname, dpi=DPI)
plt.close()

##############################################################################
############################# Foregrounds ####################################
##############################################################################

cl_ss = tp.create_cl_templates(ell, cl2bin[0, 0] + nls2bin[0, 0], exp_range=(0, 0), N=1)
cl_ls1 = tp.create_cl_templates(ell, cl2bin[0, 0] + nls2bin[0, 0], exp_range=(-3, -3), N=1)[0]
cl_ls2 = tp.create_cl_templates(ell, cl2bin[0, 0] + nls2bin[0, 0], exp_range=(-1, -1), N=1)[0]

fname = os.path.join(outdir, 'foreground.png')

f, ax = plt.subplots(1, 1, figsize=FSIZE1)
ax.loglog(ell, cl_ls1, c='orange')
ax.loglog(ell, cl_ls2, c='orange', label='Large scales')
ax.fill_between(ell, cl_ls1, cl_ls2, facecolor="orange", alpha=0.5)

ax.loglog(ell, cl_ss[0], c='b', label='Small scales')

ax.set_xlabel('l')
ax.set_ylabel(r'$C_l^{gg}$')

ax.legend(loc=0)

plt.tight_layout()
plt.savefig(fname, dpi=DPI)
plt.close()




