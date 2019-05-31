#!/usr/bin/python

import pymaster as nmt
import numpy as np
import healpy as hp
import os

#pylint: disable=C0103
##############################################################################
##############################################################################
# This script uses the covariance workspace created by the other script for NKA
##############################################################################
##############################################################################

##############################################################################
##############################################################################
##############################################################################

prefix = 'run_sph_2b_same_mask'
run_path = os.path.join('./simulations_outputs/', prefix, prefix)
full_cov_path_NKA = os.path.join('./simulations_outputs/', prefix, 'full_covariance')

output_dir = os.path.join('./simulations_outputs/', prefix, 'full_covariance_spin0')

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
out_run_path = os.path.join(output_dir, prefix)
data_folder = './data/'

nside = 512

##############################################################################
##############################################################################
##############################################################################

f = np.load(os.path.join(data_folder, "cls_lss_2bins.npz"))
lTh = f['ls']
clTh = (f['cls'] + f['nls'])
clTh_all = (f['cls'] + f['nls'])[np.triu_indices(f['cls'].shape[0])]

fmasks = ["data/mask_lss_sph1.fits", "data/mask_lss_sph1.fits"]

mask_lss_ar = []

for fmask in fmasks:
    mask_lss = hp.ud_grade(hp.read_map(fmask, verbose=False), nside_out=nside)
    mask_lss_ar.append(mask_lss)

#Set up binning scheme
fsky = np.mean(np.product(mask_lss_ar, axis=0))
d_ell = int(1./fsky)
b = nmt.NmtBin(nside, nlb=d_ell)

if not os.path.isfile(run_path + '_ells.txt'):
    np.savetxt(run_path + '_ells.txt', b.get_effective_ells())

##############################################################################
##############################################################################
##############################################################################

w00 = nmt.NmtWorkspace()
w00.read_from(run_path + "_w00_11.dat")

cw = nmt.NmtCovarianceWorkspace()
# All bins with same mask so we can reuse cw
fname = os.path.join(full_cov_path_NKA, 'run_sph_2b_same_mask_cw.dat')
cw.read_from(fname)

##############################################################################
##############################################################################
##############################################################################
# We want to compute the cov matrix given by the tensorial product of:
# T1T1 T1E1 E1E1 T1T2 T1E2 E1E2 T2T2 T2E2 E2E2

# nmt.gaussian_covariance(cw, s_bin1_1, s_bin1_2, s_bin2_1, s_bin2_2...)
# returns a covariance matrix with shape (l's, combination_fields_bin1, l's, combination_fields_bin2)
# Spin-0 approximation:
#
# CovTh0_T1E2 = nmt.gaussian_covariance(cw,0,0,0,0,[clt1t1],[clt1e2],[cle2t1],[cle2e2],w00)
# CovTh0_E1T2 = nmt.gaussian_covariance(cw,0,0,0,0,[cle1e1],[cle1t2],[clt2e1],[clt2t2],w00)
# CovTh0_E1E2 = nmt.gaussian_covariance(cw,0,0,0,0,[cle1e1],[cle1e2],[cle2e1],[cle2e2],w00)

def compute_covariance_full_spin0(nmaps, nbins, maps_bins, maps_spins, w00):

    cl_indices = []
    cl_modes = []
    cl_bins = []
    for i in range(nmaps):
        si = maps_modes[i]
        for j in range(i, nmaps):
            sj = maps_modes[j]
            cl_indices.append([i, j])
            cl_modes.append([si, sj])
            cl_bins.append([maps_bins[i], maps_bins[j]])

    cov_indices = []
    cov_modes = []
    cov_bins = []
    for i, clij in enumerate(cl_indices):
        for j, clkl in enumerate(cl_indices[i:]):
            cov_indices.append(cl_indices[i] + cl_indices[i + j])
            cov_modes.append(cl_modes[i] + cl_modes[i + j])
            cov_bins.append(cl_bins[i] + cl_bins[i + j])

    cov_indices = np.array(cov_indices)
    cov_modes = np.array(cov_modes)
    cov_bins = np.array(cov_bins)

    for i, indices in enumerate(cov_indices):
        fname = out_run_path + '_cov_c{}{}{}{}_{}{}{}{}.npz'.format(*cov_modes[i], *cov_bins[i])
        if os.path.isfile(fname):
            continue

        ibin_a1, ibin_a2, ibin_b1, ibin_b2 = indices

        cla1b1 = [clTh[ibin_a1, ibin_b1]]
        cla1b2 = [clTh[ibin_a1, ibin_b2]]
        cla2b1 = [clTh[ibin_a2, ibin_b1]]
        cla2b2 = [clTh[ibin_a2, ibin_b2]]

        cov = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                      cla1b1, cla1b2, cla2b1, cla2b2,
                                      w00)

        np.savez_compressed(fname, cov)


    # Loop through cov_indices, use below algorithm and compute the Cov
    # Check wich one has been computed, store/save it and remove them form cov_indices

nbins = 2
nmaps = 6
maps_bins = [0, 0, 0, 1, 1, 1]
maps_modes = ['T', 'E', 'B'] * nbins
compute_covariance_full_spin0(nmaps, nbins, maps_bins, maps_modes, w00)

