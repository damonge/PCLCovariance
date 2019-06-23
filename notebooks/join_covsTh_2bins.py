#!/usr/bin/python
import numpy as np
import os
import sys

# pylint: disable=C0103

##############################################################################
##############################################################################
##############################################################################

# prefix = 'run_sph_2b_same_mask'
prefix = os.path.basename(sys.argv[1].strip('/'))
run_path = os.path.join('./simulations_outputs/', prefix, prefix)
output_dir = os.path.join('./simulations_outputs/', prefix, 'full_covariance')
out_run_path = os.path.join(output_dir, prefix)

##############################################################################
##############################################################################
##############################################################################

ells = np.loadtxt(run_path + '_ells.txt')

lmax = (ells < 2*512).sum()

# We want the covariance matrix of:
# T1T1 T1E1 E1E1 T1T2 T1E2 E1T2 E1E2 T2T2 T2E2 E2E2

nbins = 2
nmaps = 4
maps_bins = [0, 0, 1, 1]
maps_spins = [0,  2] * nbins

cl_indices = []
cl_spins = []
cl_bins = []
for i in range(nmaps):
    si = maps_spins[i]
    for j in range(i, nmaps):
        sj = maps_spins[j]
        cl_indices.append([i, j])
        cl_spins.append([si, sj])
        cl_bins.append([maps_bins[i], maps_bins[j]])

cov_indices = []
cov_spins = []
cov_bins = []
for i, clij in enumerate(cl_indices):
    for j, clkl in enumerate(cl_indices[i:]):
        cov_indices.append(cl_indices[i] + cl_indices[i + j])
        cov_spins.append(cl_spins[i] + cl_spins[i + j])
        cov_bins.append(cl_bins[i] + cl_bins[i + j])

cov_indices = np.array(cov_indices)
cov_spins = np.array(cov_spins)
cov_bins = np.array(cov_bins)

##############################################################################
##############################################################################
##############################################################################

fname = out_run_path + '_cov_c0000_0000.npz'
c = np.load(fname)['arr_0']
l_bpw = c.shape[-1]
cov_arr = np.empty((cov_spins.shape[0], l_bpw, l_bpw))

def get_nelems_spin(spin):
    if spin == 0:
        return 1
    if spin == 2:
        return 2

for i, indices in enumerate(cov_indices):
    s_a1, s_a2, s_b1, s_b2 = cov_spins[i]
    bin_a1, bin_a2, bin_b1, bin_b2 = cov_bins[i]

    na1 = get_nelems_spin(s_a1)
    na2 = get_nelems_spin(s_a2)
    nb1 = get_nelems_spin(s_b1)
    nb2 = get_nelems_spin(s_b2)

    fname = out_run_path + '_cov_c{}{}{}{}_{}{}{}{}.npz'.format(*cov_spins[i], *cov_bins[i])
    c = np.load(fname)['arr_0'].reshape((l_bpw, na1 * na2, l_bpw, nb1 * nb2))

    #  TT -> 0; TE -> 0;  EE -> 0
    cov_arr[i] = c[:, 0, :, 0]

cov_mat = np.empty((len(cl_bins), l_bpw, len(cl_bins), l_bpw))
i, j = np.triu_indices(len(cl_bins))
cov_mat[i, :, j, :] = cov_arr
cov_mat[j, :, i, :] = cov_arr.swapaxes(1, 2)
cov_mat = cov_mat.reshape((len(cl_bins) * l_bpw, len(cl_bins) * l_bpw))

fname = out_run_path + '_covTh_TTTEEE_2bins.npz'
np.savez_compressed(fname, cov_mat)

####################### Short version #############################

cov_mat = np.empty((len(cl_bins), lmax, len(cl_bins), lmax))
i, j = np.triu_indices(len(cl_bins))
cov_mat[i, :, j, :] = cov_arr[:, :lmax, :lmax]
cov_mat[j, :, i, :] = cov_arr[:, :lmax, :lmax].swapaxes(1, 2)
cov_mat = cov_mat.reshape((len(cl_bins) * lmax, len(cl_bins) * lmax))

fname = out_run_path + '_covTh_TTTEEE_short_2bins.npz'
np.savez_compressed(fname, cov_mat)

# cov_mat = np.empty((len(cl_bins), len(cl_bins), lmax, lmax))
# i, j = np.triu_indices(len(cl_bins))
# cov_mat[i, j] = cov_arr[:, :lmax, :lmax]
# cov_mat[j, i] = cov_arr[:, :lmax, :lmax].swapaxes(1, 2)
# cov_mat = np.block([[mat for mat in cov_mati] for cov_mati in cov_mat])
#
# fname = out_run_path + '_covTh_TTTEEE_short_2bins_same_mask.npz'
# np.savez_compressed(fname, cov_mat)
