#!/usr/bin/python

import pymaster as nmt
import numpy as np
import healpy as hp
import os

#pylint: disable=C0103

##############################################################################
##############################################################################
##############################################################################

prefix = 'run_sph_2b_same_mask'
run_path = os.path.join('./simulations_outputs/', prefix, prefix)
output_dir = os.path.join('./simulations_outputs/', prefix, 'full_covariance')

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
out_run_path = os.path.join(output_dir, prefix)
data_folder = './data/'

nside=512

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
w02 = nmt.NmtWorkspace()
w22 = nmt.NmtWorkspace()

w00.read_from(run_path + "_w00_11.dat")
w02.read_from(run_path + "_w02_11.dat")
w22.read_from(run_path + "_w22_11.dat")

def get_fields(mask_ar, w_cont=False):
    """
    Generate a simulated field.
    It returns two NmtField objects for a spin-0 and a spin-2 field.

    :param mask: a sky mask.
    :param w_cont: deproject any contaminants? (not implemented yet)
    """
    nbins = 2
    spins = [0, 2] * nbins
    # maps == [st1, sq1, su1, st2, sq2, su2, ...] (oredered as in spins)
    maps = nmt.synfast_spherical(nside, clTh_all, spins)
    st1, sq1, su1, st2, sq2, su2 = maps

    if w_cont:
        raise ValueError('Contaminants not implemented yet')
    else:
        ff0_1 = nmt.NmtField(mask_ar[0], [st1])
        ff0_2 = nmt.NmtField(mask_ar[1], [st2])
        ff2_1 = nmt.NmtField(mask_ar[0], [sq1, su1])
        ff2_2 = nmt.NmtField(mask_ar[1], [sq2, su2])

    return (ff0_1, ff2_1), (ff0_2, ff2_2)

np.random.seed(1000)
fields = get_fields(mask_lss_ar) #, o.nss_cont or o.nls_cont)
fbin1, fbin2 = fields

cw = nmt.NmtCovarianceWorkspace()
# All bins with same mask so we can reuse cw
fname = out_run_path + '_cw.dat'
if os.path.isfile(fname):
    cw.read_from(fname)
else:
    cw.compute_coupling_coefficients(fbin1[0], fbin1[0])
    cw.write_to(fname)

##############################################################################
##############################################################################
##############################################################################
# We want to compute the cov matrix given by the tensorial product of:
# T1T1 T1E1 E1E1 T1T2 T1E2 E1E2 T2T2 T2E2 E2E2

# nmt.gaussian_covariance(cw, s_bin1_1, s_bin1_2, s_bin2_1, s_bin2_2...)
# returns a covariance matrix with shape (l's, combination_fields_bin1, l's, combination_fields_bin2)
# where combination_fields_bin1 is for spins 00, TT; for spins 02, TE and TB;
# for spins 22, EE, EB, BE, BB.

# CovTh_TT = c0000
# CovTh_TTTE, CovTh_TTTB = c0002[:, 0, :, [0, 1]]
# CovTh_TTEE, CovTh_TTEB, CovTh_TTBE, CovTh_TTBB = c0022[:, 0, :, [0, 1, 2, 3]]
# CovTh_TETE, CovTh_TETB = c0202[:, 0, :, [0, 1] ]
# CovTh_TBTE, CovTh_TBTB = c0202[:, 1, :, [0, 1] ]
# CovTh_TEEE, CovTh_TEEB, CovTh_TEBE, CovTh_TEBB =  c0222[:, 0, :, [0, 1, 2, 3] ]
# CovTh_TBEE, CovTh_TBEB, CovTh_TBBE, CovTh_TBBB = c0222[:, 1, :, [0, 1, 2, 3] ]
# CovTh_EEEE, CovTh_EEEB, CovTh_EEBE, CovTh_EEBB = c2222[:, 0, :, [0, 1, 2, 3] ]
# CovTh_EBEE, CovTh_EBEB, CovTh_EBBE, CovTh_EBBB = c2222[:, 1, :, [0, 1, 2, 3] ]
# CovTh_BEEE, CovTh_BEEB, CovTh_BEBE, CovTh_BEBB = c2222[:, 2, :, [0, 1, 2, 3] ]
# CovTh_BBEE, CovTh_BBEB, CovTh_BBBE, CovTh_BBBB = c2222[:, 3, :, [0, 1, 2, 3] ]

# nmt.gaussian_covariance(cw, spin_a1, spin_a2, spin_b1, spin_b2, cla1b1, cla1b2, cla2b1, cla2b2, wa, wb=None)

# T1T1_T1T1 = nmt.gaussian_covariance(cw, 0, 0, 0, 0, [clt1t1], [clt1t1], [clt1t1], [clt1t1], w00)
# T1T1_T1E1 = nmt.gaussian_covariance(cw, 0, 0, 0, 2, [clt1t1], [clt1e1, clt1b1], [clt1t1], [clt1e1, clt1b1], w00)



# T1T2T1T2 = nmt.gaussian_covariance(cw, 0, 0, 0, 0, [clt1t1], [clt1t2], [clt2t1], [clt2t2], w00)
# T2T2T2T2 = nmt.gaussian_covariance(cw, 0, 0, 0, 0, [clt2t2], [clt2t2], [clt2t2], [clt2t2], w00)

Cls = np.array(
       [['T0T0', 'T0E0', 'T0B0', 'T0T1', 'T0E1', 'T0B1'],
       ['E0T0', 'E0E0', 'E0B0', 'E0T1', 'E0E1', 'E0B1'],
       ['B0T0', 'B0E0', 'B0B0', 'B0T1', 'B0E1', 'B0B1'],
       ['T1T0', 'T1E0', 'T1B0', 'T1T1', 'T1E1', 'T1B1'],
       ['E1T0', 'E1E0', 'E1B0', 'E1T1', 'E1E1', 'E1B1'],
       ['B1T0', 'B1E0', 'B1B0', 'B1T1', 'B1E1', 'B1B1']])

Cls_flat = np.concatenate(Cls)

Cov = np.empty((Cls_flat.shape[0], Cls_flat.shape[0]), dtype=object)
for i, clsi in enumerate(Cls_flat):
    for j, clsj in enumerate(Cls_flat):
        Cov[i, j] = clsi + clsj

# print(Cov)
def get_workspace_from_spins(s1, s2):
    if (s1 == s2) and (s1 == 0):
        return w00
    elif (s1 == s2) and (s1 == 2):
        return w22
    else:
        return w02

def get_nelems_spin(spin):
    if spin == 0:
        return 1
    if spin == 2:
        return 2


def compute_covariance_full(nmaps, nbins, maps_bins, maps_spins):

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

    for i, indices in enumerate(cov_indices):
        s_a1, s_a2, s_b1, s_b2 = cov_spins[i]

        na1 = get_nelems_spin(s_a1)
        na2 = get_nelems_spin(s_a2)
        nb1 = get_nelems_spin(s_b1)
        nb2 = get_nelems_spin(s_b2)

        bin_a1, bin_a2, bin_b1, bin_b2 = cov_bins[i]

        fname = out_run_path + '_cov_c{}{}{}{}_{}{}{}{}.npz'.format(*cov_spins[i], *cov_bins[i])
        if os.path.isfile(fname):
            continue

        ibin_a1 = np.where(maps_bins == bin_a1)[0][0] + int(s_a1 / 2)
        ibin_a2 = np.where(maps_bins == bin_a2)[0][0] + int(s_a2 / 2)
        ibin_b1 = np.where(maps_bins == bin_b1)[0][0] + int(s_b1 / 2)
        ibin_b2 = np.where(maps_bins == bin_b2)[0][0] + int(s_b2 / 2)

        cla1b1 = np.concatenate(clTh[ibin_a1 : ibin_a1 + na1, ibin_b1 : ibin_b1 + nb1])
        cla1b2 = np.concatenate(clTh[ibin_a1 : ibin_a1 + na1, ibin_b2 : ibin_b2 + nb2])
        cla2b1 = np.concatenate(clTh[ibin_a2 : ibin_a2 + na2, ibin_b1 : ibin_b1 + nb1])
        cla2b2 = np.concatenate(clTh[ibin_a2 : ibin_a2 + na2, ibin_b2 : ibin_b2 + nb2])

        wa = get_workspace_from_spins(s_a1, s_a2)
        wb = get_workspace_from_spins(s_b1, s_b2)

        cla1b1_label = np.concatenate(Cls[ibin_a1 : ibin_a1 + na1, ibin_b1 : ibin_b1 + nb1])
        cla1b2_label = np.concatenate(Cls[ibin_a1 : ibin_a1 + na1, ibin_b2 : ibin_b2 + nb2])
        cla2b1_label = np.concatenate(Cls[ibin_a2 : ibin_a2 + na2, ibin_b1 : ibin_b1 + nb1])
        cla2b2_label = np.concatenate(Cls[ibin_a2 : ibin_a2 + na2, ibin_b2 : ibin_b2 + nb2])

        # print(np.concatenate(cla1b1))
        # print(np.concatenate(cla1b2))
        # print(np.concatenate(cla2b1))
        # print(np.concatenate(cla2b2))

        print('Computing ', fname)
        print('spins: ', s_a1, s_a2, s_b1, s_b2)
        print('cla1b1', (s_a1, s_b1), cla1b1.shape, ibin_a1, ibin_a1 + na1, ibin_b1, ibin_b1 + nb1, cla1b1_label)
        print('cla1b2', (s_a1, s_b2), cla1b2.shape, ibin_a1, ibin_a1 + na1, ibin_b2, ibin_b2 + nb2, cla1b2_label)
        print('cla2b1', (s_a2, s_b1), cla2b1.shape, ibin_a2, ibin_a2 + na2, ibin_b1, ibin_b1 + nb1, cla2b1_label)
        print('cla2b2', (s_a2, s_b2), cla2b2.shape, ibin_a2, ibin_a2 + na2, ibin_b2, ibin_b2 + nb2, cla2b2_label)

        cov = nmt.gaussian_covariance(cw, int(s_a1), int(s_a2), int(s_b1), int(s_b2),
                                      cla1b1, cla1b2, cla2b1, cla2b2,
                                      wa, wb)

        np.savez_compressed(fname, cov)


    # Loop through cov_indices, use below algorithm and compute the Cov
    # Check wich one has been computed, store/save it and remove them form cov_indices

nbins = 2
nmaps = 6
maps_bins = [0, 0, 0, 1, 1, 1]
maps_spins = [0, 2, 2] * nbins
compute_covariance_full(nmaps, nbins, maps_bins, maps_spins)

