#!/usr/bin/python

import pymaster as nmt
import numpy as np
import healpy as hp
import os
import sys

#pylint: disable=C0103

##############################################################################
##############################################################################
##############################################################################

# prefix = 'run_sph_2b_same_mask'
prefix = os.path.basename(sys.argv[1].strip('/'))
run_path = os.path.join('./simulations_outputs/', prefix, prefix)
output_dir = os.path.join('./simulations_outputs/', prefix, 'full_covariance')

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
out_run_path = os.path.join(output_dir, prefix)
data_folder = './data/'

nside = 512

##############################################################################
# Load theoretical cls and masks
##############################################################################

f = np.load(os.path.join(data_folder, "cls_lss_2bins.npz"))
lTh = f['ls']
clTh = (f['cls'] + f['nls'])
clTh_all = (f['cls'] + f['nls'])[np.triu_indices(f['cls'].shape[0])]

if 'same_mask' in prefix:
    fmasks = ["data/mask_lss_sph1.fits", "data/mask_lss_sph1.fits"]
else:
    fmasks = ["data/mask_lss_sph1.fits", "data/mask_lss_sph2.fits"]


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
# Generate field to compute the covariance workspaces
##############################################################################

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

##############################################################################
# Generate covariance workspaces
##############################################################################
masks = [0, 0, 0, 1, 1, 1]
fields = [*fbin1, *fbin2]

cl_indices = []
nmaps = len(fields)
for i in range(nmaps):
    for j in range(i, nmaps):
        cl_indices.append([i, j])

cov_indices = []
for i, clij in enumerate(cl_indices):
    for j, clkl in enumerate(cl_indices[i:]):
        cov_indices.append(cl_indices[i] + cl_indices[i + j])

for indices in cov_indices:
    i, j, k, l = indices
    mask1 = masks[i]
    mask2 = masks[j]
    mask3 = masks[k]
    mask4 = masks[l]
    fname = os.path.join(out_run_path + '_cw{}{}{}{}.dat'.format(mask1, mask2, mask3, mask4))
    if not os.path.isfile(fname):
        sys.stdout.write('cw{}{}{}{}.dat\n'.format(mask1, mask2, mask3, mask4))
        cw = nmt.NmtCovarianceWorkspace()
        f1 = fields[i]
        f2 = fields[j]
        f3 = fields[k]
        f4 = fields[l]
        cw.compute_coupling_coefficients(f1, f2, f3, f4)
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

def get_workspace_from_spins_masks(spin1, spin2, mask1, mask2):
        ws = nmt.NmtWorkspace()
        fname = run_path + '_w{}{}_{}{}.dat'.format(spin1, spin2, mask1+1, mask2+1)
        ws.read_from(fname)
        return ws

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

    fname_cw_old = ''
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

        wa = get_workspace_from_spins_masks(s_a1, s_a2, bin_a1, bin_a2)
        wb = get_workspace_from_spins_masks(s_b1, s_b2, bin_b1, bin_b2)

        print('Computing ', fname)
        print('spins: ', s_a1, s_a2, s_b1, s_b2)
        print('cla1b1', (s_a1, s_b1), cla1b1.shape, ibin_a1, ibin_a1 + na1, ibin_b1, ibin_b1 + nb1)
        print('cla1b2', (s_a1, s_b2), cla1b2.shape, ibin_a1, ibin_a1 + na1, ibin_b2, ibin_b2 + nb2)
        print('cla2b1', (s_a2, s_b1), cla2b1.shape, ibin_a2, ibin_a2 + na2, ibin_b1, ibin_b1 + nb1)
        print('cla2b2', (s_a2, s_b2), cla2b2.shape, ibin_a2, ibin_a2 + na2, ibin_b2, ibin_b2 + nb2)

        fname_cw = out_run_path + '_cw{}{}{}{}.dat'.format(*cov_bins[i])
        if fname_cw != fname_cw_old:
            cw = nmt.NmtCovarianceWorkspace()
            cw.read_from(fname_cw)
            fname_cw_old = fname_cw

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

