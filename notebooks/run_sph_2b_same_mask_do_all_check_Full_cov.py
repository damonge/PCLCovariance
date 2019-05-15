#!/usr/bin/python
import common as co
import numpy as np
import os

# pylint: disable=C0103

prefix = 'run_sph_2b_same_mask'
run_path = os.path.join('./simulations_outputs/', prefix, prefix)
outdir = os.path.join('./simulations_outputs/', prefix, 'full_covariance')
output_path = os.path.join(outdir, prefix)
figures_dir = os.path.join('./simulations_outputs/', prefix, 'figures')
figures_path = os.path.join(figures_dir, prefix)


##################
ells = np.loadtxt(run_path + '_ells.txt')

lmax = (ells < 2*512).sum()

CovSims_path = output_path + '_covSims_TTTEEE_short_0001-20000.npz'
CovTh_path = output_path + '_covTh_TTTEEE_short_2bins_same_mask.npz'

CovSims_Full = np.load(CovSims_path)['arr_0']
CovTh_Full = np.load(CovTh_path)['arr_0']

cl_ar = np.load(run_path + '_cl_0001-20000.npz')['arr_0']
Cls = [0, 1, 2] * 2  # 0 for T, 1 for E and 2 for B
Cls_Bs_ar = []

c = 0
for i, Cli in enumerate(Cls):
    for Clj in Cls[i:]:
        Cls_Bs_ar.append([Cli, Clj])

Cls_Bs_ar = np.array(Cls_Bs_ar)
cl_ar_noBs = cl_ar[np.all(Cls_Bs_ar != 2, axis=1)]

cl_for_C = np.concatenate(cl_ar_noBs[:, :, :lmax])
lbins_Full = np.concatenate([ells[:lmax]] * cl_ar_noBs.shape[0])

##################
foutput = figures_path + '_Efstathiou'
chi2_Full, corr_Full = co.do_all_checks(lbins_Full, cl_for_C, CovSims_Full,
                                        CovTh_Full, modes="All TTTEEE",
                                        row_cov=False, foutput=foutput + '_TTTEEE_Full')
