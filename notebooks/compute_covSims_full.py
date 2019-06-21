#!/usr/bin/python
import numpy as np
import os
import sys

# pylint: disable=C0103

# prefix = 'run_sph_2b_same_mask'
prefix = os.path.basename(sys.argv[1].rstrip('/'))
run_path = os.path.join('./simulations_outputs/', prefix, prefix)
outdir = os.path.join('./simulations_outputs/', prefix, 'full_covariance')
output_path = os.path.join(outdir, prefix)


##################
ells = np.loadtxt(run_path + '_ells.txt')

lmax = (ells < 2*512).sum()

cl_ar = np.load(run_path + '_cl_0001-20000.npz')['arr_0']
print(cl_ar.shape)

Cls = [0, 1, 2] * 2  # 0 for T, 1 for E and 2 for B
Cls_Bs_ar = []

c = 0
for i, Cli in enumerate(Cls):
    for Clj in Cls[i:]:
        Cls_Bs_ar.append([Cli, Clj])

Cls_Bs_ar = np.array(Cls_Bs_ar)

cl_ar_noBs = cl_ar[np.all(Cls_Bs_ar != 2, axis=1)]

cl_for_C = np.concatenate(cl_ar_noBs[:, :, :lmax].swapaxes(1, 2))

C = np.cov(cl_for_C)

fname = output_path + '_covSims_TTTEEE_short_0001-20000.npz' # sims_suffix
np.savez_compressed(fname, C)
