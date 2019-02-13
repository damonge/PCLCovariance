#!/usr/bin/python
import numpy as np
import healpy as hp
import pymaster as nmt

def create_cl_templates(l, cl_ref, l_ref=400, proportion=0.1, N=10, exp_range=(-3,-1)):
    exponents = np.random.uniform(*exp_range, N)
    pre_cl = (l+1)**(exponents[:, None])
    index_ref = np.where(l == l_ref)[0]
    amplitudes = proportion * cl_ref[index_ref] / pre_cl[:, index_ref]

    return amplitudes * pre_cl

def create_templates_flat(nx, ny, lx_rad, ly_rad, l, clTT, clEE, clBB, **kwards):
    TT_cl = create_cl_templates(l, clTT, **kwards)
    EE_cl = create_cl_templates(l, clEE, **kwards)
    BB_cl = create_cl_templates(l, clBB, **kwards)

    templates = []
    for tt, ee, bb in zip(TT_cl, EE_cl, BB_cl):
        templates.append(nmt.synfast_flat(nx, ny, lx_rad, ly_rad, [tt, 0*tt, 0*tt, ee, 0*tt, bb], [0,2]))

    return np.array(templates)

def create_templates(l, nside, clTT, clEE, clBB, **kwards):
    TT_cl = create_cl_templates(l, clTT, **kwards)
    EE_cl = create_cl_templates(l, clEE, **kwards)
    BB_cl = create_cl_templates(l, clBB, **kwards)

    templates = []
    for tt, ee, bb in zip(TT_cl, EE_cl, BB_cl):
        templates.append(hp.synfast([tt, 0*tt, 0*tt, ee, 0*tt, bb], nside,
                                    new=True, verbose=False, pol=True))

    return np.array(templates)
