import numpy as np
import pyccl as ccl
import os
import sys



def get_cosmo_ccl(pars):
    cosmo = ccl.Cosmology(
        h        = pars['h'],
        Omega_c  = pars['Omega_c'],
        Omega_b  = pars['Omega_b'],
        A_s      = pars['A_s'],
        n_s      = pars['n_s'],
        w0       = pars['w_0'],
        wa       = pars['w_a']
        )
    return cosmo


def nofz(z,z0,sz,ndens):
    return np.exp(-0.5*((z-z0)/sz)**2)*ndens/np.sqrt(2*np.pi*sz**2)


pars = {
    'h'       : 0.67,
    'Omega_c' : 0.27,
    'Omega_b' : 0.045,
    'A_s'     : 2.1e-9,
    'n_s'     : 0.96,
    'w_0'     : -1.,
    'w_a'     : -0.
}

z_ref = np.linspace(0,3,512)
cosmo = get_cosmo_ccl(pars)
bz_ref=0.95*ccl.growth_factor(cosmo,1.)/ccl.growth_factor(cosmo,1./(1+z_ref))


z = {}
pz = {}
bz = {}
fsky = {}
cov_sim = {}
cov_th = {}


# pz 1 bin
z[1]    = np.tile(z_ref,[1,1])
pz[1]   = np.array([nofz(z_ref,0.955,0.13,7.55)])
bz[1]    = np.tile(0.65*bz_ref,[1,1])
fsky[1] = 1.#TODO



# pz 2 bins
z[2]    = np.tile(z_ref,[2,1])
pz[2]   = np.array([
    nofz(z_ref,0.955,0.13,7.55),
    nofz(z_ref,0.755,0.13,7.55)
    ])
bz[2]    = np.tile(bz_ref,[2,1])
fsky[2] = 1.#TODO


# pz 10 bins
z[10]    = np.tile(z_ref,[10,1])
pz[10] = np.array([#TODO
    nofz(z[10][0],0.955,0.13,7.55),
    nofz(z[10][1],1.155,0.13,7.55),
    nofz(z[10][2],1.255,0.13,7.55),
    nofz(z[10][3],1.355,0.13,7.55),
    nofz(z[10][4],1.455,0.13,7.55),
    nofz(z[10][5],1.655,0.13,7.55),
    nofz(z[10][6],1.755,0.13,7.55),
    nofz(z[10][7],1.855,0.13,7.55),
    nofz(z[10][8],1.955,0.13,7.55),
    nofz(z[10][9],2.055,0.13,7.55)
    ])
bz[2]    = np.tile(bz_ref,[2,1])#TODO
fsky[10] = 1.#TODO



ell_bp = np.load('./data/run_sph_ells.npz')['lsims'].astype(int)

# Cov 1 bin
cov_sim[1] = np.load('./data/run_sph_covTTTEE_short_clsims_0001-20000.npz')['arr_0']
cov_th_tmp = np.load('./data/run_sph_covThTTTEEE_short.npz')['arr_0']
cov_th[1] = np.zeros(cov_th_tmp.shape)
idx = np.triu_indices(cov_th_tmp.shape[0])
cov_th[1][idx] = cov_th_tmp[idx]
cov_th[1] = cov_th[1].T
cov_th[1][idx] = cov_th_tmp[idx]

# Cov 2 bins
cov_sim[2] = np.load('/mnt/bluewhale/gravityls_3/PCLCovariance/run_sph_2b_same_mask_covSims_TTTEEE_short_0001-20000.npz')['arr_0']
cov_th_tmp = np.load('/mnt/bluewhale/gravityls_3/PCLCovariance/run_sph_2b_same_mask_covTh_TTTEEE_short_2bins_same_mask.npz')['arr_0']
cov_th[2] = np.zeros(cov_th_tmp.shape)
idx = np.triu_indices(cov_th_tmp.shape[0])
cov_th[2][idx] = cov_th_tmp[idx]
cov_th[2] = cov_th[2].T
cov_th[2][idx] = cov_th_tmp[idx]




def get_tracers_ccl(cosmo, z, pz, bz):
    n_bins = pz.shape[0]
    # Tracers
    tracers = []
    for i in range(n_bins):
        tracers.append(
            ccl.NumberCountsTracer(cosmo,has_rsd=False,dndz=(z[i],pz[i]),bias=(z[i],bz[i]))
            )
        tracers.append(
            ccl.WeakLensingTracer(cosmo,dndz=(z[i],pz[i]))
            )
    return np.array(tracers)


def get_cls_ccl(cosmo, tracers, ell_bp):
    n_bte = tracers.shape[0]
    n_ells = len(ell_bp)
    cls = np.zeros([n_bte, n_bte, n_ells])

    for c1 in range(n_bte): # c1=te1+b1*n_te
        for c2 in range(c1, n_bte):
            cls[c1,c2,:] = ccl.angular_cl(cosmo,tracers[c1],tracers[c2],ell_bp)
            cls[c2,c1,:] = cls[c1,c2,:]

    return cls


def flatten_cls(cls, n_bte, n_ells):
    flat_cls = np.moveaxis(cls,[-3,-2,-1],[0,1,2])
    flat_cls = flat_cls[np.triu_indices(n_bte)]
    flat_cls = flat_cls.reshape(((n_bte+1)*n_bte*n_ells/2,)+cls.shape[:-3])
    return flat_cls


def unflatten_cls(cls, n_bte, n_ells):
    tmp_cls = cls.reshape((n_bte+1)*n_bte/2,n_ells)
    unflat_cls = np.zeros((n_bte,n_bte,n_ells))
    unflat_cls[np.triu_indices(n_bte)] = tmp_cls
    unflat_cls = np.moveaxis(unflat_cls,[0,1],[1,0])
    unflat_cls[np.triu_indices(n_bte)] = tmp_cls
    return unflat_cls


def flatten_covmat(cov, n_bte, n_ells):
    flat_cov = flatten_cls(cov, n_bte, n_ells)
    flat_cov = flatten_cls(flat_cov, n_bte, n_ells)
    return flat_cov


def unflatten_covmat(cov, n_bte, n_ells):
    unflat_cov = np.apply_along_axis(unflatten_cls, -1, cov, n_bte, n_ells)
    unflat_cov = np.apply_along_axis(unflatten_cls, -4, unflat_cov, n_bte, n_ells)
    return unflat_cov


def get_lambda_cov(cls, ell_bp, fsky, n_bte, n_ells):
    delta_ell = np.average(np.ediff1d(ell_bp))
    clscls = np.zeros((n_bte,n_bte,n_bte,n_bte,n_ells))
    for x in range(n_ells):
        clscls[:,:,:,:,x] = np.multiply.outer(cls[:,:,x], cls[:,:,x])
    cov = np.moveaxis(clscls,[0,1,2,3],[0,2,1,3]) + np.moveaxis(clscls,[0,1,2,3],[0,3,1,2])
    cov = cov/fsky/(2.*ell_bp+1.)/delta_ell
    return cov


def get_cov_N(cov_lambda):
    cov = np.apply_along_axis(np.diag,-1,cov_lambda)
    cov = np.moveaxis(cov,[0,1,2,3,4,5],[0,1,3,4,2,5])
    return cov


def get_idx_cov(pos, idx):
    if pos>=len(idx[0]):
        raise IOError('pos is larger than the number of elements. Maximum: {}'.format(len(idx[0])-1))
    return idx[0][pos], idx[1][pos]


def get_cov(cov_c, nb, nb_ref):
    # Common
    n_ells = ell_bp.shape[0]
    cosmo = get_cosmo_ccl(pars)
    # 1 bin
    tracers = get_tracers_ccl(cosmo, z[nb_ref], pz[nb_ref], bz[nb_ref])
    n_bte_1 = tracers.shape[0]
    cls = get_cls_ccl(cosmo, tracers, ell_bp)
    lambda_cov = get_lambda_cov(cls, ell_bp, fsky[nb_ref], n_bte_1, n_ells)
    cov_n = get_cov_N(lambda_cov)
    cov_f = 1./np.diagonal(cov_n,axis1=2,axis2=5)
    cov_r = unflatten_covmat(cov_c[nb_ref], n_bte_1, n_ells)
    # n bins
    tracers = get_tracers_ccl(cosmo, z[nb], pz[nb], bz[nb])
    n_bte_n = tracers.shape[0]
    cls = get_cls_ccl(cosmo, tracers, ell_bp)
    lambda_cov = get_lambda_cov(cls, ell_bp, fsky[nb], n_bte_n, n_ells)
    # Wrap everything together
    n_cls_n = n_bte_n*(n_bte_n+1)/2
    cov = np.zeros((n_cls_n, n_ells, n_cls_n, n_ells))
    idx_l = np.triu_indices(n_cls_n)
    idx_s = np.triu_indices(n_bte_n)
    for pos in range(n_cls_n*(n_cls_n+1)/2):
        x, y = get_idx_cov(pos, idx_l)
        x1, x2 = get_idx_cov(x, idx_s)
        y1, y2 = get_idx_cov(y, idx_s)
        x1_1, x2_1 = np.mod(x1, n_bte_1), np.mod(x2, n_bte_1)
        y1_1, y2_1 = np.mod(y1, n_bte_1), np.mod(y2, n_bte_1)
        tmp_cov = np.sqrt(lambda_cov[x1,x2,y1,y2,:,None]*lambda_cov[x1,x2,y1,y2,None,:])
        tmp_cov = tmp_cov*np.sqrt(cov_f[x1_1,x2_1,y1_1,y2_1,:,None]*cov_f[x1_1,x2_1,y1_1,y2_1,None,:])
        tmp_cov = tmp_cov*cov_r[x1_1,x2_1,:,y1_1,y2_1,:]
        cov[x,:,y,:] = tmp_cov
        cov[y,:,x,:] = tmp_cov.T
    cov = cov.reshape((n_cls_n*n_ells,n_cls_n*n_ells))

    return cov

# A = np.array([
#     [1,2,3],
#     [4,5,6]
# ])
# print(A[0])
# print(A[1])
# print(A.reshape(6))
# exit(1)

cov1 = get_cov(cov_sim, nb=2, nb_ref=1)
cosmo = get_cosmo_ccl(pars)
tracers = get_tracers_ccl(cosmo, z[2], pz[2], bz[2])
cl = get_cls_ccl(cosmo, tracers, ell_bp)
cl = flatten_cls(cl, tracers.shape[0], ell_bp.shape[0])
chi2 = np.linalg.solve(cov1, cl)
chi2 = cl.dot(chi2)
print(chi2)
chi2 = np.linalg.solve(cov_sim[2], cl)
chi2 = cl.dot(chi2)
print(chi2)

    #     res_th_2 = np.linalg.solve(cov_th_tot, diff)
    #     res_th_2 = (diff.dot(res_th_2))**(-1./2.)

import matplotlib.pyplot as plt
def get_corr(cov):
    return cov/np.sqrt(np.diag(cov)[None,:]*np.diag(cov)[:,None])
plt.figure()
plt.imshow(get_corr(cov1))
plt.colorbar()
plt.figure()
plt.imshow(get_corr(cov_sim[2]))
plt.colorbar()
plt.figure()
plt.imshow(cov_sim[2]/cov1-1.)
plt.colorbar()
plt.show()
print('cccc')
print(cov1-cov_sim[2])
print(cov1[0,0])
print(cov_sim[2][0,0])
print(cov1[0,0])
print(cov_sim[2][0,0]/cov1[0,0])
print(cov_sim[2][22,314]/cov1[22,314])
print(cov_sim[2][0,10]/cov1[0,10])
exit(1)


def diff_cls(pars, var, z, pz, bz, ell_bp, dx=0.01):
    if pars[var] == 0.:
        Dx = dx
    else:
        Dx = dx*np.abs(pars[var])
    parstmp = pars.copy()
    parstmp[var] = pars[var]+Dx
    cosmo = get_cosmo_ccl(parstmp)
    tracers = get_tracers_ccl(cosmo, z, pz, bz)
    clP = get_cls_ccl(cosmo, tracers, ell_bp)
    clP = flatten_cls(clP, tracers.shape[0], ell_bp.shape[0])
    parstmp[var] = pars[var]-Dx
    cosmo = get_cosmo_ccl(parstmp)
    tracers = get_tracers_ccl(cosmo, z, pz, bz)
    clM = get_cls_ccl(cosmo, tracers, ell_bp)
    clM = flatten_cls(clM, tracers.shape[0], ell_bp.shape[0])
    return (clP-clM)/(2.*Dx)


def check_diffs(cov, b1, b2):
    cov_ex = get_cov(cov, b1, b2)
    diff_max = np.abs((cov[b2]/cov_ex-1.).flatten()).max()
    return diff_max


def get_fisher(pars, nb, nbref=None, dx=0.01):
    if nbref==None:
        cov_th_tot = cov_th[nb]
        cov_sim_tot = cov_sim[nb]
    else:
        cov_th_tot = get_cov(cov_th, nb, nbref)
        cov_sim_tot = get_cov(cov_sim, nb, nbref)
    for var in pars.keys():
        diff = diff_cls(pars, var, z[nb], pz[nb], bz[nb], ell_bp, dx=dx)
        res_th = np.linalg.solve(cov_th_tot, diff)
        res_th = (diff.dot(res_th))**(-1./2.)
        res_sim = np.linalg.solve(cov_sim_tot, diff)
        res_sim = (diff.dot(res_sim))**(-1./2.)
        diff = diff_cls(pars, var, z[nb], pz[nb], bz[nb], ell_bp, dx=2.*dx)
        res_th_2 = np.linalg.solve(cov_th_tot, diff)
        res_th_2 = (diff.dot(res_th_2))**(-1./2.)
        res_sim_2 = np.linalg.solve(cov_sim_tot, diff)
        res_sim_2 = (diff.dot(res_sim_2))**(-1./2.)
        print('----> Parameter {}:'.format(var))
        print('--------> Theory      : {:.2e} (stability: {:.2e})'.format(res_th,np.abs(res_th/res_th_2-1.)))
        print('--------> Simulations : {:.2e} (stability: {:.2e})'.format(res_sim,np.abs(res_sim/res_sim_2-1.)))
        print('--------> Rel diff    : {:.2e}'.format(np.abs(res_th/res_sim-1.)))
        sys.stdout.flush()
    return





# cov_c = cov_sim[1]
# b1 = 1
# b2 = 2
# z_1 = z[1]
# pz_1 = pz[1]
# bz_1 = bz[1]
# z_n = z[2]
# pz_n = pz[2]
# bz_n = bz[2]
#
# n_ells = ell_bp.shape[0]
# cosmo = get_cosmo_ccl(pars)
# # 1 bin
# tracers = get_tracers_ccl(cosmo, z_1, pz_1, bz_1)
# n_bte_1 = tracers.shape[0]
# cls = get_cls_ccl(cosmo, tracers, ell_bp)
# lambda_cov = get_lambda_cov(cls, ell_bp, fsky[1], n_bte_1, n_ells)
# cov_n = get_cov_N(lambda_cov)
# cov_f = 1./np.diagonal(cov_n,axis1=2,axis2=5)
# cov_r = unflatten_covmat(cov_c, n_bte_1, n_ells)
# # n bins
# tracers = get_tracers_ccl(cosmo, z_n, pz_n, bz_n)
# n_bte_n = tracers.shape[0]
# cls = get_cls_ccl(cosmo, tracers, ell_bp)
# lambda_cov = get_lambda_cov(cls, ell_bp, fsky[2], n_bte_n, n_ells)
#
# print(cov_f.shape)
# print(cov_r.shape)
# print(lambda_cov.shape)
# print(cov_c.shape)
#
# # print(cov_c[22,314])
# # print(cov_r[0,0,22,0,0,314])
# print(1./cov_f[0,0,0,0,22])
# print(lambda_cov[0,0,0,0,22]*cov_f[0,0,0,0,22])
# print(1./cov_f[0,0,0,0,314])
# print(lambda_cov[0,0,0,0,314]*cov_f[0,0,0,0,314])
#
#
# print(lambda_cov[0,0,0,0,0]*cov_f[0,0,0,0,0])
# exit(1)



# cov = cov_sim
# b1 = 1
# b2 = 2
#
# cov_ex = get_cov(cov_sim, nb, nbref)
# diff = np.abs(cov[b2]/cov_ex-1.)
# max = diff[:341,:341].flatten().max()
# print(np.argwhere(diff==max))
# print(diff[:341,:341].flatten().max())
#
# exit(1)


# # Internal checks 1 bin
# print('Internal checks 1 bin:')
# print('----> Rel diff cov, cov_ex SIM: {:.2e}'.format(check_diffs(cov_sim, 1, 1)))
# print('----> Rel diff cov, cov_ex  TH: {:.2e}'.format(check_diffs(cov_th, 1, 1)))
# sys.stdout.flush()
#
# # External checks 1 bin -> 2 bins
# print('External checks 1 bin -> 2 bins:')
# print('----> Rel diff cov, cov_ex SIM: {:.2e}'.format(check_diffs(cov_sim, 1, 2)))
# print('----> Rel diff cov, cov_ex  TH: {:.2e}'.format(check_diffs(cov_th, 1, 2)))
# sys.stdout.flush()



# # Fisher 1 bin
# print('Results Fisher 1 bin:')
# get_fisher(pars, 1, nbref=None)

# Fisher 2 bins
print('Results Fisher 2 bins:')
get_fisher(pars, 2, nbref=None)

# Fisher 2 bins extrapolated
print('Results Fisher 2 bins extrapolated:')
get_fisher(pars, 2, nbref=1)

# Fisher 10 bins extrapolated
print('Results Fisher 10 bins extrapolated:')
# get_fisher(pars, 10, nbref=1)
