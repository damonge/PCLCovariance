import numpy as np
import pyccl as ccl
import os


def nofz(z,z0,sz,ndens):
    return np.exp(-0.5*((z-z0)/sz)**2)*ndens/np.sqrt(2*np.pi*sz**2)

# pz single bin
z_1  = np.array([np.linspace(0,3,512)])
pz_1 = np.array([nofz(z_1[0],0.955,0.13,7.55)])

# pz new bins
z_2  = np.array([np.linspace(0,3,512),np.linspace(0,3,512)])
pz_2 = np.array([nofz(z_2[0],0.955,0.13,7.55),nofz(z_2[1],1.355,0.13,7.55)])
z_3  = np.array([np.linspace(0,3,512),np.linspace(0,3,512),np.linspace(0,3,512)])
pz_3 = np.array([nofz(z_3[0],0.955,0.13,7.55),nofz(z_3[1],1.355,0.13,7.55),nofz(z_3[2],1.755,0.13,7.55)])
z_n  = np.array([
    np.linspace(0,3,512),
    np.linspace(0,3,512),
    np.linspace(0,3,512),
    np.linspace(0,3,512),
    np.linspace(0,3,512),
    np.linspace(0,3,512),
    np.linspace(0,3,512),
    np.linspace(0,3,512),
    np.linspace(0,3,512),
    np.linspace(0,3,512)
    ])
pz_n = np.array([
    nofz(z_n[0],0.955,0.13,7.55),
    nofz(z_n[1],1.155,0.13,7.55),
    nofz(z_n[2],1.255,0.13,7.55),
    nofz(z_n[3],1.355,0.13,7.55),
    nofz(z_n[4],1.455,0.13,7.55),
    nofz(z_n[5],1.655,0.13,7.55),
    nofz(z_n[6],1.755,0.13,7.55),
    nofz(z_n[7],1.855,0.13,7.55),
    nofz(z_n[8],1.955,0.13,7.55),
    nofz(z_n[9],2.055,0.13,7.55)
    ])

pars = {
    'h'       : 0.67,
    'Omega_c' : 0.27,
    'Omega_b' : 0.045,
    'A_s'     : 2.1e-9,
    'n_s'     : 0.96,
    'w_0'     : -1.,
    'w_a'     : -0.,
    'fsky'    : 1.
}

ell_bp = np.load('./data/run_sph_ells.npz')['lsims'].astype(int)
cov_th_tmp = np.load('./data/run_sph_covThTTTEEE_short.npz')['arr_0']
cov_sim = np.load('./data/run_sph_covTTTEE_short_clsims_0001-20000.npz')['arr_0']
cov_th = np.zeros((1023,1023))

idx = np.triu_indices(1023)
cov_th[idx] = cov_th_tmp[idx]
cov_th.T[idx] = cov_th_tmp[idx]


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


def get_tracers_ccl(cosmo, z, pz):
    n_bins = pz.shape[0]
    bz = np.ones(pz.shape)
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


def get_cls_ccl(cosmo, tracers, ells):
    n_bte = tracers.shape[0]
    n_ells = len(ells)
    cls = np.zeros([n_bte, n_bte, n_ells])

    for c1 in range(n_bte): # c1=te1+b1*n_te
        for c2 in range(c1, n_bte):
            cls[c1,c2,:] = ccl.angular_cl(cosmo,tracers[c1],tracers[c2],ells)
            cls[c2,c1,:] = cls[c1,c2,:]

    return cls


def flatten_ell_cls(cls, n_bte, n_ells):
    flat_cls = np.moveaxis(cls,[-3,-2,-1],[0,1,2])
    flat_cls = flat_cls[np.triu_indices(n_bte)]
    return flat_cls


def flatten_cls(cls, n_bte, n_ells):
    flat_cls = flatten_ell_cls(cls, n_bte, n_ells)
    flat_cls = flat_cls.reshape(((n_bte+1)*n_bte*n_ells/2,)+cls.shape[:-3])
    return flat_cls


def unflatten_ell_cls(cls, n_bte, n_ells):
    unflat_cls = np.zeros((n_bte,n_bte,n_ells))
    unflat_cls[np.triu_indices(n_bte)] = cls
    unflat_cls = np.moveaxis(unflat_cls,[0,1],[1,0])
    unflat_cls[np.triu_indices(n_bte)] = cls
    return unflat_cls


def unflatten_cls(cls, n_bte, n_ells):
    tmp_cls = cls.reshape((n_bte+1)*n_bte/2,n_ells)
    unflat_cls = unflatten_ell_cls(tmp_cls, n_bte, n_ells)
    return unflat_cls


def flatten_ell_covmat(cov, n_bte, n_ells):
    flat_cov = flatten_ell_cls(cov, n_bte, n_ells)
    flat_cov = flatten_ell_cls(flat_cov, n_bte, n_ells)
    return flat_cov


def flatten_covmat(cov, n_bte, n_ells):
    flat_cov = flatten_cls(cov, n_bte, n_ells)
    flat_cov = flatten_cls(flat_cov, n_bte, n_ells)
    return flat_cov


def unflatten_covmat(cov, n_bte, n_ells):
    unflat_cov = np.apply_along_axis(unflatten_cls, -1, cov, n_bte, n_ells)
    unflat_cov = np.apply_along_axis(unflatten_cls, -4, unflat_cov, n_bte, n_ells)
    return unflat_cov


def unflatten_ell_covmat(cov, n_bte, n_ells):
    flat_cov = cov.reshape(((n_bte+1)*n_bte*n_ells/2,(n_bte+1)*n_bte*n_ells/2))
    unflat_cov = unflatten_covmat(flat_cov, n_bte, n_ells)
    return unflat_cov


def get_lambda_cov(cls, ells, fsky, n_bte, n_ells):
    delta_ell = np.average(np.ediff1d(ells))
    clscls = np.zeros((n_bte,n_bte,n_bte,n_bte,n_ells))
    for x in range(n_ells):
        clscls[:,:,:,:,x] = np.multiply.outer(cls[:,:,x], cls[:,:,x])
    cov = np.moveaxis(clscls,[0,1,2,3],[0,2,1,3]) + np.moveaxis(clscls,[0,1,2,3],[0,3,1,2])
    cov = cov/fsky/(2.*ells+1.)/delta_ell
    return cov


def get_cov_N(cov_lambda):
    cov = np.apply_along_axis(np.diag,-1,cov_lambda)
    cov = np.moveaxis(cov,[0,1,2,3,4,5],[0,1,3,4,2,5])
    return cov


def get_cov_F(cov_c, cov_n, n_bte, n_ells):
    cov_c_l = unflatten_covmat(cov_c, n_bte, n_ells)
    cov_c_l = np.diagonal(cov_c_l,axis1=2,axis2=5)
    cov_n_l = np.diagonal(cov_n,axis1=2,axis2=5)
    cov_f = cov_c_l/cov_n_l
    return cov_f


def get_cov_R(cov_c, n_bte, n_ells):
    cov_c_l = unflatten_covmat(cov_c, n_bte, n_ells)
    cov_c_l = flatten_ell_covmat(cov_c_l, n_bte, n_ells)
    cov_c_l_d = np.diagonal(cov_c_l,axis1=1,axis2=3)
    cov_c_l_2 = np.multiply.outer(cov_c_l_d, cov_c_l_d)
    cov_c_l_2 = np.diagonal(cov_c_l_2,axis1=0,axis2=3)
    cov_c_l_2 = np.diagonal(cov_c_l_2,axis1=0,axis2=2)
    cov_c_l_2 = np.moveaxis(cov_c_l_2,[0,1],[-3,-1])
    cov_r = cov_c_l/np.sqrt(cov_c_l_2)
    cov_r = unflatten_ell_covmat(cov_r, n_bte, n_ells)
    return cov_r


def get_idx_cov(pos, idx):
    if pos>=len(idx[0]):
        raise IOError('pos is larger than the number of elements. Maximum: {}'.format(len(idx[0])-1))
    return idx[0][pos], idx[1][pos]


def get_cov(cov_c, pars, ells, z_1, pz_1, z_n, pz_n):
    # Common
    n_ells = ells.shape[0]
    cosmo = get_cosmo_ccl(pars)
    # 1 bin
    tracers = get_tracers_ccl(cosmo, z_1, pz_1)
    n_bte_1 = tracers.shape[0]
    cls = get_cls_ccl(cosmo, tracers, ells)
    lambda_cov = get_lambda_cov(cls, ells, pars['fsky'], n_bte_1, n_ells)
    cov_n = get_cov_N(lambda_cov)
    cov_f = get_cov_F(cov_c, cov_n, n_bte_1, n_ells)
    cov_r = get_cov_R(cov_c, n_bte_1, n_ells)
    # n bins
    tracers = get_tracers_ccl(cosmo, z_n, pz_n)
    n_bte_n = tracers.shape[0]
    cls = get_cls_ccl(cosmo, tracers, ells)
    lambda_cov = get_lambda_cov(cls, ells, pars['fsky'], n_bte_n, n_ells)
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


cov = get_cov(cov_th, pars, ell_bp, z_1, pz_1, z_1, pz_1)
print(np.count_nonzero(cov-cov.T))
print(np.abs((cov/cov_th-1.).flatten()).max())
