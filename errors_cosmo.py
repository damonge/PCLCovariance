import numpy as np
import pyccl as ccl
import os


nell=30000
dx = 0.1

pars = {
    'h'       : 0.67,
    'Omega_c' : 0.27,
    'Omega_b' : 0.045,
    'A_s'     : 2.1e-9,
    'n_s'     : 0.96,
    'w_0'     : -1.,
    'w_a'     : -0.
}

# var = 'w_0'

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


# Flat sky
ell_bp = np.loadtxt('./Simulations/flat/run_ells.txt').astype(int)
cov_th = np.load('./Simulations/flat/run_covThTTTEEE.npz')['arr_0']
cov_sim = np.load('./Simulations/flat/run_covTTTEEE_clsims_0001-20000.npz')['arr_0']
# Flat sky
ell_bp_sph = np.loadtxt('./Simulations/sph/run_sph_ells.txt').astype(int)
cov_th_sph = np.load('./Simulations/sph/run_sph_covThTTTEEE.npz')['arr_0']
cov_sim_sph = np.load('./Simulations/sph/run_sph_covTTTEEE_clsims_0001-20000.npz')['arr_0']

factor = (20000-len(cov_sim)-2.)/(20000.-1.)
inv_cov_th = np.linalg.inv(cov_th)
inv_cov_sim = factor*np.linalg.inv(cov_sim)

factor_sph = (20000-len(cov_sim_sph)-2.)/(20000.-1.)
inv_cov_th_sph = np.linalg.inv(cov_th_sph)
inv_cov_sim_sph = factor_sph*np.linalg.inv(cov_sim_sph)


z, pz = np.loadtxt('./nz.txt', unpack=True)
bz = np.ones(len(z))
# def nofz(z,z0,sz,ndens):
#     return np.exp(-0.5*((z-z0)/sz)**2)*ndens/np.sqrt(2*np.pi*sz**2)
# z=np.linspace(0,3,512)
# pz=nofz(z,0.955,0.13,7.55)

def get_cl_ccl(pars,ell_bp):

    cosmo = get_cosmo_ccl(pars)
    clust=ccl.NumberCountsTracer(cosmo,has_rsd=False,dndz=(z,pz),bias=(z,bz))
    lens=ccl.WeakLensingTracer(cosmo,dndz=(z,pz))

    ell=np.arange(nell)
    cl0=np.zeros(nell)*0.

    cls=np.zeros([3,3,nell])
    cls[0,0,:]=ccl.angular_cl(cosmo,clust,clust,ell)
    cls[0,1,:]=ccl.angular_cl(cosmo,clust,lens,ell)
    cls[0,2,:]=cl0
    cls[1,0,:]=cls[0,1,:]
    cls[1,1,:]=ccl.angular_cl(cosmo,lens,lens,ell)
    cls[1,2,:]=cl0
    cls[2,0,:]=cls[0,2,:]
    cls[2,1,:]=cls[1,2,:]
    cls[2,2,:]=cl0

    cl_flat = np.concatenate(
        [cls[0,0,ell_bp],
        cls[0,1,ell_bp],
        cls[1,1,ell_bp]]
    )

    return cl_flat


def diff_cls(pars, var,ell_bp, dx=0.01):
    if pars[var] == 0.:
        Dx = dx
    else:
        Dx = dx*np.abs(pars[var])
    parstmp = pars.copy()
    parstmp[var] = pars[var]+Dx
    clP = get_cl_ccl(parstmp,ell_bp)
    parstmp[var] = pars[var]-Dx
    clM = get_cl_ccl(parstmp,ell_bp)
    return (clP-clM)/(2.*Dx)


print('Flat sky:')
for var in pars.keys():
    diff = diff_cls(pars, var,ell_bp, dx=dx)
    res_th = (diff.dot(inv_cov_th).dot(diff))**(-1./2.)
    res_sim = (diff.dot(inv_cov_sim).dot(diff))**(-1./2.)
    rel_diff = np.abs(res_th/res_sim-1.)
    print('Parameter {}:'.format(var))
    print('----> Theory      : {}'.format(res_th))
    print('----> Simulations : {}'.format(res_sim))
    print('----> Rel diff    : {}'.format(rel_diff))

print('Full sky:')
for var in pars.keys():
    diff = diff_cls(pars, var,ell_bp_sph, dx=dx)
    res_th = (diff.dot(inv_cov_th_sph).dot(diff))**(-1./2.)
    res_sim = (diff.dot(inv_cov_sim_sph).dot(diff))**(-1./2.)
    rel_diff = np.abs(res_th/res_sim-1.)
    print('Parameter {}:'.format(var))
    print('----> Theory      : {}'.format(res_th))
    print('----> Simulations : {}'.format(res_sim))
    print('----> Rel diff    : {}'.format(rel_diff))
