import numpy as np
import pyccl as ccl
import os


def nofz(z,z0,sz,ndens):
    return np.exp(-0.5*((z-z0)/sz)**2)*ndens/np.sqrt(2*np.pi*sz**2)

# pz single bin
z_1  = np.array([np.linspace(0,3,512)])
pz_1 = np.array([nofz(z_1[0],0.955,0.13,7.55)])

# pz new bins
z  = np.array([np.linspace(0,3,512),np.linspace(0,3,512),np.linspace(0,3,512)])
pz = np.array([nofz(z[0],0.955,0.13,7.55),nofz(z[1],1.355,0.13,7.55),nofz(z[1],1.755,0.13,7.55)])
z_2  = np.array([np.linspace(0,3,512),np.linspace(0,3,512)])
pz_2 = np.array([nofz(z[0],0.955,0.13,7.55),nofz(z[1],1.355,0.13,7.55)])

pars = {
    'h'       : 0.67,
    'Omega_c' : 0.27,
    'Omega_b' : 0.045,
    'A_s'     : 2.1e-9,
    'n_s'     : 0.96,
    'w_0'     : -1.,
    'w_a'     : -0.
}



ell_bp = np.load('./data/run_sph_ells.npz')['lsims'].astype(int)
cov_th = np.load('./data/run_sph_covThTTTEEE.npz')['arr_0']
cov_sim = np.load('./data/run_sph_covTTTEEE_clsims_0001-20000.npz')['arr_0']
print(cov_th.shape)
print(cov_sim.shape)



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
            [ccl.NumberCountsTracer(cosmo,has_rsd=False,dndz=(z[i],pz[i]),bias=(z[i],bz[i])),
            ccl.WeakLensingTracer(cosmo,dndz=(z[i],pz[i]))]
            )
    return np.array(tracers)

def get_cl_ccl(cosmo, tracers, ells):
    n_bins = tracers.shape[0]
    n_te = tracers.shape[1]
    n_ells = len(ells)
    cls = np.zeros([n_bins, n_te, n_bins, n_te, n_ells])

    for c1 in range(n_bins*n_te): # c1=te1+b1*n_te
        for c2 in range(c1, n_bins*n_te):
            b1, te1 = np.divmod(c1, n_te)
            b2, te2 = np.divmod(c2, n_te)
            cls[b1,te1,b2,te2,:] = ccl.angular_cl(cosmo,tracers[b1,te1],tracers[b2,te2],ells)
            cls[b2,te2,b1,te1,:] = cls[b1,te1,b2,te2,:]

    return cls

def flatten_cl(cl):
    n_bins = cl.shape[0]
    n_te = cl.shape[1]
    n_ells = cl.shape[-1]
    flat_cl = cl.reshape((n_bins*n_te,n_bins*n_te,n_ells))
    flat_cl = flat_cl[np.triu_indices((n_bins*n_te))]
    flat_cl = flat_cl.flatten()
    return flat_cl


def unflatten_cl(cl, shape):
    n_bins = shape[0]
    n_te = shape[1]
    n_ells = shape[-1]
    tmp_cl = cl.reshape((n_bins*n_te+1)*n_bins*n_te/2,n_ells)
    unflat_cl = np.zeros((n_bins*n_te,n_bins*n_te,n_ells))
    unflat_cl[np.triu_indices(n_bins*n_te)] = tmp_cl
    unflat_cl = np.moveaxis(unflat_cl,[0,1],[1,0])
    unflat_cl[np.triu_indices(n_bins*n_te)] = tmp_cl
    unflat_cl = unflat_cl.reshape(n_bins,n_te,n_bins,n_te,n_ells)
    return unflat_cl



cosmo = get_cosmo_ccl(pars)
tracers = get_tracers_ccl(cosmo, z_2, pz_2)
cls = get_cl_ccl(cosmo, tracers, ell_bp)
flat_cls = flatten_cl(cls)
print(flat_cls.shape)
print(len(ell_bp))
unflat_cls = unflatten_cl(flat_cls, cls.shape)
print(np.min((cls-unflat_cls).flatten()))

# cls = np.zeros((3,2,3,2,5))
# std = np.ones(5)
# cls[0,0,0,0,:] = 1.*std
# cls[0,0,0,1,:] = 2.*std
# cls[0,0,1,0,:] = 3.*std
# cls[0,0,1,1,:] = 4.*std
# cls[0,0,2,0,:] = 5.*std
# cls[0,0,2,1,:] = 6.*std
# cls[0,1,0,0,:] = cls[0,0,0,1,:]
# cls[0,1,0,1,:] = 7.*std
# cls[0,1,1,0,:] = 8.*std
# cls[0,1,1,1,:] = 9.*std
# cls[0,1,2,0,:] = 10.*std
# cls[0,1,2,1,:] = 11.*std
#
# cls[1,0,0,0,:] = cls[0,0,1,0,:]
# cls[1,0,0,1,:] = cls[0,1,1,0,:]
# cls[1,0,1,0,:] = 12.*std
# cls[1,0,1,1,:] = 13.*std
# cls[1,0,2,0,:] = 14.*std
# cls[1,0,2,1,:] = 15.*std
# cls[1,1,0,0,:] = cls[0,0,1,1,:]
# cls[1,1,0,1,:] = cls[0,1,1,1,:]
# cls[1,1,1,0,:] = cls[1,0,1,1,:]
# cls[1,1,1,1,:] = 16.*std
# cls[1,1,2,0,:] = 17.*std
# cls[1,1,2,1,:] = 18.*std
#
# cls[2,0,0,0,:] = cls[0,0,2,0,:]
# cls[2,0,0,1,:] = cls[0,1,2,0,:]
# cls[2,0,1,0,:] = cls[1,0,2,0,:]
# cls[2,0,1,1,:] = cls[1,1,2,0,:]
# cls[2,0,2,0,:] = 19.*std
# cls[2,0,2,1,:] = 20.*std
# cls[2,1,0,0,:] = cls[0,0,2,1,:]
# cls[2,1,0,1,:] = cls[0,1,2,1,:]
# cls[2,1,1,0,:] = cls[1,0,2,1,:]
# cls[2,1,1,1,:] = cls[1,1,2,1,:]
# cls[2,1,2,0,:] = cls[2,0,2,1,:]
# cls[2,1,2,1,:] = 21.*std
#
# flat_cls = flatten_cl(cls)
# unflat_cls = unflatten_cl(flat_cls, cls.shape)
# print((cls-unflat_cls).flatten())

exit(1)


def get_cl_ccl(pars, ell_bp, z, pz):

    n_ells = len(ell_bp)


    # Cosmo
    cosmo = get_cosmo_ccl(pars)


    # tracers = np.empty([2,n_bins])
    # tracers[0,0]=ccl.NumberCountsTracer(cosmo,has_rsd=False,dndz=(z[0],pz[0]),bias=(z[0],bz[0]))
    try:
        print(tracers.shape)
    except:
        pass

    # Cls
    cls = np.zeros([2,2,n_bins,n_bins,n_ells])

    #
    # lens=
    #
    # ell=np.arange(nell)
    # cl0=np.zeros(nell)*0.
    #
    # cls=np.zeros([3,3,nell])
    # cls[0,0,:]=ccl.angular_cl(cosmo,clust,clust,ell)
    # cls[0,1,:]=ccl.angular_cl(cosmo,clust,lens,ell)
    # cls[0,2,:]=cl0
    # cls[1,0,:]=cls[0,1,:]
    # cls[1,1,:]=ccl.angular_cl(cosmo,lens,lens,ell)
    # cls[1,2,:]=cl0
    # cls[2,0,:]=cls[0,2,:]
    # cls[2,1,:]=cls[1,2,:]
    # cls[2,2,:]=cl0

    cls = 0
    # cl_flat = np.concatenate(
    #     [cls[0,0,ell_bp],
    #     cls[0,1,ell_bp],
    #     cls[1,1,ell_bp]]
    # )

    return cls

get_cl_ccl(pars, ell_bp, z_1, pz_1)
get_cl_ccl(pars, ell_bp, z, pz)

#print(cov_th)
#print(cov_sim.shape)
#print(ell_bp)

exit(1)

nell=30000
dx = 0.1

factor = (20000-len(cov_sim)-2.)/(20000.-1.)
inv_cov_th = np.linalg.inv(cov_th)
inv_cov_sim = factor*np.linalg.inv(cov_sim)

factor_sph = (20000-len(cov_sim_sph)-2.)/(20000.-1.)
inv_cov_th_sph = np.linalg.inv(cov_th_sph)
inv_cov_sim_sph = factor_sph*np.linalg.inv(cov_sim_sph)


z, pz = np.loadtxt('./nz.txt', unpack=True)

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
