import os, sys
import argparse
import numpy as np
import pyccl as ccl


default_pars = {
    'h'               :  0.67,
    'Omega_c'         :  0.27,
    'Omega_b'         :  0.045,
    'ln10_A_s'        :  3.044522438,
    'sigma8'          :  0.840421163375,
    'n_s'             :  0.96,
    'w_0'             : -1.0,
    'w_a'             :  0.0
}


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


def get_cosmo_ccl(pars):
    try:
        cosmo = ccl.Cosmology(
            h        = pars['h'],
            Omega_c  = pars['Omega_c'],
            Omega_b  = pars['Omega_b'],
            A_s      = (10.**(-10.))*np.exp(pars['ln10_A_s']),
            n_s      = pars['n_s'],
            w0       = pars['w_0'],
            wa       = pars['w_a']
            )
    except:
        cosmo = ccl.Cosmology(
            h        = pars['h'],
            Omega_c  = pars['Omega_c'],
            Omega_b  = pars['Omega_b'],
            sigma8   = pars['sigma8'],
            n_s      = pars['n_s'],
            w0       = pars['w_0'],
            wa       = pars['w_a']
            )
    ccl.linear_matter_power(cosmo,0.1,0.5)
    return cosmo


def get_tracers_ccl(cosmo, data):
    n_bins = data['pz'].shape[0]
    # Tracers
    tracers = []
    for i in range(n_bins):
        tracers.append(
            ccl.NumberCountsTracer(cosmo,has_rsd=False,dndz=(data['z'][i],data['pz'][i]),bias=(data['z'][i],data['bz'][i]))
            )
        tracers.append(
            ccl.WeakLensingTracer(cosmo,dndz=(data['z'][i],data['pz'][i]))
            )
    return np.array(tracers)


def get_cls_ccl(cosmo, tracers, data):
    n_bte = tracers.shape[0]
    n_ells = len(data['ell_bp'])
    cls = np.zeros([n_bte, n_bte, n_ells])
    for c1 in range(n_bte): # c1=te1+b1*n_te
        for c2 in range(c1, n_bte):
            cls[c1,c2,:] = ccl.angular_cl(cosmo,tracers[c1],tracers[c2],data['ell_bp'])
            cls[c2,c1,:] = cls[c1,c2,:]
    cls_flat = flatten_cls(cls, n_bte, n_ells)
    return cls_flat


def diff_cls(par,all_pars,data,dx=0.01):
    if all_pars[par] == 0.:
        Dx = dx
    else:
        Dx = dx*np.abs(all_pars[par])
    parstmp = all_pars.copy()
    parstmp[par] = all_pars[par]+Dx
    cosmo = get_cosmo_ccl(parstmp)
    tracers = get_tracers_ccl(cosmo, data)
    clP = get_cls_ccl(cosmo, tracers, data)
    parstmp[par] = all_pars[par]-Dx
    cosmo = get_cosmo_ccl(parstmp)
    tracers = get_tracers_ccl(cosmo, data)
    clM = get_cls_ccl(cosmo, tracers, data)
    return (clP-clM)/(2.*Dx)


def get_fisher(par,all_pars,data,dx=0.01):
    diff = diff_cls(par,all_pars,data,dx=dx)
    fisher_1 = diff.dot(data['icov']).dot(diff)
    diff = diff_cls(par,all_pars,data,dx=2.*dx)
    fisher_2 = diff.dot(data['icov']).dot(diff)
    if np.abs(fisher_1/fisher_2-1.)>2.e-3:
        print('WARNING: decrease dx in the calculation of the Fisher matrix to stabilize the result (rel_diff for {} = {})'.format(par,np.abs(fisher_1/fisher_2-1.)))
    return fisher_1


def get_chi2(pars, data):
    # Get theory Cls
    cosmo = get_cosmo_ccl(pars)
    tracers = get_tracers_ccl(cosmo, data)
    theory = get_cls_ccl(cosmo, tracers, data)
    chi2 = (data['cls']-theory).dot(data['icov']).dot(data['cls']-theory)
    return chi2


def get_range(var_pars,all_pars,data,f=6.,dx=0.01):
    range_m_M = np.zeros((len(var_pars),2))
    for npar,par in enumerate(var_pars):
        delta = f*get_fisher(par, all_pars, data, dx=dx)**(-1./2.)
        range_m_M[npar] = all_pars[par]-delta, all_pars[par]+delta
    return range_m_M


def get_idx_list(shape):
    flat = np.arange(np.prod(shape))
    unflat = flat.reshape(shape)
    idx_list = np.array([np.array([np.where(unflat==x)]).flatten() for x in flat])
    return idx_list


def get_grid(var_pars,all_pars,data,range_m_M,n_grid):
    idx_list = get_idx_list(tuple(n_grid))
    grid = np.zeros(tuple(n_grid))
    parstmp = all_pars.copy()
    var_range = np.array([np.linspace(range_m_M[x,0],range_m_M[x,1],n_grid[x]) for x in range(len(var_pars))])
    for n_idx, idx in enumerate(idx_list):
        for n, i in enumerate(idx):
            parstmp[var_pars[n]] = var_range[n][i]
        try:
            grid[tuple(idx)] = get_chi2(parstmp, data)
        except:
            grid[tuple(idx)] = 1.e30
        if np.mod(n_idx+1,100)==0:
            print('Done {} points over {}'.format(n_idx+1,len(idx_list)))
            sys.stdout.flush()
    return grid


def load_data(data_dir,cov_m,n_bins,ell_min,n_sims=20000,n_bins_ref=1,cov_m_ref='sim'):
    data = {}
    data_ref = {}
    # Load photo_z
    data_ref['z']  = np.load('{}/z_{}.npz'.format(data_dir,n_bins_ref))['arr_0']
    data_ref['pz'] = np.load('{}/pz_{}.npz'.format(data_dir,n_bins_ref))['arr_0']
    data_ref['bz'] = np.load('{}/bz_{}.npz'.format(data_dir,n_bins_ref))['arr_0']
    # Load photo_z
    data['z']  = np.load('{}/z_{}.npz'.format(data_dir,n_bins))['arr_0']
    data['pz'] = np.load('{}/pz_{}.npz'.format(data_dir,n_bins))['arr_0']
    data['bz'] = np.load('{}/bz_{}.npz'.format(data_dir,n_bins))['arr_0']
    # Load reference ell bandpowers
    data_ref['ell_bp'] = np.load('{}/ell_bp.npz'.format(data_dir))['lsims'].astype(int)
    # Load ell bandpowers
    data['ell_bp'] = data_ref['ell_bp'][data_ref['ell_bp']>=ell_min]
    start_ell = len(data_ref['ell_bp'])-len(data['ell_bp'])
    # Load data
    arr = np.load('{}/cls_{}.npz'.format(data_dir,n_bins))['arr_0']
    arr = unflatten_cls(arr, 2*n_bins, len(data_ref['ell_bp']))
    arr = arr[:,:,start_ell:]
    data['cls'] = flatten_cls(arr, 2*n_bins, len(data['ell_bp']))
    # Load reference covariance matrix
    arr = np.load('{}/cov_{}_{}.npz'.format(data_dir,cov_m_ref,n_bins_ref))['arr_0']
    factor = (n_sims-arr.shape[0]-2.)/(n_sims-1.)
    data_ref['icov'] = factor*np.linalg.inv(arr)
    # Load covariance matrix
    arr = np.load('{}/cov_{}_{}.npz'.format(data_dir,cov_m,n_bins))['arr_0']
    arr = unflatten_covmat(arr, 2*n_bins, len(data_ref['ell_bp']))
    arr = arr[:,:,start_ell:,:,:,start_ell:]
    arr = flatten_covmat(arr, 2*n_bins, len(data['ell_bp']))
    if cov_m=='sim':
        factor = (n_sims-arr.shape[0]-2.)/(n_sims-1.)
    else:
        factor = 1.
    # factor = 1.#TODO:remove this
    data['icov'] = factor*np.linalg.inv(arr)
    return data, data_ref


def argument_parser():

    parser = argparse.ArgumentParser(
    'Run grid and get plots.'
    )

    #Add supbarser to select between run and prep modes.
    subparsers = parser.add_subparsers(dest='mode',
        help='Options are: '
        '(i) run: run grid. '
        '(ii) info: get plots. ')

    def check_positive(value):
        ivalue = int(value)
        if ivalue < 0:
            raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
        return ivalue

    run_parser = subparsers.add_parser('run')
    info_parser = subparsers.add_parser('info')

    #Arguments for 'run'
    run_parser.add_argument('--n_bins', '-nb', type=int, choices=[1, 2],
        default=1, help='Number of bins')
    run_parser.add_argument('--cov_m', '-c', type=str, choices=['sim', 'th', 'spin0'],
        default='sim', help='Covariance matrix used')
    run_parser.add_argument('--n_grid', '-ng', nargs='+', type=check_positive,
        default=[10], help='Number of points in grid')
    run_parser.add_argument('--ell_min', '-lm', type=check_positive,
        default=0, help='Ell min used to get constraints')
    run_parser.add_argument('--data', '-d', type=str,
        default='data/', help='Where the data are stored')
    run_parser.add_argument('--output', '-o', type=str,
        default='output/', help='Where to store the output')
    run_parser.add_argument('--pars', '-p', nargs='+', type=str,
        choices=['h', 'Omega_c', 'Omega_b', 'ln10_A_s', 'sigma8', 'n_s', 'w_0', 'w_a'],
        default=['Omega_c','sigma8'], help='List of cosmological parameters to vary')
    #Arguments for 'info'
    info_parser.add_argument('params_file', nargs='+', type=str, help='Parameters file')

    return parser.parse_args()
