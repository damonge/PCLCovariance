import os, sys
import re, pickle
import numpy as np
import get_grid_tools as tools

# Call the parser
args = tools.argument_parser()


# Run grid
if args.mode=='run':


    # Folders
    data_dir   = os.path.abspath(args.data)
    output_dir = os.path.abspath(args.output)

    # Parameters
    n_bins       = args.n_bins
    cov_m    = args.cov_m
    ell_min  = args.ell_min
    var_pars = np.array(args.pars)

    if len(args.n_grid)==len(var_pars):
        n_grid  = args.n_grid
    elif len(args.n_grid)==1:
        n_grid = np.full(var_pars.shape,args.n_grid[0])
    else:
        raise IOError('n_grid should either contain one element or an array with the same dimension of params!')

    # Initialize varying parameters
    all_pars = tools.default_pars.copy()
    is_s8 = 'sigma8' in var_pars
    is_As = 'ln10_A_s' in var_pars
    if is_s8 and is_As:
        raise IOError('sigma8 and ln10_A_s can not be varied at the same time. Choose one of the two!')
    elif is_s8:
        all_pars.pop('ln10_A_s',None)
    else:
        all_pars.pop('sigma8',None)

    # Create name file output
    s1 = '_'.join(var_pars)
    s2 = '_'.join([str(i) for i in n_grid])
    grid_fn = '{}/chi2_{}_ng_{}_lmin_{}_nb_{}_cov_{}.npz'.format(output_dir,s1,s2,ell_min,n_bins,cov_m)

    # Load data
    data, data_ref = tools.load_data(data_dir,cov_m,n_bins,ell_min)

    # Get range of variation for each parameter
    range_m_M = tools.get_range(var_pars,all_pars,data_ref)

    sets = {
        'n_bins'         : n_bins,
        'cov_m'          : cov_m,
        'ell_min'        : ell_min,
        'var_pars'       : var_pars,
        'n_grid'         : n_grid,
        'all_pars'       : all_pars,
        'grid_fn'        : os.path.basename(grid_fn),
        'range_m_M'      : range_m_M
    }

    # Save settings to file
    sets_out = open(grid_fn.replace('.npz','.pickle'),'wb')
    pickle.dump(sets, sets_out)
    sets_out.close()


    print('')
    print('Starting grid:')
    print('----> Covariance with    : {}'.format(cov_m))
    print('----> Number of bins     : {}'.format(n_bins))
    print('----> Minimum ell        : {}'.format(ell_min))
    print('----> Fixed parameters   :')
    for npar, par in enumerate(np.setdiff1d(all_pars.keys(),var_pars)):
        print('--------> {:10} : {}'.format(par,all_pars[par]))
    print('----> Varying parameters (grid {}):'.format(' x '.join([str(i) for i in n_grid])))
    for npar, par in enumerate(var_pars):
        print('--------> {:10} : [{}, {}]'.format(par,range_m_M[npar,0],range_m_M[npar,1]))
    print('')
    sys.stdout.flush()


    # Get grid
    grid = tools.get_grid(var_pars,all_pars,data,range_m_M,n_grid)
    np.savez_compressed(grid_fn, grid)

    print('')
    print('Saving grid to : {}'.format(grid_fn))
    print('Success!!')
    sys.stdout.flush()






elif args.mode=='info':

    import matplotlib.pyplot as plt
    # from matplotlib import colors as mcolors
    #
    # colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    # by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
    #             for name, color in colors.items())
    # sorted_names = [name for hsv, name in by_hsv]
    # print(colors[sorted_names[0]])

    l1s = 2.30
    l2s = 6.18


    # Read arguments
    files = np.array([x for x in np.unique(args.params_file) if re.match('.+\.pickle',x)])
    folders = np.array([os.path.abspath(os.path.dirname(x)) for x in files])
    sets = np.array([pickle.load(open(fn,'rb')) for fn in files])

    #Prepare legend
    legend = np.array(['{}/{}'.format(folders[x],os.path.basename(sets[x]['grid_fn'])) for x in range(len(sets))])
    legend = np.array([x.replace('/','_') for x in legend])
    legend = np.array([x.replace('.','_') for x in legend])
    legend = np.array([x.split('_') for x in legend])
    legend = legend[:, ~np.all(legend[1:] == legend[:-1], axis=0)]
    legend = np.array(['_'.join(tuple(x)) for x in legend])


    def prepare_ranges(settings):
        ranges = settings['range_m_M']
        ng = settings['n_grid']
        ranges = np.array([np.linspace(ranges[x,0],ranges[x,1],ng[x]) for x in range(len(ng))])
        return ranges

    def prepare_grids(settings, folder):
        fn = '{}/{}'.format(folder,os.path.basename(settings['grid_fn']))
        grid = np.load(fn)['arr_0']
        grid = grid - grid.min()
        return grid


    ranges = np.array([prepare_ranges(s) for s in sets])

    grids = np.array([prepare_grids(sets[x],folders[x]) for x in range(len(sets))])

    n_plots = grids.shape[0]

    colors = ['k', 'r', 'b', 'g', 'y', 'm', 'c']

    linestyles = ['solid', 'dashed', 'dotted', 'dotted', 'dotted']
    #linestyles = np.array(['solid' for x in range(n_plots)])
    #linestyles[1] = 'dashed'

    fig, ax = plt.subplots()
    h = []
    for i in range(n_plots):
        cntr = ax.contour(ranges[i][0], ranges[i][1], grids[i], [l1s,l2s], alpha=1., colors=colors[i], linestyles=linestyles[i])
        h.append(cntr.legend_elements()[0][0])
    ax.legend(h, legend)
    # plt.show()
    plt.savefig('{}/2_bin.pdf'.format(folders[0]))
