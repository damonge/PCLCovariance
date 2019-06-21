import matplotlib.pyplot as plt
import numpy as np
import os
import re, pickle


l1s = 2.30
l2s = 6.18


files=np.array(['output/data_ref/chi2_Omega_c_sigma8_ng_100_100_lmin_0_nb_2_cov_sim.pickle',
                'output/data_ref/chi2_Omega_c_sigma8_ng_100_100_lmin_0_nb_2_cov_th.pickle'])
folders = np.array([os.path.abspath(os.path.dirname(x)) for x in files])
sets = np.array([pickle.load(open(fn,'rb')) for fn in files])
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

def get_mean_width(grid,rangs):
    p=np.exp(-0.5*grid)
    p/=np.sum(p)
    m1=np.sum(p*rangs[0][None,:])
    m2=np.sum(p*rangs[1][:,None])
    m12=np.sum(p*rangs[0][None,:]**2)
    m22=np.sum(p*rangs[1][:,None]**2)
    s1=np.sqrt(m12-m1**2)
    s2=np.sqrt(m22-m2**2)
    return [s1,s2]
sigmas=np.array([get_mean_width(g,r) for g,r in zip(grids,ranges)])
print(np.diff(sigmas,axis=0)/np.mean(sigmas,axis=0)*100)

n_plots = grids.shape[0]

fig, ax = plt.subplots()
h = []
cntr = ax.contourf(ranges[1][0], ranges[1][1], grids[1], [0,l1s,l2s],cmap=plt.cm.autumn)
h.append(cntr.legend_elements()[0][0])
cntr = ax.contour(ranges[0][0], ranges[0][1], grids[0], [l1s,l2s], alpha=1., colors='k', linestyles='dashed')
h.append(cntr.legend_elements()[0][0])
legend=['NKA','Sims']
ax.set_xticks([0.267,0.269,0.271,0.273])
ax.set_yticks([0.838,0.840,0.842])
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(13)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(13)
ax.legend(h, legend,fontsize=15,frameon=False)
ax.set_xlim([0.2665,0.2735])
ax.set_ylim([0.8375,0.8435])
ax.set_xlabel('$\\Omega_{\\rm cdm}$',fontsize=15)
ax.set_ylabel('$\\sigma_8$',fontsize=15)
plt.savefig("contours_2bin.pdf",bbox_inches='tight')
plt.show()
