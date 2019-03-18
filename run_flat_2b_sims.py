from __future__ import print_function
from optparse import OptionParser
import pymaster as nmt
import numpy as np
import matplotlib.pyplot as plt
import flatmaps as fm
import templates as tp
import os
import sys

def opt_callback(option, opt, value, parser):
        setattr(parser.values, option.dest, value.split(','))
parser = OptionParser()
parser.add_option('--prefix-out',dest='prefix_out',default='run',type=str,
                  help='Output prefix')
parser.add_option('--isim-ini', dest='isim_ini', default=1, type=int,
                  help='Index of first simulation')
parser.add_option('--isim-end', dest='isim_end', default=100, type=int,
                  help='Index of last simulation')
parser.add_option('--nls-contaminants', dest='nls_cont', default=0, type=int,
                  help='Number of Large Scales contaminants')
parser.add_option('--nss-contaminants', dest='nss_cont', default=0, type=int,
                  help='Number of Small Scales contaminants')
parser.add_option('--wo-contaminants', dest='wo_cont', default=False, action='store_true',
                  help='Set if you don\'t want to use contaminants (ignore for now)')
parser.add_option('--plot', dest='plot_stuff', default=False, action='store_true',
                  help='Set if you want to produce plots')
parser.add_option('--no-deproject',dest='no_deproject',default=False,action='store_true',
                  help='Set if you will include contaminants but won\'t clean them (ignore for now)')
parser.add_option('--no-debias',dest='no_debias',default=False,action='store_true',
                  help='Set if you will include contaminants, clean them but won\'t correct for the bias (ignore for now)')
parser.add_option('--low-noise-ee-bb',dest='low_noise_ee_bb',default=False,action='store_true',
                  help='Set if you want the noise for ee and bb modes be multiplied by 1e-2')
(o, args) = parser.parse_args()

nsims=o.isim_end-o.isim_ini+1

#Read input power spectra
f = np.load("data/cls_lss_2bins.npz")
l = f['ls']
cls_ar = f['cls'][np.triu_indices(f['cls'].shape[0])]
nls_ar = f['nls'][np.triu_indices(f['cls'].shape[0])]

if o.low_noise_ee_bb:
    nlee *= 1e-2
    nlbb *= 1e-2

if o.plot_stuff :
    plt.figure()
    plt.plot(l,cltt,'r-',label='TT')
    plt.plot(l,clee,'b-',label='EE')
    plt.plot(l,clbb,'g-',label='BB')
    plt.plot(l,clte,'y-',label='TE')
    plt.loglog()
    plt.legend()

#Read mask
fmi,mask_hsc=fm.read_flat_map("data/mask_lss_flat.fits")
if o.plot_stuff :
    fmi.view_map(mask_hsc)

#Set up binning scheme
ell_min=max(2*np.pi/fmi.lx_rad,2*np.pi/fmi.ly_rad)
ell_max=min(fmi.nx*np.pi/fmi.lx_rad,fmi.ny*np.pi/fmi.ly_rad)
d_ell=2*ell_min
n_ell=int((ell_max-ell_min)/d_ell)-1
l_bpw=np.zeros([2,n_ell])
l_bpw[0,:]=ell_min+np.arange(n_ell)*d_ell
l_bpw[1,:]=l_bpw[0,:]+d_ell
b=nmt.NmtBinFlat(l_bpw[0,:],l_bpw[1,:])

#Generate an initial simulation
def get_fields(fsk,mask,w_cont=False) :
    """
    Generate a simulated field.
    It returns two NmtField objects for a spin-0 and a spin-2 field.

    :param fsk: a fm.FlatMapInfo object.
    :param mask: a sky mask.
    :param w_cont: deproject any contaminants? (not implemented yet)
    """
    nbins = 2
    spins = [0,2] * nbins
    # maps == [st1, sq1, su1, st2, sq2, su2, ...] (oredered as in spins)
    maps = nmt.synfast_flat(int(fsk.nx),int(fsk.ny),fsk.lx_rad,fsk.ly_rad,
                              cls_ar + nls_ar, spins)
    st1, sq1, su1, st2, sq2, su2 = maps

    if w_cont :
        raise ValueError('Contaminants not implemented yet')
        # if np.any(templates_all):
        #     tst, tsq, tsu = templates_all.sum(axis=0)
        #     st+=tst; sq+=tsq; su+=tsu;
        # if o.no_deproject :
        #     ff0=nmt.NmtFieldFlat(fsk.lx_rad,fsk.ly_rad,mask.reshape([fsk.ny,fsk.nx]),
        #                          [st])
        #     ff2=nmt.NmtFieldFlat(fsk.lx_rad,fsk.ly_rad,mask.reshape([fsk.ny,fsk.nx]),
        #                          [sq, su])
        # else :
        #     ff0=nmt.NmtFieldFlat(fsk.lx_rad,fsk.ly_rad,mask.reshape([fsk.ny,fsk.nx]),
        #                          [st],
        #                          templates=templates_all[:,0,None,:])
        #     ff2=nmt.NmtFieldFlat(fsk.lx_rad,fsk.ly_rad,mask.reshape([fsk.ny,fsk.nx]),
        #                          [sq,su],
        #                          templates=templates_all[:,1:, :])
    else :
        ff0_1=nmt.NmtFieldFlat(fsk.lx_rad, fsk.ly_rad, mask.reshape([fsk.ny, fsk.nx]),
                             [st1])
        ff0_2=nmt.NmtFieldFlat(fsk.lx_rad, fsk.ly_rad, mask.reshape([fsk.ny, fsk.nx]),
                             [st2])
        ff2_1=nmt.NmtFieldFlat(fsk.lx_rad, fsk.ly_rad, mask.reshape([fsk.ny, fsk.nx]),
                             [sq1, su1])
        ff2_2=nmt.NmtFieldFlat(fsk.lx_rad, fsk.ly_rad, mask.reshape([fsk.ny, fsk.nx]),
                             [sq2, su2])
    return (ff0_1,ff2_1), (ff0_2, ff2_2)

def get_coupling_matrix(w, f0, f1, suffix):
    """
    :param w:  NmtWorkspaceFlat initialized
    :param f0: field 0 to correlate with f1
    :param f1: field 1 to correlate with f0
    :param suffix: field suffix where coupling matrix is stored.
    """
    if not os.path.isfile(o.prefix_out + suffix):
        w.compute_coupling_matrix(f0, f1, b)
        w.write_to(o.prefix_out + suffix);
    else:
        w.read_from(o.prefix_out + suffix)

def get_workspaces(fields):
    """
    :param fields: tuple of tuple of fields to compute the mode-coupling matrix for. Shape is nbis x (f0, f2)
    """
    spins = [0, 2] * len(fields)

    ws = []

    fields = sum(fields, ())  # Flatten the tuple of tuples

    zbin1 = 1
    c1 = 0
    for f1, s1 in zip(fields, spins):
        c2 = c1
        zbin2 = int(c2 / 2) + 1

        for f2, s2 in zip(fields[c1:], spins[c1:]):
            print("Computing {}{}_{}{}".format(s1, s2, zbin1, zbin2))
            suffix = "_w{}{}_{}{}.dat".format(s1, s2, zbin1, zbin2)
            w = nmt.NmtWorkspaceFlat()
            get_coupling_matrix(w, f1, f2, suffix)
            ws.append(w)
            if not ((c2 + 1) % 2):
                zbin2 += 1
            c2 += 1

        if not ((c1 + 1) % 2):
            zbin1 += 1

        c1 +=1

    return ws

def get_cls_th(ws, ells, cl_th):
    """
    :param ws: workspaces for fields in fields.
    :param fields: tuple of tuple of fields to compute the mode-coupling matrix for. Shape is nbis x (f0, f2)
    """
    dof = [1, 2] * int(cl_th.shape[0] / 3)

    cl_ar = np.empty((cl_th.shape[0], cl_th.shape[0], b.get_n_bands()))

    index1 = 0
    c = 0
    print(dof)
    for c1, dof1 in enumerate(dof):
        index2 = index1
        for dof2 in dof[c1:]:
            cls_true = cl_th[index1 : index1 + dof1, index2 : index2 + dof2].reshape(dof1 * dof2, -1)
            cls = ws[c].decouple_cell(ws[c].couple_cell(ells, cls_true)).reshape((dof1, dof2, -1))

            # from matplotlib import pyplot as plt
            # for cli_true, cli in zip(cls_true, cls):
            #     plt.suptitle("{}, {}".format(dof1, dof2))
            #     plt.loglog(ells, cli_true, b.get_effective_ells(), cli, 'o')
            #     plt.show()
            #     plt.close()

            cl_ar[index1 : index1 + dof1, index2 : index2 + dof2] = cls

            # from matplotlib import pyplot as plt
            # for cli_true, cli in zip(cls_true,
            #                          cl_ar[index1 : index1 + dof1, index2 : index2 + dof2].reshape(dof1 * dof2, -1)):
            #     plt.suptitle("{}, {}".format(dof1, dof2))
            #     plt.loglog(ells, cli_true, b.get_effective_ells(), cli, 'o')
            #     plt.show()
            #     plt.close()

            index2 += dof2
            c += 1
        index1 += dof1

    return cl_ar[np.triu_indices(cl_ar.shape[0])]

def get_cls_sim(ws, fields):
    """
    :param ws: workspaces for fields in fields.
    :param fields: tuple of tuple of fields to compute the mode-coupling matrix for. Shape is nbis x (f0, f2)
    """
    dof = [1, 2] * len(fields)

    nfs = 3 * len(fields)

    fields = sum(fields, ())  # Flatten the tuple of tuples

    cl_ar = np.empty((nfs, nfs, b.get_n_bands()))

    index1 = 0
    c = 0
    for c1, dof1 in enumerate(dof):
        f1 = fields[c1]
        index2 = index1
        for c2, dof2 in enumerate(dof[c1:], c1):
            f2 = fields[c2]
            cls = ws[c].decouple_cell(nmt.compute_coupled_cell_flat(f1, f2, b)).reshape((dof1, dof2, -1))

            # from matplotlib import pyplot as plt
            # cls_true = (f['cls'] + f['nls'])[index1 : index1 + dof1, index2 : index2 + dof2].reshape(dof1 * dof2, -1)
            # print(cls_true.shape)
            # print(cls.reshape(dof1 * dof2, -1).shape)
            # for cli_true, cli in zip(cls_true, cls.reshape(dof1 * dof2, -1)):
            #     print(cli)
            #     plt.suptitle("{}, {}".format(dof1, dof2))
            #     plt.loglog(l, cli_true, b.get_effective_ells(), cli, 'o')
            #     plt.show()
            #     plt.close()

            cl_ar[index1 : index1 + dof1, index2 : index2 + dof2] = cls

             # from matplotlib import pyplot as plt
             # for cli_true, cli in zip(cls_true,
             #                          cl_ar[index1 : index1 + dof1, index2 : index2 + dof2].reshape(dof1 * dof2, -1)):
             #     plt.suptitle("{}, {}".format(dof1, dof2))
             #     plt.loglog(l, cli_true, b.get_effective_ells(), cli, 'o')
             #     plt.show()
             #     plt.close()

            index2 += dof2
            c += 1
        index1 += dof1

    return cl_ar[np.triu_indices(cl_ar.shape[0])]

############## Generate fields #####################
np.random.seed(1000)
fields = get_fields(fmi, mask_hsc) #, o.nss_cont or o.nls_cont)
fbin1,  fbin2 = fields
workspaces = get_workspaces(fields)

############## Generate theory prediction #####################
if not os.path.isfile(o.prefix_out+'_cl_th.npz') :
    print("Computing theory prediction")
    cl_ar = get_cls_th(workspaces, l, f['cls'])
    np.savez_compressed(o.prefix_out+"_cl_th.npz",
                        ls=b.get_effective_ells(), cls=cl_ar)

############## Generate simulations #####################
#Compute mean and variance over nsims simulations
for i in np.arange(nsims):
    print("%d-th sim"%(i+o.isim_ini))
    if not os.path.isfile(o.prefix_out+"_cl_%04d.npz"%(o.isim_ini+i)):
        cl_ar = get_cls_sim(workspaces, fields)
        np.savez(o.prefix_out+"_cl_%04d"%(o.isim_ini + i),
                 l=b.get_effective_ells(), cls=cl_ar)


#    cld=np.load(o.prefix_out+"_cl_%04d.npz"%(o.isim_ini+i))
#     cl00_all.append([cld['cltt']])
#     cl02_all.append([cld['clte'],cld['cltb']])
#     cl22_all.append([cld['clee'],cld['cleb'],cld['clbe'],cld['clbb']])
#
# cl00_all=np.array(cl00_all)
# cl02_all=np.array(cl02_all)
# cl22_all=np.array(cl22_all)

# np.savez(o.prefix_out+'_clsims_%04d-%04d'%(o.isim_ini,o.isim_end),
#          l=b.get_effective_ells(),cl00=cl00_all,cl02=cl02_all,cl22=cl22_all)

if o.plot_stuff :
    plt.show()
