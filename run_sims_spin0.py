from __future__ import print_function
from optparse import OptionParser
import pymaster as nmt
import numpy as np
import matplotlib.pyplot as plt
import flatmaps as fm
import os

def opt_callback(option, opt, value, parser):
        setattr(parser.values, option.dest, value.split(','))
parser = OptionParser()
parser.add_option('--prefix-out',dest='prefix_out',default='run',type=str,
                  help='Output prefix')
parser.add_option('--isim-ini', dest='isim_ini', default=1, type=int,
                  help='Index of first simulation')
parser.add_option('--isim-end', dest='isim_end', default=100, type=int,
                  help='Index of last simulation')
parser.add_option('--wo-contaminants', dest='wo_cont', default=False, action='store_true',
                  help='Set if you don\'t want to use contaminants (ignore for now)')
parser.add_option('--plot', dest='plot_stuff', default=False, action='store_true',
                  help='Set if you want to produce plots')
parser.add_option('--no-deproject',dest='no_deproject',default=False,action='store_true',
                  help='Set if you will include contaminants but won\'t clean them (ignore for now)')
parser.add_option('--no-debias',dest='no_debias',default=False,action='store_true',
                  help='Set if you will include contaminants, clean them but won\'t correct for the bias (ignore for now)')
(o, args) = parser.parse_args()
        

#Read input power spectra
l,cltt,clee,clbb,clte,nltt,nlee,nlbb,nlte=np.loadtxt("data/cls_lss.txt",unpack=True)
cltt[0]=0; clee[0]=0; clbb[0]=0; clte[0]=0;
nltt[0]=0; nlee[0]=0; nlbb[0]=0; nlte[0]=0;

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
    st,sq,su=nmt.synfast_flat(int(fsk.nx),int(fsk.ny),fsk.lx_rad,fsk.ly_rad,
                              [cltt+nltt,clte+nlte,0*cltt,clee+nlee,0*clee,clbb+nlbb],[0,2])
    st=st.flatten(); sq=sq.flatten(); su=su.flatten()
    if w_cont :
        raise NotImplemented("Not yet")
        st+=np.sum(fgt,axis=0)[0,:]; sq+=np.sum(fgp,axis=0)[0,:]; su+=np.sum(fgp,axis=0)[1,:];
        if o.no_deproject :
            ff0=nmt.NmtFieldFlat(fsk.lx_rad,fsk.ly_rad,mask.reshape([fsk.ny,fsk.nx]),
                                 [st.reshape([fsk.ny,fsk.nx])])
            ff2=nmt.NmtFieldFlat(fsk.lx_rad,fsk.ly_rad,mask.reshape([fsk.ny,fsk.nx]),
                                 [sq.reshape([fsk.ny,fsk.nx]),su.reshape([fsk.ny,fsk.nx])])
        else :
            ff0=nmt.NmtFieldFlat(fsk.lx_rad,fsk.ly_rad,mask.reshape([fsk.ny,fsk.nx]),
                                 [st.reshape([fsk.ny,fsk.nx])],
                                 templates=fgt.reshape([2,1,fsk.ny,fsk.nx]))
            ff2=nmt.NmtFieldFlat(fsk.lx_rad,fsk.ly_rad,mask.reshape([fsk.ny,fsk.nx]),
                                 [sq.reshape([fsk.ny,fsk.nx]),su.reshape([fsk.ny,fsk.nx])],
                                 templates=fgp.reshape([2,2,fsk.ny,fsk.nx]))
    else :
        ff0=nmt.NmtFieldFlat(fsk.lx_rad,fsk.ly_rad,mask.reshape([fsk.ny,fsk.nx]),
                             [st.reshape([fsk.ny,fsk.nx])])
        ff2=nmt.NmtFieldFlat(fsk.lx_rad,fsk.ly_rad,mask.reshape([fsk.ny,fsk.nx]),
                             [sq.reshape([fsk.ny,fsk.nx]),su.reshape([fsk.ny,fsk.nx])])
    return ff0,ff2

np.random.seed(1000)
f0,f2=get_fields(fmi,mask_hsc)

#Compute mode-coupling matrix
#Use initial fields to generate coupling matrix
w00=nmt.NmtWorkspaceFlat();
if not os.path.isfile(o.prefix_out+"_w00.dat") : #spin0-spin0
    print("Computing 00")
    w00.compute_coupling_matrix(f0,f0,b)
    w00.write_to(o.prefix_out+"_w00.dat");
else :
    w00.read_from(o.prefix_out+"_w00.dat")
w02=nmt.NmtWorkspaceFlat();
if not os.path.isfile(o.prefix_out+"_w02.dat") : #spin0-spin2
    print("Computing 02")
    w02.compute_coupling_matrix(f0,f2,b)
    w02.write_to(o.prefix_out+"_w02.dat");
else :
    w02.read_from(o.prefix_out+"_w02.dat")
w22=nmt.NmtWorkspaceFlat();
if not os.path.isfile(o.prefix_out+"_w22.dat") : #spin2-spin2
    print("Computing 22")
    w22.compute_coupling_matrix(f2,f2,b)
    w22.write_to(o.prefix_out+"_w22.dat");
else :
    w22.read_from(o.prefix_out+"_w22.dat")

#Generate theory prediction
if not os.path.isfile(o.prefix_out+'_cl_th.txt') :
    print("Computing theory prediction")
    cl00_th=w00.decouple_cell(w00.couple_cell(l,np.array([cltt])))
    cl02_th=w02.decouple_cell(w02.couple_cell(l,np.array([clte,0*clte])))
    cl22_th=w22.decouple_cell(w22.couple_cell(l,np.array([clee,0*clee,0*clbb,clbb])))
    np.savetxt(o.prefix_out+"_cl_th.txt",
               np.transpose([b.get_effective_ells(),cl00_th[0],cl02_th[0],cl02_th[1],
                             cl22_th[0],cl22_th[1],cl22_th[2],cl22_th[3]]))
else :
    cl00_th=np.zeros([1,b.get_n_bands()])
    cl02_th=np.zeros([2,b.get_n_bands()])
    cl22_th=np.zeros([4,b.get_n_bands()])
    dum,cl00_th[0],cl02_th[0],cl02_th[1],cl22_th[0],cl22_th[1],cl22_th[2],cl22_th[3]=np.loadtxt(o.prefix_out+"_cl_th.txt",unpack=True)
    

if o.plot_stuff :
    plt.show()
