from __future__ import print_function
from optparse import OptionParser
import pymaster as nmt
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import templates as tp
import os

def opt_callback(option, opt, value, parser):
        setattr(parser.values, option.dest, value.split(','))
parser = OptionParser()
parser.add_option('--prefix-out',dest='prefix_out',default='run',type=str,
                  help='Output prefix')
parser.add_option('--nside', dest='nside', default=512, type=int,
                  help='HEALPix nside param')
parser.add_option('--isim-ini', dest='isim_ini', default=1, type=int,
                  help='Index of first simulation')
parser.add_option('--isim-end', dest='isim_end', default=100, type=int,
                  help='Index of last simulation')
parser.add_option('--wo-contaminants', dest='wo_cont', default=False, action='store_true',
                  help='Set if you don\'t want to use contaminants (ignore for now)')
parser.add_option('--nls-contaminants', dest='nls_cont', default=0, type=int,
                  help='Number of Large Scales contaminants')
parser.add_option('--nss-contaminants', dest='nss_cont', default=0, type=int,
                  help='Number of Small Scales contaminants')
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
l,cltt,clee,clbb,clte,nltt,nlee,nlbb,nlte=np.loadtxt("data/cls_lss.txt",unpack=True)
cltt=cltt[:3*o.nside]; clee=clee[:3*o.nside]; clbb=clbb[:3*o.nside];
clte=clte[:3*o.nside];
nltt=nltt[:3*o.nside]; nlee=nlee[:3*o.nside]; nlbb=nlbb[:3*o.nside];
nlte=nlte[:3*o.nside];
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

if o.low_noise_ee_bb:
    nlee *= 1e-2
    nlbb *= 1e-2

#Read mask
mask_lss=hp.ud_grade(hp.read_map("data/mask_lss_sph.fits",verbose=False),nside_out=o.nside)
if o.plot_stuff :
    hp.mollview(mask_lss)

#Set up binning scheme
fsky=np.mean(mask_lss)
d_ell=int(1./fsky)
b=nmt.NmtBin(o.nside,nlb=d_ell)

#Read/Generate Large Scale contaminant fields
templates_ls = []
temp_ls_filename = o.prefix_out+"_cont_ls.npz"
print("%d Large Scale contaminant"%(o.nls_cont))
if os.path.isfile(temp_ls_filename):
    templates_ls = np.load(temp_ls_filename)['arr_0'][:o.nls_cont]

if len(templates_ls) < o.nls_cont:
    new_templates_ls = tp.create_templates(l, o.nside, cltt+nltt, clee+nlee, clbb+nlbb, N=o.nls_cont - len(templates_ls))
    if templates_ls == []:
        templates_ls = new_templates_ls
    else:
        templates_ls = np.concatenate((templates_ls, new_templates_ls))
    np.savez_compressed(temp_ls_filename, templates_ls)

#Read/Generate Small Scale contaminant fields
templates_ss = []
temp_ss_filename = o.prefix_out+"_cont_ss.npz"
print("%d Small Scale contaminant"%(o.nss_cont))
if os.path.isfile(temp_ss_filename):
    templates_ss = np.load(temp_ss_filename)['arr_0'][:o.nss_cont]

if len(templates_ss) < o.nss_cont:
    new_templates_ss = tp.create_templates(l, o.nside, cltt+nltt, clee+nlee, clbb+nlbb, N=o.nss_cont - len(templates_ss), exp_range=(0,0))
    if templates_ss == []:
        templates_ss = new_templates_ss
    else:
        templates_ls = np.concatenate((templates_ss, new_templates_ss))
    np.savez_compressed(temp_ss_filename, templates_ss)

# Sum LS + SS templates
if (templates_ls != []) and (templates_ss != []):
    templates_all = np.concatenate((templates_ls, templates_ss))
elif templates_ls != []:
    templates_all = templates_ls
elif templates_ss != []:
    templates_all = templates_ss
else:
    templates_all = np.array([])

#Generate an initial simulation
def get_fields(w_cont=False) :
    """
    Generate a simulated field.
    It returns two NmtField objects for a spin-0 and a spin-2 field.

    :param fsk: a fm.FlatMapInfo object.
    :param mask: a sky mask.
    :param w_cont: deproject any contaminants? (not implemented yet)
    """
    st,sq,su=hp.synfast([cltt+nltt,clee+nlee,clbb+nlbb,clte+nlte],o.nside,new=True,verbose=False,pol=True)
    if w_cont:
        if np.any(templates_all):
            tst, tsq, tsu = templates_all.sum(axis=0)
            st+=tst; sq+=tsq; su+=tsu;

        if o.no_deproject:
            ff0 = nmt.NmtField(mask_lss, [st])
            ff2 = nmt.NmtField(mask_lss, [sq, su])
        else:
            ff0 = nmt.NmtField(mask_lss, [st], templates_all[:, 0, None, :])
            ff2 = nmt.NmtField(mask_lss, [sq, su], templates_all[:, 1:, :])
    else :
        ff0=nmt.NmtField(mask_lss,[st])
        ff2=nmt.NmtField(mask_lss,[sq,su])
    return ff0,ff2

np.random.seed(1000)
f0,f2=get_fields()

#Compute mode-coupling matrix
#Use initial fields to generate coupling matrix
w00=nmt.NmtWorkspace();
if not os.path.isfile(o.prefix_out+"_w00.dat") : #spin0-spin0
    print("Computing 00")
    w00.compute_coupling_matrix(f0,f0,b)
    w00.write_to(o.prefix_out+"_w00.dat");
else :
    w00.read_from(o.prefix_out+"_w00.dat")
w02=nmt.NmtWorkspace();
if not os.path.isfile(o.prefix_out+"_w02.dat") : #spin0-spin2
    print("Computing 02")
    w02.compute_coupling_matrix(f0,f2,b)
    w02.write_to(o.prefix_out+"_w02.dat");
else :
    w02.read_from(o.prefix_out+"_w02.dat")
w22=nmt.NmtWorkspace();
if not os.path.isfile(o.prefix_out+"_w22.dat") : #spin2-spin2
    print("Computing 22")
    w22.compute_coupling_matrix(f2,f2,b)
    w22.write_to(o.prefix_out+"_w22.dat");
else :
    w22.read_from(o.prefix_out+"_w22.dat")

#Generate theory prediction
if not os.path.isfile(o.prefix_out+'_cl_th.txt') :
    print("Computing theory prediction")
    cl00_th=w00.decouple_cell(w00.couple_cell(np.array([cltt])))
    cl02_th=w02.decouple_cell(w02.couple_cell(np.array([clte,0*clte])))
    cl22_th=w22.decouple_cell(w22.couple_cell(np.array([clee,0*clee,0*clbb,clbb])))
    np.savetxt(o.prefix_out+"_cl_th.txt",
               np.transpose([b.get_effective_ells(),cl00_th[0],cl02_th[0],cl02_th[1],
                             cl22_th[0],cl22_th[1],cl22_th[2],cl22_th[3]]))
else :
    cl00_th=np.zeros([1,b.get_n_bands()])
    cl02_th=np.zeros([2,b.get_n_bands()])
    cl22_th=np.zeros([4,b.get_n_bands()])
    dum,cl00_th[0],cl02_th[0],cl02_th[1],cl22_th[0],cl22_th[1],cl22_th[2],cl22_th[3]=np.loadtxt(o.prefix_out+"_cl_th.txt",unpack=True)


#Compute mean and variance over nsims simulations
cl00_all=[]
cl02_all=[]
cl22_all=[]
for i in np.arange(nsims) :
    #if i%100==0 :
    print("%d-th sim"%(i+o.isim_ini))

    if not os.path.isfile(o.prefix_out+"_cl_%04d.npz"%(o.isim_ini+i)) :
        f0,f2=get_fields()
        cl00=w00.decouple_cell(nmt.compute_coupled_cell(f0,f0))#,cl_bias=clb00)
        cl02=w02.decouple_cell(nmt.compute_coupled_cell(f0,f2))#,cl_bias=clb02)
        cl22=w22.decouple_cell(nmt.compute_coupled_cell(f2,f2))#,cl_bias=clb22)
        np.savez(o.prefix_out+"_cl_%04d"%(o.isim_ini+i),
                 l=b.get_effective_ells(),cltt=cl00[0],clte=cl02[0],cltb=cl02[1],
                 clee=cl22[0],cleb=cl22[1],clbe=cl22[2],clbb=cl22[3])
    cld=np.load(o.prefix_out+"_cl_%04d.npz"%(o.isim_ini+i))
    cl00_all.append([cld['cltt']])
    cl02_all.append([cld['clte'],cld['cltb']])
    cl22_all.append([cld['clee'],cld['cleb'],cld['clbe'],cld['clbb']])
cl00_all=np.array(cl00_all)
cl02_all=np.array(cl02_all)
cl22_all=np.array(cl22_all)

#Save output
np.savez(o.prefix_out+'_clsims_%04d-%04d'%(o.isim_ini,o.isim_end),
         l=b.get_effective_ells(),cl00=cl00_all,cl02=cl02_all,cl22=cl22_all)

if o.plot_stuff :
    plt.show()
