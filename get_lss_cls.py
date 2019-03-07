from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pyccl as ccl
import os

nell=30000
def nofz(z,z0,sz,ndens):
    return np.exp(-0.5*((z-z0)/sz)**2)*ndens/np.sqrt(2*np.pi*sz**2)
z=np.linspace(0,3,512)
pz1=nofz(z,0.955,0.13,7.55)
pz2=nofz(z,0.755,0.13,7.55)
ndens1=np.sum(pz1)*np.mean(z[1:]-z[:-1])*(180*60./np.pi)**2
ndens2=np.sum(pz2)*np.mean(z[1:]-z[:-1])*(180*60./np.pi)**2
cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=2.1e-9, n_s=0.96)

bz=0.95*ccl.growth_factor(cosmo,1.)/ccl.growth_factor(cosmo,1./(1+z))
clust1=ccl.NumberCountsTracer(cosmo,has_rsd=False,dndz=(z,pz1),bias=(z,bz))
clust2=ccl.NumberCountsTracer(cosmo,has_rsd=False,dndz=(z,pz2),bias=(z,bz))
lens1=ccl.WeakLensingTracer(cosmo,dndz=(z,pz1))
lens2=ccl.WeakLensingTracer(cosmo,dndz=(z,pz2))
ell=np.arange(nell)
cl0=np.zeros(nell)*0.

cls=np.zeros([6,6,nell])
cls[0,0,:]=ccl.angular_cl(cosmo,clust1,clust1,ell);
cls[0,1,:]=ccl.angular_cl(cosmo,clust1,lens1,ell)
cls[0,2,:]=cl0
cls[0,3,:]=ccl.angular_cl(cosmo,clust1,clust2,ell)
cls[0,4,:]=ccl.angular_cl(cosmo,clust1,lens2,ell)
cls[0,5,:]=cl0
cls[1,0,:]=cls[0,1,:]
cls[1,1,:]=ccl.angular_cl(cosmo,lens1,lens1,ell)
cls[1,2,:]=cl0
cls[1,3,:]=ccl.angular_cl(cosmo,lens1,clust2,ell)
cls[1,4,:]=ccl.angular_cl(cosmo,lens1,lens2,ell)
cls[1,5,:]=cl0
cls[2,0,:]=cls[0,2,:]
cls[2,1,:]=cls[1,2,:]
cls[2,2,:]=cl0
cls[2,3,:]=cl0
cls[2,4,:]=cl0
cls[2,5,:]=cl0
cls[3,0,:]=cls[0,3,:]
cls[3,1,:]=cls[1,3,:]
cls[3,2,:]=cls[2,3,:]
cls[3,3,:]=ccl.angular_cl(cosmo,clust2,clust2,ell)
cls[3,4,:]=ccl.angular_cl(cosmo,clust2,lens2,ell)
cls[3,5,:]=cl0
cls[4,0,:]=cls[0,4,:]
cls[4,1,:]=cls[1,4,:]
cls[4,2,:]=cls[2,4,:]
cls[4,3,:]=cls[3,4,:]
cls[4,4,:]=ccl.angular_cl(cosmo,lens2,lens2,ell)
cls[4,5,:]=cl0
cls[5,0,:]=cls[0,5,:]
cls[5,1,:]=cls[1,5,:]
cls[5,2,:]=cls[2,5,:]
cls[5,3,:]=cls[3,5,:]
cls[5,4,:]=cls[4,5,:]
cls[5,5,:]=cl0

nls=np.zeros([6,6,nell])
nls[0,0,:]=1./ndens1
nls[1,1,:]=0.28**2/ndens1
nls[2,2,:]=0.28**2/ndens1
nls[3,3,:]=1/ndens2
nls[4,4,:]=0.28**2/ndens2
nls[5,5,:]=0.28**2/ndens2

np.savez("cls_lss_2bins",ls=ell,cls=cls,nls=nls)

plt.figure()
plt.plot(ell,cls[0,0],'-',label='$\\delta_1-\\delta_1$')
plt.plot(ell,cls[0,1],'-',label='$\\delta_1-\\gamma_1$')
plt.plot(ell,cls[0,3],'-',label='$\\delta_1-\\delta_2$')
plt.plot(ell,cls[0,4],'-',label='$\\delta_1-\\gamma_2$')
plt.plot(ell,cls[1,1],'-',label='$\\gamma_1-\\gamma_2$')
plt.plot(ell,cls[1,3],'-',label='$\\gamma_1-\\delta_2$')
plt.plot(ell,cls[1,4],'-',label='$\\gamma_1-\\gamma_2$')
plt.plot(ell,cls[3,3],'-',label='$\\delta_2-\\delta_2$')
plt.plot(ell,cls[3,4],'-',label='$\\delta_2-\\gamma_2$')
plt.plot(ell,cls[4,4],'-',label='$\\gamma_2-\\gamma_2$')
plt.plot(ell,nls[0,0],'--',label='$N_{d1}$')
plt.plot(ell,nls[1,1],'--',label='$N_{g1}$')
plt.plot(ell,nls[3,3],'--',label='$N_{d2}$')
plt.plot(ell,nls[4,4],'--',label='$N_{g2}$')
plt.loglog()
plt.xlabel('$\\ell$',fontsize=16)
plt.ylabel('$C_\ell$',fontsize=16)
plt.legend(loc='lower left',frameon=False,fontsize=16,labelspacing=0.1,ncol=2)
plt.show()
