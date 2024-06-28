import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import lines as mpllines
from bilby.core.prior import Interped
from tqdm import tqdm
import h5py as h5
import seaborn as sns
import pickle
sns.set_context('talk')
sns.set_style('ticks')
sns.set_palette('colorblind')
c=sns.color_palette('colorblind')
from plot_utils import cal_quantiles, plot_corner, plot_corner_2, plot_ma_dist, cal_BBH_quantiles
colors=['lightcoral','sandybrown','darkorange','goldenrod','olive','palegreen','lightseagreen','darkcyan','skyblue','navy','indigo','crimson']

outdir='results'
fig_dir='figures'

###########################
#read data
###########################

#Twospin
with open('./{}/pls_twospin.pickle'.format(outdir), 'rb') as fp:
    twospin_post = pickle.load(fp)

#PP
with open('./{}/pp.pickle'.format(outdir), 'rb') as fp:
    pp_post = pickle.load(fp)
    
#PP
with open('./{}/pls.pickle'.format(outdir), 'rb') as fp:
    pls_post = pickle.load(fp)

#GW170817
H0_GW170817=np.loadtxt(open('./data/ExtendedDataFigure2.csv'),skiprows=2,delimiter=',',usecols=[0,1,2]).T[0]
# Other H0 measurements
planck_h = 0.6766*100
sigma_planck_h = 0.0042*100
riess_h = 0.7330*100
sigma_riess_h = 0.0104*100


#####################
#plot mass spin
#####################

size=2000
filename='./{}/mass_spin.pdf'.format(fig_dir)
#plot_ma_dist(twospin_post,['navy','darkorange','crimson'],size,filename)

#########################
#plot corner
############################

params=['Om0','H0','gamma','kappa','zp','alpha1','mmin1','mmax1','alpha2','mmin2','mmax2','r2','beta']
show_keys=[r'$\Omega_{\rm m}$',r'$H_0~[{\rm km}~{\rm s}^{-1}~{\rm Mpc}^{-1}]$',r'$\gamma$',r'$\kappa$',r'$z_{\rm p}$',r'$\alpha_1$',r'$m_{\rm min,1}~[M_{\odot}]$',r'$m_{\rm max,1}~[M_{\odot}]$',r'$\alpha_2$',r'$m_{\rm min,2}~[M_{\odot}]$',r'$m_{\rm max,2}~[M_{\odot}]$',r'$r_2$',r'$\beta$']
filename='./{}/H0_snr_11_corner.pdf'.format(fig_dir)
color='skyblue'
plot_corner_2(twospin_post,params,show_keys,color,filename,smooth=0.5)


###########################
####compare H0
###########################

H0_pls_twospin=twospin_post['H0']

H0_PP=pp_post['H0']

H0_PLS=pls_post['H0']
#prior
hpr=np.linspace(10,200,100)
prior_uniform = np.ones(len(hpr))/(200.-10.)

plt.figure(figsize=(10, 7.5), dpi=80)
plt.xlim(10,200)
plt.ylim(0, 0.028)
ymin,ymax=0, 0.028
plt.plot(hpr, prior_uniform,ls=':', linewidth = 3.0, c='black', alpha=0.6,label='Prior')

plt.axvline(planck_h, label='Planck', color=c[4], alpha=0.7)
plt.fill_betweenx([ymin,ymax], planck_h-2*sigma_planck_h, planck_h+2*sigma_planck_h, color=c[4], alpha=0.2)
plt.axvline(riess_h, label='SH0ES', color=c[2], alpha=0.7)
plt.fill_betweenx([ymin,ymax], riess_h-2*sigma_riess_h, riess_h+2*sigma_riess_h, color=c[2], alpha=0.2)

#plt.hist(H0_GW170817,bins='auto',density=True,histtype='step',linewidth = 2.0, label='GW170817 Counterpart', color=c[1])

plt.hist(H0_PP,bins='auto',density=True,histtype='step',linewidth = 3.0, label='PP', color=c[3])
plt.hist(H0_PLS,bins='auto',density=True,histtype='step',linewidth = 3.0, label='PS', color=c[5])

#plt.hist(H0_pl_twospin,bins='auto',density=True,histtype='step',linewidth = 3.0, label='Two Spin PL', color=c[1])
plt.hist(H0_pls_twospin,bins='auto',density=True,histtype='step',linewidth = 3.0, label='Two Spin', color=c[0])

plt.xlabel(r'$H_0[{\rm km\, s^{-1} \, Mpc^{-1} }]$',fontsize=18)
plt.ylabel(r'$p(H_0|x)[{\rm km^{-1}\,  s \, Mpc }]$',fontsize=18)
plt.legend(prop={'size': 14})
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.tight_layout()

plt.savefig('./{}/H0_compare.pdf'.format(fig_dir))

####################################
#Hz
####################################

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

H0_pls_twospin=twospin_post['H0']
Om0_pls_twospin=twospin_post['Om0']

H0_PP=pp_post['H0']
Om0_PP=pp_post['Om0']

xx=np.linspace(0,2,100)
Hzs_pls_twospin=[]
Nsample_pls_twospin=len(H0_pls_twospin)
for i in tqdm(np.arange(Nsample_pls_twospin)):
    cosmo=FlatLambdaCDM(Om0=Om0_pls_twospin[i],H0=H0_pls_twospin[i])
    Hzs_pls_twospin.append(cosmo.efunc(xx)*H0_pls_twospin[i])
Hzs_pls_twospin=np.array(Hzs_pls_twospin)
Hz_pls_twospin_up=np.percentile(Hzs_pls_twospin,95,axis=0)
Hz_pls_twospin_low=np.percentile(Hzs_pls_twospin,5,axis=0)
Hz_pls_twospin_mid=np.percentile(Hzs_pls_twospin,50,axis=0)

Hzs_PP=[]
Nsample_PP=len(H0_PP)
for i in tqdm(np.arange(Nsample_PP)):
    cosmo=FlatLambdaCDM(Om0=Om0_PP[i],H0=H0_PP[i])
    Hzs_PP.append(cosmo.efunc(xx)*H0_PP[i])
Hzs_PP=np.array(Hzs_PP)
Hz_up_PP=np.percentile(Hzs_PP,95,axis=0)
Hz_low_PP=np.percentile(Hzs_PP,5,axis=0)
Hz_mid_PP=np.percentile(Hzs_PP,50,axis=0)


plt.figure(figsize=(8, 6), dpi=80)
plt.xlim(0,1)
plt.ylim(0,300)

plt.fill_between(xx,Hz_pls_twospin_low,Hz_pls_twospin_up,color=c[0],alpha=0.3,label='TwoSpin')
plt.plot(xx,Hz_pls_twospin_mid,color=c[0],alpha=0.8)

plt.fill_between(xx,Hz_low_PP,Hz_up_PP,color=c[3],alpha=0.3,label='PP')
plt.plot(xx,Hz_mid_PP,color=c[3],alpha=0.8)

plt.ylabel(r'$H(z)[{\rm km\, s^{-1} \, Mpc^{-1} }]$',fontsize=18)
plt.xlabel(r'$z$',fontsize=18)
plt.legend(prop={'size': 14})
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.tight_layout()

plt.savefig('./{}/Hz.pdf'.format(fig_dir))



########################
####With GW170817
########################
from scipy.stats import gaussian_kde
xmin,xmax=10,200
xx=np.linspace(xmin,xmax,1000)
Dx=xmax-xmin
from bilby.core.prior import LogUniform
########################
#GW170817
########################
#data from LIGO Nature
H0_GW170817=np.loadtxt(open('./data/ExtendedDataFigure2.csv'),skiprows=2,delimiter=',',usecols=[0,1,2]).T[1]
#prior_weight=LogUniform(xmin,xmax).prob(H0_GW170817)
prior_weight=np.ones(len(H0_GW170817))
kdeBNS=gaussian_kde(H0_GW170817,weights=prior_weight)
yyBNS=kdeBNS(xx)

pdfs=yyBNS
midindx=np.argmax(yyBNS)
m=xx[midindx]
cdf=0
i=1
j=1
while cdf<0.683:
    if pdfs[midindx+i]>pdfs[midindx-j]:
        cdf+=pdfs[midindx+i]*Dx/1000.
        i+=1
    else:
        cdf+=pdfs[midindx-j]*Dx/1000.
        j+=1
low68=xx[midindx-j]
high68=xx[midindx+i]
while cdf<0.954:
    if pdfs[midindx+i]>pdfs[midindx-j]:
        cdf+=pdfs[midindx+i]*Dx/1000.
        i+=1
    else:
        cdf+=pdfs[midindx-j]*Dx/1000.
        j+=1
low95=xx[midindx-j]
high95=xx[midindx+i]

print('GW170817 HMI:',m,'+',high68-m,'-',m-low68)


#Twospin

H0_pls_twospin=twospin_post['H0']
BNSweights=kdeBNS(H0_pls_twospin)
kdetwospin=gaussian_kde(dataset=H0_pls_twospin,weights=list(BNSweights))
yytwospin=kdetwospin(xx)

pdfs=yytwospin
midindx=np.argmax(yytwospin)
m=xx[midindx]
cdf=0
i=1
j=1
while cdf<0.683:
    if pdfs[midindx+i]>pdfs[midindx-j]:
        cdf+=pdfs[midindx+i]*Dx/1000.
        i+=1
    else:
        cdf+=pdfs[midindx-j]*Dx/1000.
        j+=1
low68=xx[midindx-j]
high68=xx[midindx+i]
while cdf<0.954:
    if pdfs[midindx+i]>pdfs[midindx-j]:
        cdf+=pdfs[midindx+i]*Dx/1000.
        i+=1
    else:
        cdf+=pdfs[midindx-j]*Dx/1000.
        j+=1
low95=xx[midindx-j]
high95=xx[midindx+i]

print('twospin HMI:',m,'+',high68-m,'-',m-low68)


#PP
H0_PP=pp_post['H0']
BNSweights=kdeBNS(H0_PP)
kdePP=gaussian_kde(dataset=H0_PP,weights=list(BNSweights))
yyPP=kdePP(xx)

pdfs=yyPP
midindx=np.argmax(yyPP)
m=xx[midindx]
cdf=0
i=1
j=1
while cdf<0.683:
    if pdfs[midindx+i]>pdfs[midindx-j]:
        cdf+=pdfs[midindx+i]*Dx/1000.
        i+=1
    else:
        cdf+=pdfs[midindx-j]*Dx/1000.
        j+=1
low68=xx[midindx-j]
high68=xx[midindx+i]
while cdf<0.954:
    if pdfs[midindx+i]>pdfs[midindx-j]:
        cdf+=pdfs[midindx+i]*Dx/1000.
        i+=1
    else:
        cdf+=pdfs[midindx-j]*Dx/1000.
        j+=1
low95=xx[midindx-j]
high95=xx[midindx+i]

print('PP HMI:',m,'+',high68-m,'-',m-low68)

# Other H0 measurements
planck_h = 0.6766*100
sigma_planck_h = 0.0042*100
riess_h = 0.7330*100
sigma_riess_h = 0.0104*100
##
plt.figure(figsize=(8, 6), dpi=80)
plt.xlim(10,200)
plt.ylim(0, 0.05)
ymin,ymax=0, 0.05

plt.plot(xx,yyBNS, linewidth = 3.0, c='black', alpha=0.6,label='GW170817')
plt.axvline(planck_h, label='Planck', color=c[4], alpha=0.7)
plt.fill_betweenx([ymin,ymax], planck_h-2*sigma_planck_h, planck_h+2*sigma_planck_h, color=c[4], alpha=0.2)
plt.axvline(riess_h, label='SH0ES', color=c[2], alpha=0.7)
plt.fill_betweenx([ymin,ymax], riess_h-2*sigma_riess_h, riess_h+2*sigma_riess_h, color=c[2], alpha=0.2)
plt.plot(xx,yytwospin, linewidth = 3.0, c=c[0], alpha=0.6,label='TwoSpin\&GW170817')
plt.plot(xx,yyPP, linewidth = 3.0, c=c[1], alpha=0.6,label='PP\&GW170817')
plt.xlabel(r'$H_0[{\rm km\, s^{-1} \, Mpc^{-1} }]$',fontsize=18)
plt.ylabel(r'$p(H_0)[{\rm km^{-1}\,  s \, Mpc }]$',fontsize=18)
plt.legend(prop={'size': 14})
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.tight_layout()

plt.savefig('./{}/H0_with_GW170817.pdf'.format(fig_dir))

