import bilby
import numpy as np
import pickle
from pandas.core.frame import DataFrame
import h5py
from astropy.cosmology import Planck15
from bilby.hyper.likelihood import HyperparameterLikelihood
from bilby.core.prior import Interped, Uniform, LogUniform
from bilby.core.prior import TruncatedGaussian as TG
from bilby.core.prior import Beta as Bt
from bilby.core.prior import PowerLaw 
from scipy.interpolate import RegularGridInterpolator, interp1d
import astropy.units as u
import sys
from scipy.integrate import quad,cumtrapz
from scipy.special._ufuncs import xlogy, erf
import json
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.stats import gaussian_kde
import matplotlib.gridspec as gridspec
from tqdm import tqdm

####
#ploting
from model_libs import PS_mass

def cal_quantiles(post,values,Nsample=None):
    quants=[]
    keys=['alpha1','mmin1','mmax1','delta_low']
    for i in np.arange(15):
        keys.append('n'+str(i+1))
    ms=np.linspace(2,100,10000)
    if Nsample==None:
        Nsample=len(post['mmin1'])
    for i in tqdm(np.arange(Nsample)):
        para=[post[key][i] for key in keys]
        pdf=PS_mass(ms,*para)
        cdf=np.cumsum(pdf)/np.sum(pdf)
        f=interp1d(cdf,ms)
        quants.append(f(values))
    quants=np.array(quants).T
    return quants

####################################################################
#plot corners
####################################################################
import corner
from matplotlib import lines as mpllines
def plot_corner(post,params,show_keys,color,filename,smooth=1.):
    print('ploting')
    data2=np.array([np.array(post[key]) for key in params])
    levels = (1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.))
    c1=color
    ranges=[[min(data2[i]),max(data2[i])] for i in np.arange(len(data2))]
    percentiles=[[np.percentile(data2[i],5),np.percentile(data2[i],50),np.percentile(data2[i],95)] for i in np.arange(len(data2))]
    titles=[r"${:.2f}_{{-{:.2f}}}^{{+{:.2f}}}$".format(percentiles[i][1],percentiles[i][1]-percentiles[i][0], percentiles[i][2]-percentiles[i][1]) for i in np.arange(len(data2)) ]
    kwargs = dict(title_kwargs=dict(fontsize=10), labels=show_keys, smooth=smooth, bins=25,  quantiles=[0.05,0.5,0.95], range=ranges,\
    levels=levels, show_titles=True, titles=None, plot_density=False, plot_datapoints=True, fill_contours=True, title_qs=[0.05,0.95],\
    label_kwargs=dict(fontsize=15), max_n_ticks=1, alpha=0.5, hist_kwargs=dict(color=c1))
    groupdata=[data2]
    plt.cla()
    fig = corner.corner(groupdata[0].T, color=c1, **kwargs)
    lines = [mpllines.Line2D([0], [0], color=c1)]
    axes = fig.get_axes()
    ndim = int(np.sqrt(len(axes)))
    #axes[ndim - 1].legend(lines, labels, fontsize=14)
    plt.savefig(filename)

        

####
import corner
import numpy as np
from matplotlib import lines as mpllines
def plot_corner_2(post,params,show_keys,color,filename,smooth=1.5):
    print('ploting')
    data2=np.array([np.array(post[key]) for key in params])
    levels = (1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.))
    c1=color
    ranges=[[min(data2[i]),max(data2[i])] for i in np.arange(len(data2))]
    percentiles=[[np.percentile(data2[i],16),np.percentile(data2[i],50),np.percentile(data2[i],84)] for i in np.arange(len(data2))]
    titles=[r"${:.2f}_{{-{:.2f}}}^{{+{:.2f}}}$".format(percentiles[i][1],percentiles[i][1]-percentiles[i][0], percentiles[i][2]-percentiles[i][1]) for i in np.arange(len(data2)) ]
    kwargs = dict(title_kwargs=dict(fontsize=15), labels=show_keys, smooth=smooth, bins=25,  quantiles=[0.16,0.5,0.84], range=ranges,\
    levels=levels, show_titles=True, titles=None, plot_density=False, plot_datapoints=True, fill_contours=True, title_qs=[0.16,0.84],\
    label_kwargs=dict(fontsize=15), max_n_ticks=1, alpha=0.5, hist_kwargs=dict(color=c1))
    groupdata=[data2]
    plt.cla()
    fig = corner.corner(groupdata[0].T, color=c1, **kwargs)
    lines = [mpllines.Line2D([0], [0], color=c1)]
    axes = fig.get_axes()
    ndim = int(np.sqrt(len(axes)))
    #axes[ndim - 1].legend(lines, labels, fontsize=14)
    plt.savefig(filename)


#####
from mass_cosmo_spin_models import pls_twospin_model_prior
def plot_ma_dist(post,colors,size,filename):

    fig=plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot()
    ax3 = fig.add_axes([0.55,0.55,0.35,0.35])

    parameters={}
    indx=np.random.choice(np.arange(len(post['log_likelihood'])),size=size)
    keys=post.keys()

    for key in keys:
        parameters.update({key:np.array(post[key])[indx]})

    a_1G=[]
    a_2G=[]
    a_sam=np.linspace(0,1,500)
    m_1G=[]
    m_2G=[]
    x=np.linspace(2,100,5000)
    m1_sam=x
    for i in tqdm(np.arange(size)):
        para={key:parameters[key][i] for key in parameters.keys()}
        ms_model=pls_twospin_model_prior(name=None,hyper_params_dict=para)
        m_1G.append(ms_model.dist['mpl1'].prob(m1_sam)*(1-ms_model.r2))
        m_2G.append(ms_model.dist['mpl2'].prob(m1_sam)*(ms_model.r2))
        a_1G.append(ms_model.dist['ap1'].prob(a_sam))
        a_2G.append(ms_model.dist['ap2'].prob(a_sam))
    m_1G=np.array(m_1G)
    m_1G_pup=np.percentile(m_1G,95,axis=0)
    m_1G_plow=np.percentile(m_1G,5,axis=0)
    m_1G_pmid=np.percentile(m_1G,50,axis=0)
    m_1G_pmean=np.mean(m_1G,axis=0)
    m_2G=np.array(m_2G)
    m_2G_pup=np.percentile(m_2G,95,axis=0)
    m_2G_plow=np.percentile(m_2G,5,axis=0)
    m_2G_pmid=np.percentile(m_2G,50,axis=0)
    m_2G_pmean=np.mean(m_2G,axis=0)

    ax1.xaxis.grid(True,which='major',ls=':',color='grey')
    ax1.yaxis.grid(True,which='major',ls=':',color='grey')
    #plt.fill_between(m1_sam,plow,pup,color=colors[0],alpha=0.4,label='total')
    #plt.plot(m1_sam,pmean,color=colors[0],alpha=0.9)
    ax1.fill_between(m1_sam,m_1G_plow,m_1G_pup,color=colors[0],alpha=0.2,label='low spin')
    ax1.plot(m1_sam,m_1G_plow,color=colors[0],alpha=0.8,lw=0.5)
    ax1.plot(m1_sam,m_1G_pup,color=colors[0],alpha=0.8,lw=0.5)
    #ax1.plot(m1_sam,m_1G_pmean,color=colors[0],alpha=0.9,ls=':')
    ax1.plot(m1_sam,m_1G_pmid,color=colors[0],alpha=0.9)
    ax1.fill_between(m1_sam,m_2G_plow,m_2G_pup,color=colors[1],alpha=0.2,label='high spin')
    ax1.plot(m1_sam,m_2G_plow,color=colors[1],alpha=0.8,lw=0.5)
    ax1.plot(m1_sam,m_2G_pup,color=colors[1],alpha=0.8,lw=0.5)
    #ax1.plot(m1_sam,m_2G_pmean,color=colors[1],alpha=0.9,ls=':')
    ax1.plot(m1_sam,m_2G_pmid,color=colors[1],alpha=0.9)

    ax1.set_yscale('log')
    ax1.set_ylim(1e-4,1)
    ax1.set_xlim(0,100)
    ax1.set_xlabel(r'$m/M_{\odot}$')
    ax1.set_ylabel(r'$p(m)$')
    ax1.legend(loc=2)


    a_1G=np.array(a_1G)
    a_1G_pup=np.percentile(a_1G,95,axis=0)
    a_1G_plow=np.percentile(a_1G,5,axis=0)
    a_1G_pmid=np.percentile(a_1G,50,axis=0)
    a_1G_pmean=np.mean(a_1G,axis=0)
    a_2G=np.array(a_2G)
    a_2G_pup=np.percentile(a_2G,95,axis=0)
    a_2G_plow=np.percentile(a_2G,5,axis=0)
    a_2G_pmid=np.percentile(a_2G,50,axis=0)
    a_2G_pmean=np.mean(a_2G,axis=0)

    ax3.xaxis.grid(True,which='major',ls=':',color='grey')
    ax3.yaxis.grid(True,which='major',ls=':',color='grey')

    ax3.fill_between(a_sam,a_1G_plow,a_1G_pup,color=colors[0],alpha=0.2,label='low spin')
    #ax3.plot(a_sam,a_1G_pmean,color=colors[0],alpha=0.8)
    ax3.plot(a_sam,a_1G_pmid,color=colors[0],alpha=0.8)
    ax3.fill_between(a_sam,a_2G_plow,a_2G_pup,color=colors[1],alpha=0.2,label='high spin')
    #ax3.plot(a_sam,a_2G_pmean,color=colors[1],alpha=0.8)
    ax3.plot(a_sam,a_2G_pmid,color=colors[1],alpha=0.8)
    
    ax3.set_xlabel(r'$a$')
    ax3.set_ylabel(r'$p(a)$')
    ax3.set_xlim(0,1)
    ax3.set_ylim(0,6)
    plt.tight_layout()
    plt.savefig(filename)


def cal_BBH_quantiles(post,values,Nsample=None):
    quants=[]
    
    m1_sam=np.linspace(2,100,1000)
    m2_sam=np.linspace(2,100,499)
    x,y = np.meshgrid(m1_sam,m2_sam)
    dx = m1_sam[1]-m1_sam[0]
    dy = m2_sam[1]-m2_sam[0]
    if Nsample is None:
        Nsample=len(post['mmin1'])
    for i in tqdm(np.arange(Nsample)):
        para={key:post[key][i] for key in post.keys()}
        ms_model=pls_twospin_model_prior(name=None,hyper_params_dict=para)
        pdf=np.sum(ms_model.dist['mpl1'].prob(x)*ms_model.dist['mpl1'].prob(y)*ms_model.dist['pair'].prob(y/x),axis=0)
        cdf=np.cumsum(pdf)/np.sum(pdf)
        f=interp1d(cdf,m1_sam)
        quants.append(f(values))
    quants=np.array(quants).T
    return quants
    

