import icarogw
import os
import pickle
import bilby
from bilby.core.prior import Uniform
from scipy.interpolate import CubicSpline

import h5py
import numpy as np
import sys

from scipy.interpolate import interp1d, splev, splrep
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from bilby.core.prior import TruncatedGaussian as TG
from bilby.core.prior import Beta as Bt

######################################################
#spin mass
######################################################
#tilt angle
def default_ct(ct1,ct2,sigmat,zeta):
    aligned=bilby.core.prior.TruncatedGaussian(1,sigmat,-1,1)
    iso=bilby.core.prior.Uniform(-1,1)
    return aligned.prob(ct1)*aligned.prob(ct2)*zeta+(1-zeta)*iso.prob(ct1)*iso.prob(ct2)+1e-100

#conversion
def E_of_z(z,H0,Om0):
    return np.sqrt(Om0*(1+z)**3+(1-Om0))

from astropy import constants
class cosmo_conversion(object):
   
    def __init__(self, Om0=0.308,H0=67,zmax=10):
        """ Initialize the cosmology

        Parameters
        ----------
        Omega_m : float
            Fraction of matter energy
        H0 : float
            Hubble constant today in km/Mpc/s
        zmax : float
            Maximum redshift used for the cosmology
        """

        self.astropy_cosmology = FlatLambdaCDM(Om0=Om0,H0=H0)
        self.zmax=zmax
        self.H0=H0
        self.Om0=Om0

        z_array = np.logspace(-4,np.log10(self.zmax),  2500)
        log_dl_trials = np.log10(self.astropy_cosmology.luminosity_distance(z_array).value)
        log_dvc_dz_trials =  np.log10(4*np.pi*self.astropy_cosmology.differential_comoving_volume(z_array).to(u.Gpc**3/u.sr).value)

        log_z_array = np.log10(z_array)

        # Interpolate the lookup tables
        self.interp_dvc_dz = splrep(log_z_array,log_dvc_dz_trials,s=0)
        self.interp_dl_to_z = splrep(log_dl_trials,log_z_array,s=0)
        self.interp_z_to_dl = splrep(log_z_array,log_dl_trials,s=0)
        
    def dl_at_z(self, z):

        return np.nan_to_num(10.**splev(np.log10(z),self.interp_z_to_dl,ext=0))

    def z_at_dl(self,dl):
        
        return np.nan_to_num(10.**splev(np.log10(dl),self.interp_dl_to_z,ext=0))

    def dVc_by_dz(self,z):
        
        return np.nan_to_num(10.**splev(np.log10(z),self.interp_dvc_dz,ext=0))

    def dL_by_dz(self, z):
        """
        Calculates the d_dL/dz for this cosmology
        """
        speed_of_light = constants.c.to('km/s').value
        # Calculate the Jacobian of the luminosity distance w.r.t redshift
        return self.dl_at_z(z)/(1+z) + speed_of_light*(1+z)/(self.H0*E_of_z(z,self.H0,self.Om0))
        
    def source_to_detector_jacobian(self,dl):
   
        z_samples = self.z_at_dl(dl)
        jacobian = self.detector_to_source_jacobian(z_samples)
        return 1./jacobian

    def detector_to_source_jacobian(self,z):
 
        jacobian = (1+z)**2*self.dL_by_dz(z)
        return jacobian

    def source_frame_to_detector_frame(self,ms_1,ms_2,redshift_samples):

        distance_samples = self.dl_at_z(redshift_samples)
        md1 = ms_1*(1+redshift_samples)
        md2 = ms_2*(1+redshift_samples)

        return md1, md2, distance_samples

    def detector_frame_to_source_frame(self,md_1,md_2,distance_samples):

        z_samples = self.z_at_dl(distance_samples)
        m1s = md_1/(1+z_samples)
        m2s = md_2/(1+z_samples)

        return m1s, m2s, z_samples


import copy
from icarogw.priors import custom_math_priors as cmp
from icarogw.priors.custom_math_priors import _S_factor as low_S_factor

def high_S_factor(mass,mmax,delta_high):
    '''
    This function return the value of the window function defined as Eqs B6 and B7 of https://arxiv.org/pdf/2010.14533.pdf

    Parameters
    ----------
    mass: np.array or float
        array of x or masses values
        
    mmax: float or np.array (in this case len(mmin) == len(mass))
        minimum value of window function
    delta_high: float or np.array (in this case len(delta_m) == len(mass))
        width of the window function

    Returns
    -------
    Values of the window function
    '''

    if not isinstance(mass,np.ndarray):
        mass = np.array([mass])

    to_ret = np.ones_like(mass)
    if delta_high == 0:
        return to_ret

    mprime = mmax-mass

    # Defines the different regions of thw window function ad in Eq. B6 of  https://arxiv.org/pdf/2010.14533.pdf
    select_window = (mass<mmax) & (mass>(mmax-delta_high))
    select_one = mass<=(mmax-delta_high)
    select_zero = mass>=mmax

    effe_prime = np.ones_like(mass)

    # Definethe f function as in Eq. B7 of https://arxiv.org/pdf/2010.14533.pdf
    effe_prime[select_window] = np.exp(np.nan_to_num((delta_high/mprime[select_window])+(delta_high/(mprime[select_window]-delta_high))))
    to_ret = 1./(effe_prime+1)
    to_ret[select_zero]=0.
    to_ret[select_one]=1.
    return to_ret
    
class SmoothedProb(object):
    '''
    Class for smoothing the low part of a PDF. The smoothing follows Eq. B7 of
    2010.14533.

    Parameters
    ----------
    origin_prob: class
        Original prior class to smooth from this module
    bottom: float
        minimum cut-off. Below this, the window is 0.
    bottom_smooth: float
        smooth factor. The smoothing acts between bottom and bottom+bottom_smooth
    '''

    def __init__(self,origin_prob,bottom,bottom_smooth,top,top_smooth):

        self.origin_prob = copy.deepcopy(origin_prob)
        self.bottom_smooth = bottom_smooth
        self.bottom = bottom
        self.top_smooth = top_smooth
        self.top = top
        self.maximum=self.origin_prob.maximum
        self.minimum=self.origin_prob.minimum

        # Find the values of the integrals in the region of the window function before and after the smoothing
        low_int_array = np.linspace(self.origin_prob.minimum,bottom+bottom_smooth,1000)
        low_integral_before = np.trapz(self.origin_prob.prob(low_int_array),low_int_array)
        low_integral_now = np.trapz(self.prob(low_int_array),low_int_array)
        
        high_int_array = np.linspace(self.origin_prob.maximum-top_smooth,top,1000)
        high_integral_before = np.trapz(self.origin_prob.prob(high_int_array),high_int_array)
        high_integral_now = np.trapz(self.prob(high_int_array),high_int_array)

        self.integral_before = low_integral_before+high_integral_before
        self.integral_now = low_integral_now+high_integral_now
        # Renormalize the the smoother function.
        self.norm = 1 - self.integral_before + self.integral_now

    def prob(self,x):
        """
        Returns the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        return np.exp(self.log_prob(x))

    def log_prob(self,x):
        """
        Returns the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        # Return the window function
        window_low = low_S_factor(x, self.bottom,self.bottom_smooth)
        window_high = high_S_factor(x, self.top,self.top_smooth)

        if hasattr(self,'norm'):
            prob_ret =self.origin_prob.log_prob(x)+np.log(window_low)+np.log(window_high)-np.log(self.norm)
        else:
            prob_ret =self.origin_prob.log_prob(x)+np.log(window_low)+np.log(window_high)

        return prob_ret

#########

class PowerLawSpline_math(object):
    """
    Class for a powerlaw probability :math:`p(x) \\propto x^{\\alpha}` defined in
    [a,b]

    Parameters
    ----------
    alpha: float
        Powerlaw slope
    min_pl: float
        lower cutoff
    max_pl: float
        upper cutoff
    n_i: float
        interped value at i-th knot
    n_low: float
        location of the first knot
    n_high: float
        location of the last knot
    """

    def __init__(self,alpha,min_pl,max_pl,delta_low,delta_high,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,n_low=5,n_high=100):

        self.minimum = min_pl
        self.maximum = max_pl
        self.min_pl = min_pl
        self.max_pl = max_pl
        self.alpha = alpha
        self.n_low=n_low
        self.n_high=n_high
        
        self.n1,self.n2,self.n3,self.n4,self.n5,self.n6,self.n7,self.n8,self.n9,self.n10,self.n11,self.n12,self.n13,self.n14,self.n15 = \
        n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15

        self.pl= SmoothedProb(cmp.PowerLaw_math(-alpha,min_pl,max_pl),min_pl,delta_low,max_pl,delta_high)
        xi=np.exp(np.linspace(np.log(n_low),np.log(n_high),15))
        yi=np.array([n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15])
        self.cs = CubicSpline(xi,yi,bc_type='natural')
        xx=np.linspace(2,100,1000)
        yy=np.exp(self.cs(xx)*(xx>n_low)*(xx<n_high))*self.pl.prob(xx)
        self.norm=np.sum(yy)*98./1000.

    def prob(self,x):
        """
        Returns the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """
        
        return np.exp(self.log_prob(x))

    def log_prob(self,x):
        """
        Returns the logarithm of the probability density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        """

        to_ret = self.cs(x)*(x>self.n_low)*(x<self.n_high)+self.pl.log_prob(x)-np.log(self.norm)
        to_ret[(x<self.min_pl) | (x>self.max_pl)] = -np.inf

        return to_ret

    def log_conditioned_prob(self,x,a,b):
        """
        Returns the conditional probability between two new boundaries [a,b]

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        a: np.array or float
            New lower boundary
        b: np.array or float
            New upper boundary
        """

        return np.log(self.conditioned_prob(x,a,b))

    def cdf(self,x):
        """
        Returns the cumulative density function normalized

        Parameters
        ----------
        x: np.array or float
            Value at which compute the cumulative
        """

        # Define the cumulative density function, see  Eq. 24 to see the integral form

        xx=np.linspace(2,x,1000)
        yy=np.exp(self.cs(xx)*(xx>n_low)*(xx<n_high))*self.pl.prob(xx)
        to_ret=np.sum(yy)*(x-2.)/1000./self.norm

        return to_ret

    def conditioned_prob(self,x,a,b):
        """
        Returns the conditional probability between two new boundaries [a,b]

        Parameters
        ----------
        x: np.array or float
            Value at which compute the probability
        a: np.array or float
            New lower boundary
        b: np.array or float
            New upper boundary
        """
        xx=np.linspace(a,b,1000)
        yy=np.exp(self.cs(xx)*(xx>n_low)*(xx<n_high))*self.pl.prob(xx)
        norm=np.sum(yy)*(b-a)/1000.
        to_ret=np.exp(self.cs(x)*(x>n_low)*(x<n_high))*self.pl.prob(x)
        to_ret[(x<a) | (x>b)] = 1e-100

        return to_ret

##############################################################################
class pl_twospin_model_prior(object):
    """
    Wrapper for for managing the priors on source frame masses.
    The prior is factorized as :math:`p(m_1,m_2) \\propto p(m_1)p(m_2|m_1)`

    Parameters
    ----------
    name: str
        Name of the model that you want. Available 'BBH-powerlaw', 'BBH-powerlaw-gaussian'
        'BBH-broken-powerlaw', 'BBH-powerlaw-double-gaussian'.
    hyper_params_dict: dict
        Dictionary of hyperparameters for the prior model you want to use. See code lines for more details
    bilby_priors: boolean
        If you want to use bilby priors or not. It is faster to use the analytical functions.
    """

    def __init__(self,name, hyper_params_dict):

        self.hyper_params_dict=copy.deepcopy(hyper_params_dict)
        dist = {}
        
        alpha1 = hyper_params_dict['alpha1']
        beta = hyper_params_dict['beta']
        mmin1 = hyper_params_dict['mmin1']
        mmax1 = hyper_params_dict['mmax1']

        alpha_peak = hyper_params_dict['alpha_peak']

        delta_low = hyper_params_dict['delta_low']
        delta_high = hyper_params_dict['delta_high']
        
        alpha2 = hyper_params_dict['alpha2']
        mmin2 = hyper_params_dict['mmin2']
        mmax2 = hyper_params_dict['mmax2']
        
        mua1 = hyper_params_dict['mua1']
        sigmaa1 = hyper_params_dict['sigmaa1']
        mua2 = hyper_params_dict['mua2']
        sigmaa2 = hyper_params_dict['sigmaa2']

        self.dist=dict(mpl1 = SmoothedProb(cmp.PowerLaw_math(alpha=-alpha1,min_pl=mmin1,max_pl=mmax1),bottom=mmin1,bottom_smooth=delta_low,top=mmax1,top_smooth=delta_high),
                mplp = SmoothedProb(cmp.PowerLaw_math(alpha=-alpha_peak,min_pl=mmin1,max_pl=mmax1),bottom=mmin1,bottom_smooth=delta_low,top=mmax1,top_smooth=delta_high),
                mpl2 = cmp.PowerLaw_math(alpha=-alpha2,min_pl=mmin2,max_pl=mmax2),
                
                ap1 = cmp.Truncated_Gaussian_math(mua1,sigmaa1,0,1),
                ap2 = cmp.Truncated_Gaussian_math(mua2,sigmaa2,0,1),
                
                pair = cmp.PowerLaw_math(alpha=beta,min_pl=0.01,max_pl=1))

        # TODO Assume that the gaussian peak does not overlap too much with the mmin
        self.mmin = mmin1
        self.mmax = mmax2
        self.r_peak=hyper_params_dict['lambda_peak']
        self.r2=hyper_params_dict['r2']
        self.sigmat = hyper_params_dict['sigmat']
        self.zeta = hyper_params_dict['zeta']
        
        m1_sam = np.linspace(mmin1,mmax2,1000)
        m2_sam = np.linspace(mmin1,mmax2,999)
        x,y = np.meshgrid(m1_sam,m2_sam)

        pgrid1 = ((self.dist['mpl1'].prob(x)*(1-self.r_peak) + self.dist['mplp'].prob(x)*self.r_peak)*(1-self.r2)+self.r2*self.dist['mpl2'].prob(x))*\
                        ((self.dist['mpl1'].prob(y)*(1-self.r_peak) + self.dist['mplp'].prob(y)*self.r_peak)*(1-self.r2)+self.r2*self.dist['mpl2'].prob(y))*self.dist['pair'].prob(y/x)
        dx = m1_sam[1]-m1_sam[0]
        dy = m2_sam[1]-m2_sam[0]

        self.mass12_norm=np.sum(pgrid1*dx*dy)
            
    def mass_prob(self, m1s, m2s):
        """
        This method returns the joint probability :math:`p(m_1,m_2)`

        Parameters
        ----------
        m1s: np.array(matrix)
            mass one in solar masses
        m2s: dict
            mass two in solar masses
        """

        # Returns the joint probability Factorized as in Eq. 33 on the paper
        to_ret = ((self.dist['mpl1'].prob(m1s)*(1-self.r_peak) + self.dist['mplp'].prob(m1s)*self.r_peak)*(1-self.r2)+self.r2*self.dist['mpl2'].prob(m1s))*\
                        ((self.dist['mpl1'].prob(m2s)*(1-self.r_peak) + self.dist['mplp'].prob(m2s)*self.r_peak)*(1-self.r2)+self.r2*self.dist['mpl2'].prob(m2s))*\
                        self.dist['pair'].prob(m2s/m1s)/self.mass12_norm

        return to_ret+1e-100
    
    def log_mass_prob(self, m1s, m2s):
        """
        This method returns the log of the joint probability :math:`p(m_1,m_2)`

        Parameters
        ----------
        m1s: np.array(matrix)
            mass one in solar masses
        m2s: dict
            mass two in solar masses
        """

        # Returns the joint probability Factorized as in Eq. 33 on the paper
        to_ret =np.log(self.mass_prob(m1s, m2s))
        to_ret[np.isnan(to_ret)]=-np.inf

        return to_ret


    def mass_spin_prob(self, m1s, m2s, a1s, a2s, ct1s, ct2s):
        """
        This method returns the joint probability :math:`p(m_1,m_2)`

        Parameters
        ----------
        m1s: np.array(matrix)
            mass one in solar masses
        m2s: dict
            mass two in solar masses
        """

        # Returns the joint probability Factorized as in Eq. 33 on the paper
        to_ret = ((self.dist['mpl1'].prob(m1s)*(1-self.r_peak) + self.dist['mplp'].prob(m1s)*self.r_peak)*(1-self.r2)*self.dist['ap1'].prob(a1s)+\
                        self.r2*self.dist['mpl2'].prob(m1s)*self.dist['ap2'].prob(a1s))*\
                        ((self.dist['mpl1'].prob(m2s)*(1-self.r_peak) + self.dist['mplp'].prob(m2s)*self.r_peak)*(1-self.r2)*self.dist['ap1'].prob(a2s)+\
                        self.r2*self.dist['mpl2'].prob(m2s)*self.dist['ap2'].prob(a2s))*\
                        default_ct(ct1s,ct2s,self.sigmat,self.zeta)*self.dist['pair'].prob(m2s/m1s)/self.mass12_norm

        return to_ret+1e-100
    
    def log_mass_spin_prob(self, m1s, m2s, a1s, a2s, ct1s, ct2s):
        """
        This method returns the log of the joint probability :math:`p(m_1,m_2)`

        Parameters
        ----------
        m1s: np.array(matrix)
            mass one in solar masses
        m2s: dict
            mass two in solar masses
        """

        # Returns the joint probability Factorized as in Eq. 33 on the paper
        to_ret =np.log(self.mass_spin_prob(m1s, m2s, a1s, a2s, ct1s, ct2s))
        to_ret[np.isnan(to_ret)]=-np.inf

        return to_ret

###########
#twospin
class pls_twospin_model_prior(object):
    """
    Wrapper for for managing the priors on source frame masses.
    The prior is factorized as :math:`p(m_1,m_2) \\propto p(m_1)p(m_2|m_1)`

    Parameters
    ----------
    name: str
        Name of the model that you want. Available 'BBH-powerlaw', 'BBH-powerlaw-gaussian'
        'BBH-broken-powerlaw', 'BBH-powerlaw-double-gaussian'.
    hyper_params_dict: dict
        Dictionary of hyperparameters for the prior model you want to use. See code lines for more details
    bilby_priors: boolean
        If you want to use bilby priors or not. It is faster to use the analytical functions.
    """

    def __init__(self,name, hyper_params_dict):

        self.hyper_params_dict=copy.deepcopy(hyper_params_dict)
        dist = {}
        
        alpha1 = hyper_params_dict['alpha1']
        beta = hyper_params_dict['beta']
        mmin1 = hyper_params_dict['mmin1']
        mmax1 = hyper_params_dict['mmax1']

        delta_low = hyper_params_dict['delta_low']
        delta_high = hyper_params_dict['delta_high']
        
        alpha2 = hyper_params_dict['alpha2']
        mmin2 = hyper_params_dict['mmin2']
        mmax2 = hyper_params_dict['mmax2']
        
        mua1 = hyper_params_dict['mua1']
        sigmaa1 = hyper_params_dict['sigmaa1']
        mua2 = hyper_params_dict['mua2']
        sigmaa2 = hyper_params_dict['sigmaa2']
        n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15 = hyper_params_dict['n1'],hyper_params_dict['n2'],hyper_params_dict['n3'],hyper_params_dict['n4'],hyper_params_dict['n5'],\
            hyper_params_dict['n6'],hyper_params_dict['n7'],hyper_params_dict['n8'],hyper_params_dict['n9'],hyper_params_dict['n10'],\
            hyper_params_dict['n11'],hyper_params_dict['n12'],hyper_params_dict['n13'],hyper_params_dict['n14'],hyper_params_dict['n15']

        self.dist=dict(mpl1 = PowerLawSpline_math(alpha1,mmin1,mmax1,delta_low,delta_high,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15),
                mpl2 = cmp.PowerLaw_math(alpha=-alpha2,min_pl=mmin2,max_pl=mmax2),
                
                ap1 = cmp.Truncated_Gaussian_math(mua1,sigmaa1,0,1),
                ap2 = cmp.Truncated_Gaussian_math(mua2,sigmaa2,0,1),
                
                pair = cmp.PowerLaw_math(alpha=beta,min_pl=0.01,max_pl=1))

        # TODO Assume that the gaussian peak does not overlap too much with the mmin
        self.mmin = mmin1
        self.mmax = mmax2
        self.r2=hyper_params_dict['r2']
        self.sigmat = hyper_params_dict['sigmat']
        self.zeta = hyper_params_dict['zeta']
        
        m1_sam = np.linspace(mmin1,mmax2,1000)
        m2_sam = np.linspace(mmin1,mmax2,999)
        x,y = np.meshgrid(m1_sam,m2_sam)

        pgrid1 = (self.dist['mpl1'].prob(x)*(1-self.r2)+self.r2*self.dist['mpl2'].prob(x))*\
                        (self.dist['mpl1'].prob(y)*(1-self.r2)+self.r2*self.dist['mpl2'].prob(y))*self.dist['pair'].prob(y/x)
        dx = m1_sam[1]-m1_sam[0]
        dy = m2_sam[1]-m2_sam[0]

        self.mass12_norm=np.sum(pgrid1*dx*dy)
            
    def mass_prob(self, m1s, m2s):
        """
        This method returns the joint probability :math:`p(m_1,m_2)`

        Parameters
        ----------
        m1s: np.array(matrix)
            mass one in solar masses
        m2s: dict
            mass two in solar masses
        """

        # Returns the joint probability Factorized as in Eq. 33 on the paper
        to_ret = (self.dist['mpl1'].prob(m1s)*(1-self.r2)+self.r2*self.dist['mpl2'].prob(m1s))*\
                (self.dist['mpl1'].prob(m2s)*(1-self.r2)+self.r2*self.dist['mpl2'].prob(m2s))*\
                        self.dist['pair'].prob(m2s/m1s)/self.mass12_norm

        return to_ret+1e-100
    
    def log_mass_prob(self, m1s, m2s):
        """
        This method returns the log of the joint probability :math:`p(m_1,m_2)`

        Parameters
        ----------
        m1s: np.array(matrix)
            mass one in solar masses
        m2s: dict
            mass two in solar masses
        """

        # Returns the joint probability Factorized as in Eq. 33 on the paper
        to_ret =np.log(self.mass_prob(m1s, m2s))
        to_ret[np.isnan(to_ret)]=-np.inf

        return to_ret


    def mass_spin_prob(self, m1s, m2s, a1s, a2s, ct1s, ct2s):
        """
        This method returns the joint probability :math:`p(m_1,m_2)`

        Parameters
        ----------
        m1s: np.array(matrix)
            mass one in solar masses
        m2s: dict
            mass two in solar masses
        """

        # Returns the joint probability Factorized as in Eq. 33 on the paper
        to_ret = (self.dist['mpl1'].prob(m1s)*(1-self.r2)*self.dist['ap1'].prob(a1s)+\
                        self.r2*self.dist['mpl2'].prob(m1s)*self.dist['ap2'].prob(a1s))*\
                (self.dist['mpl1'].prob(m2s)*(1-self.r2)*self.dist['ap1'].prob(a2s)+\
                        self.r2*self.dist['mpl2'].prob(m2s)*self.dist['ap2'].prob(a2s))*\
                        default_ct(ct1s,ct2s,self.sigmat,self.zeta)*self.dist['pair'].prob(m2s/m1s)/self.mass12_norm

        return to_ret+1e-100
    
    def log_mass_spin_prob(self, m1s, m2s, a1s, a2s, ct1s, ct2s):
        """
        This method returns the log of the joint probability :math:`p(m_1,m_2)`

        Parameters
        ----------
        m1s: np.array(matrix)
            mass one in solar masses
        m2s: dict
            mass two in solar masses
        """

        # Returns the joint probability Factorized as in Eq. 33 on the paper
        to_ret =np.log(self.mass_spin_prob(m1s, m2s, a1s, a2s, ct1s, ct2s))
        to_ret[np.isnan(to_ret)]=-np.inf

        return to_ret

############################################
#pp_twospin
class pp_twospin_model_prior(object):
    """
    Wrapper for for managing the priors on source frame masses.
    The prior is factorized as :math:`p(m_1,m_2) \\propto p(m_1)p(m_2|m_1)`

    Parameters
    ----------
    name: str
        Name of the model that you want. Available 'BBH-powerlaw', 'BBH-powerlaw-gaussian'
        'BBH-broken-powerlaw', 'BBH-powerlaw-double-gaussian'.
    hyper_params_dict: dict
        Dictionary of hyperparameters for the prior model you want to use. See code lines for more details
    bilby_priors: boolean
        If you want to use bilby priors or not. It is faster to use the analytical functions.
    """

    def __init__(self,name, hyper_params_dict):

        self.hyper_params_dict=copy.deepcopy(hyper_params_dict)
        dist = {}
        
        alpha1 = hyper_params_dict['alpha1']
        beta = hyper_params_dict['beta']
        mmin1 = hyper_params_dict['mmin1']
        mmax1 = hyper_params_dict['mmax1']

        delta_low = hyper_params_dict['delta_low']
    
        mu_g = hyper_params_dict['mu_g']
        sigma_g = hyper_params_dict['sigma_g']
        lambda_peak = hyper_params_dict['lambda_peak']
        
        alpha2 = hyper_params_dict['alpha2']
        mmin2 = hyper_params_dict['mmin2']
        mmax2 = hyper_params_dict['mmax2']
        
        mua1 = hyper_params_dict['mua1']
        sigmaa1 = hyper_params_dict['sigmaa1']
        mua2 = hyper_params_dict['mua2']
        sigmaa2 = hyper_params_dict['sigmaa2']

        self.dist=dict(mpp1 = cmp.SmoothedProb(origin_prob=cmp.PowerLawGaussian_math(alpha=-alpha1,min_pl=mmin1,max_pl=mmax1,lambda_g=lambda_peak,mean_g=mu_g,sigma_g=sigma_g,min_g=mmin1,max_g=mmax1),bottom=mmin1,bottom_smooth=delta_low),
                mpl2 = cmp.PowerLaw_math(alpha=-alpha2,min_pl=mmin2,max_pl=mmax2),
                
                ap1 = cmp.Truncated_Gaussian_math(mua1,sigmaa1,0,1),
                ap2 = cmp.Truncated_Gaussian_math(mua2,sigmaa2,0,1),
                
                pair = cmp.PowerLaw_math(alpha=beta,min_pl=0.01,max_pl=1))

        # TODO Assume that the gaussian peak does not overlap too much with the mmin
        self.mmin = mmin1
        self.mmax = mmax2
        self.r2=hyper_params_dict['r2']
        self.sigmat = hyper_params_dict['sigmat']
        self.zeta = hyper_params_dict['zeta']
        
        m1_sam = np.linspace(mmin1,mmax2,1000)
        m2_sam = np.linspace(mmin1,mmax2,999)
        x,y = np.meshgrid(m1_sam,m2_sam)

        pgrid1 = (self.dist['mpp1'].prob(x)*(1-self.r2)+self.r2*self.dist['mpl2'].prob(x))*\
                        (self.dist['mpp1'].prob(y)*(1-self.r2)+self.r2*self.dist['mpl2'].prob(y))*self.dist['pair'].prob(y/x)
        dx = m1_sam[1]-m1_sam[0]
        dy = m2_sam[1]-m2_sam[0]

        self.mass12_norm=np.sum(pgrid1*dx*dy)
            
    def mass_prob(self, m1s, m2s):
        """
        This method returns the joint probability :math:`p(m_1,m_2)`

        Parameters
        ----------
        m1s: np.array(matrix)
            mass one in solar masses
        m2s: dict
            mass two in solar masses
        """

        # Returns the joint probability Factorized as in Eq. 33 on the paper
        to_ret = (self.dist['mpp1'].prob(m1s)*(1-self.r2)+self.r2*self.dist['mpl2'].prob(m1s))*\
                    (self.dist['mpp1'].prob(m2s)*(1-self.r2)+self.r2*self.dist['mpl2'].prob(m2s))*\
                        self.dist['pair'].prob(m2s/m1s)/self.mass12_norm

        return to_ret+1e-100
    
    def log_mass_prob(self, m1s, m2s):
        """
        This method returns the log of the joint probability :math:`p(m_1,m_2)`

        Parameters
        ----------
        m1s: np.array(matrix)
            mass one in solar masses
        m2s: dict
            mass two in solar masses
        """

        # Returns the joint probability Factorized as in Eq. 33 on the paper
        to_ret =np.log(self.mass_prob(m1s, m2s))
        to_ret[np.isnan(to_ret)]=-np.inf

        return to_ret


    def mass_spin_prob(self, m1s, m2s, a1s, a2s, ct1s, ct2s):
        """
        This method returns the joint probability :math:`p(m_1,m_2)`

        Parameters
        ----------
        m1s: np.array(matrix)
            mass one in solar masses
        m2s: dict
            mass two in solar masses
        """

        # Returns the joint probability Factorized as in Eq. 33 on the paper
        to_ret = (self.dist['mpp1'].prob(m1s)*(1-self.r2)*self.dist['ap1'].prob(a1s)+\
                        self.r2*self.dist['mpl2'].prob(m1s)*self.dist['ap2'].prob(a1s))*\
                (self.dist['mpp1'].prob(m2s)*(1-self.r2)*self.dist['ap1'].prob(a2s)+\
                        self.r2*self.dist['mpl2'].prob(m2s)*self.dist['ap2'].prob(a2s))*\
                        default_ct(ct1s,ct2s,self.sigmat,self.zeta)*self.dist['pair'].prob(m2s/m1s)/self.mass12_norm

        return to_ret+1e-100
    
    def log_mass_spin_prob(self, m1s, m2s, a1s, a2s, ct1s, ct2s):
        """
        This method returns the log of the joint probability :math:`p(m_1,m_2)`

        Parameters
        ----------
        m1s: np.array(matrix)
            mass one in solar masses
        m2s: dict
            mass two in solar masses
        """

        # Returns the joint probability Factorized as in Eq. 33 on the paper
        to_ret =np.log(self.mass_spin_prob(m1s, m2s, a1s, a2s, ct1s, ct2s))
        to_ret[np.isnan(to_ret)]=-np.inf

        return to_ret


##############
#default spin
##############

def default_spin(a1,a2,ct1,ct2,mu_a,sigma_a,sigma_t,zeta):
    t=mu_a*(1-mu_a)/sigma_a**2-1
    alpha_a=mu_a**2*(1-mu_a)/sigma_a**2-mu_a
    beta_a=t-alpha_a
    pa=(Bt(alpha_a,beta_a).prob(a1.reshape(-1))*Bt(alpha_a,beta_a).prob(a2.reshape(-1))).reshape(a1.shape)
    p1ct=TG(1,sigma_t,-1,1).prob(ct1)*TG(1,sigma_t,-1,1).prob(ct2)
    p2ct=Uniform(-1,1).prob(ct1)*Uniform(-1,1).prob(ct2)
    pct = p1ct*zeta+(1-zeta)*p2ct
    return pct*pa

#LIGO model
from icarogw.priors.mass import mass_prior
class LIGO_model_prior(object):
    """
    Wrapper for for managing the priors on source frame masses.
    The prior is factorized as :math:`p(m_1,m_2) \\propto p(m_1)p(m_2|m_1)`

    Parameters
    ----------
    name: str
        Name of the model that you want. Available 'BBH-powerlaw', 'BBH-powerlaw-gaussian'
        'BBH-broken-powerlaw', 'BBH-powerlaw-double-gaussian'.
    hyper_params_dict: dict
        Dictionary of hyperparameters for the prior model you want to use. See code lines for more details
    bilby_priors: boolean
        If you want to use bilby priors or not. It is faster to use the analytical functions.
    """

    def __init__(self,name, hyper_params_dict):

        self.mass_model=mass_prior(name,hyper_params_dict)
            
    def mass_prob(self, m1s, m2s):
        """
        This method returns the joint probability :math:`p(m_1,m_2)`

        Parameters
        ----------
        m1s: np.array(matrix)
            mass one in solar masses
        m2s: dict
            mass two in solar masses
        """

        # Returns the joint probability Factorized as in Eq. 33 on the paper

        return self.mass_model.joint_prob(m1s,m2s)+1e-100
    
    def log_mass_prob(self, m1s, m2s):
        """
        This method returns the log of the joint probability :math:`p(m_1,m_2)`

        Parameters
        ----------
        m1s: np.array(matrix)
            mass one in solar masses
        m2s: dict
            mass two in solar masses
        """

        # Returns the joint probability Factorized as in Eq. 33 on the paper

        return self.mass_model.log_joint_prob(m1s,m2s)

    def mass_spin_prob(self, m1s, m2s, a1s, a2s, ct1s, ct2s, mu_a,sigma_a,sigma_t,zeta):
        """
        This method returns the joint probability :math:`p(m_1,m_2)`

        Parameters
        ----------
        m1s: np.array(matrix)
            mass one in solar masses
        m2s: dict
            mass two in solar masses
        """

        # Returns the joint probability Factorized as in Eq. 33 on the paper
        to_ret = default_spin(a1s, a2s, ct1s, ct2s ,mu_a,sigma_a,sigma_t,zeta)*self.mass_model.joint_prob(m1s,m2s)+1e-100

        return to_ret+1e-100
    
    def log_mass_spin_prob(self, m1s, m2s, a1s, a2s, ct1s, ct2s ,mu_a,sigma_a,sigma_t,zeta):
        """
        This method returns the log of the joint probability :math:`p(m_1,m_2)`

        Parameters
        ----------
        m1s: np.array(matrix)
            mass one in solar masses
        m2s: dict
            mass two in solar masses
        """

        # Returns the joint probability Factorized as in Eq. 33 on the paper

        return np.log(self.mass_spin_prob(m1s, m2s, a1s, a2s, ct1s, ct2s ,mu_a,sigma_a,sigma_t,zeta))

################################################
#priors
################################################

                   
def default_spin_constraint(params):
    mu_a=params['mu_a']
    sigma_a=params['sigma_a']
    t=mu_a*(1-mu_a)/sigma_a**2-1
    alpha_a=mu_a**2*(1-mu_a)/sigma_a**2-mu_a
    beta_a=t-alpha_a
    params['constraint']=np.sign(alpha_a)+np.sign(beta_a)-2
    return params


def default_spin_priors():
    prior_dict=bilby.prior.PriorDict(conversion_function=default_spin_constraint)
        
    prior_dict['mu_a'] = bilby.core.prior.Uniform(0,1,name='mu_a')
    prior_dict['sigma_a'] = bilby.core.prior.Uniform(0.04,0.5,name='sigma_a')
        
    prior_dict['sigma_t'] = bilby.core.prior.Uniform(0.1,4,name='sigma_t')
    prior_dict['zeta'] = bilby.core.prior.Uniform(0.,1,name='zeta')
    prior_dict['constraint'] = bilby.prior.Constraint(minimum=-0.1, maximum=0.1)
    return prior_dict
    
    

def cosmo_prior():
    prior_dict={}
    prior_dict['Om0']= Uniform(0,1)
    #0.308
    prior_dict['H0'] = Uniform(minimum=10, maximum=200, name='H0', latex_label='H0', unit=None, boundary=None)
    prior_dict['gamma'] = bilby.core.prior.Uniform(0,12,name='gamma')
    prior_dict['kappa'] = bilby.core.prior.Uniform(0,6,name='kappa')
    prior_dict['zp'] = bilby.core.prior.Uniform(0,4,name='zp')
    
    prior_dict['R0'] = bilby.core.prior.Uniform(0,100,name='R0')
    return prior_dict

def PP_prior():
    prior_dict=cosmo_prior()
    prior_dict['alpha'] = bilby.core.prior.Uniform(1.5,12,name='alpha')
    prior_dict['beta'] = bilby.core.prior.Uniform(-4,12,name='beta')
    prior_dict['mmax'] = bilby.core.prior.Uniform(50,200,name='mmax')
    prior_dict['mmin'] = bilby.core.prior.Uniform(2,10,name='mmin')
    prior_dict['mu_g'] = bilby.core.prior.Uniform(20,50,name='mu_g')
    prior_dict['sigma_g'] = bilby.core.prior.Uniform(0.4,10,name='sigma_g')
    prior_dict['lambda_peak'] = bilby.core.prior.Uniform(0,1,name='lambda_peak')
    prior_dict['delta_m'] = bilby.core.prior.Uniform(0,10,name='delta_m')
    return prior_dict

def TPL_prior():
    prior_dict=cosmo_prior()
    prior_dict['alpha'] = bilby.core.prior.Uniform(1.5,12,name='alpha')
    prior_dict['beta'] = bilby.core.prior.Uniform(-4,12,name='beta')
    prior_dict['mmax'] = bilby.core.prior.Uniform(50,200,name='mmax')
    prior_dict['mmin'] = bilby.core.prior.Uniform(2,10,name='mmin')
    return prior_dict

def BPL_prior():
    prior_dict=cosmo_prior()

    prior_dict['alpha_1'] = bilby.core.prior.Uniform(1.5,12,name='alpha_1')
    prior_dict['alpha_2'] = bilby.core.prior.Uniform(1.5,12,name='alpha_2')
    prior_dict['beta'] = bilby.core.prior.Uniform(-4,12,name='beta')
    prior_dict['mmax'] = bilby.core.prior.Uniform(50,200,name='mmax')
    prior_dict['mmin'] = bilby.core.prior.Uniform(2,10,name='mmin')
    prior_dict['b'] = bilby.core.prior.Uniform(0,1,name='b')
    prior_dict['delta_m'] = bilby.core.prior.Uniform(0,10,name='delta_m')
    return prior_dict

def PLS_prior():
    prior_dict=cosmo_prior()
    prior_dict['alpha'] = bilby.core.prior.Uniform(1.5,12,name='alpha')
    prior_dict['beta'] = bilby.core.prior.Uniform(-4,12,name='beta')
    prior_dict['mmax'] = bilby.core.prior.Uniform(50,200,name='mmax')
    prior_dict['mmin'] = bilby.core.prior.Uniform(2,10,name='mmin')
    
    prior_dict['delta_m'] = bilby.core.prior.Uniform(0,10,name='delta_m')
    prior_dict.update({'n'+str(i+1): bilby.core.prior.TruncatedGaussian(0,1,-100,100,name='n'+str(i+1))  for i in np.arange(15)})
    prior_dict['n1'] = 0
    prior_dict['n15'] = 0
    return prior_dict
 
def Double_constraint(params):
    params['constraint']=0
    params['constraint']+=np.sign(params['mua2']-params['mua1'])-1
    params['constraint']+=np.sign(params['mmax2']-params['mmin2']-20)-1
    params['constraint']+=np.sign(params['mmax1']-params['mmin1']-20)-1
    return params

def pl_twospin_priors():
    prior_dict=bilby.prior.PriorDict(conversion_function=Double_constraint)
    prior_dict.update(cosmo_prior())
    prior_dict['alpha1'] = bilby.core.prior.Uniform(1.5,12,name='alpha1')
    prior_dict['alpha_peak'] = bilby.core.prior.Uniform(-4,4,name='alpha_peak')
    prior_dict['lambda_peak'] = bilby.core.prior.Uniform(0,0.5,name='lambda_peak')
    prior_dict['beta'] = bilby.core.prior.Uniform(-4,12,name='beta')
    prior_dict['mmax1'] = bilby.core.prior.Uniform(20,200,name='mmax1')
    prior_dict['mmin1'] = bilby.core.prior.Uniform(2,10,name='mmin1')
    prior_dict['delta_low'] = bilby.core.prior.Uniform(0,10,name='delta_low')
    prior_dict['delta_high'] = bilby.core.prior.Uniform(0,10,name='delta_high')
    
    prior_dict['alpha2'] = bilby.core.prior.Uniform(-4,4,name='alpha2')
    prior_dict['mmax2'] = bilby.core.prior.Uniform(50,200,name='mmax2')
    prior_dict['mmin2'] = bilby.core.prior.Uniform(10,60,name='mmin2')
    prior_dict['r2'] = bilby.core.prior.Uniform(0,1,name='r2')
        
    prior_dict['mua1'] = bilby.core.prior.Uniform(0,1,name='mua1')
    prior_dict['sigmaa1'] = bilby.core.prior.Uniform(0.04,0.5,name='sigmaa1')
    prior_dict['mua2'] = bilby.core.prior.Uniform(0,1,name='mua2')
    prior_dict['sigmaa2'] = bilby.core.prior.Uniform(0.04,0.5,name='sigmaa2')
        
    prior_dict['sigmat'] = bilby.core.prior.Uniform(0.1,4,name='sigmat')
    prior_dict['zeta'] = bilby.core.prior.Uniform(0.,1,name='zeta')
    prior_dict['constraint'] = bilby.prior.Constraint(minimum=-0.1, maximum=0.1)
    return prior_dict

def pls_twospin_priors():
    priors=pl_twospin_priors()
    priors.pop('alpha_peak')
    priors.pop('lambda_peak')
    priors.update({'n'+str(i+1): TG(0,1,-100,100,name='n'+str(i+1))  for i in np.arange(15)})
    priors.update({'n1':0,'n'+str(15): 0})
    return priors

def pp_twospin_priors():
    priors=pl_twospin_priors()
    priors.pop('alpha_peak')
    priors.pop('delta_high')
    priors['mu_g'] = bilby.core.prior.Uniform(20,50,name='mu_g')
    priors['sigma_g'] = bilby.core.prior.Uniform(0.4,10,name='sigma_g')
    
    return priors
