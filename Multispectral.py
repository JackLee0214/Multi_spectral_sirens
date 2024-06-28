import icarogw
import os
import pickle
import bilby
from bilby.core.prior import Uniform
from bilby.core.prior import TruncatedGaussian as TG
from bilby.core.prior import Beta as Bt
from icarogw.analyses.cosmo_pop_rate_marginalized import hierarchical_analysis

from icarogw.posterior_samples import posterior_samples
import h5py
import numpy as np
import sys

from scipy.interpolate import interp1d, splev, splrep
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u



run=1
outdir='results'
label=['pp','pls','pls_twospin'][int(sys.argv[1])]
add_label=''


withspin=0
snr_cut=11
ifar_cut=4

with open('./data/BBH_snr_11_Nobs_42.pickle', 'rb') as fp:
    samples, ln_evidences = pickle.load(fp)


from mass_cosmo_spin_models import cosmo_prior, PP_prior, TPL_prior, BPL_prior, PLS_prior, pl_twospin_priors, pls_twospin_priors,pls_twospin_model_prior,pl_twospin_model_prior,LIGO_model_prior

######################################################
#likelihood
######################################################
#redshift

from icarogw.priors.redshift import redshift_prior
def madau(z,gamma,zp,kappa,cosmo_conversion):
    paras=dict(gamma=gamma,zp=zp,kappa=kappa)
    z_model=redshift_prior(cosmo_conversion,'madau',paras)
    return z_model.prob(z)+1e-100
    
#mass
from mass_cosmo_spin_models import pl_twospin_model_prior, pls_twospin_model_prior,LIGO_model_prior, pp_twospin_model_prior

def pl_twospin_m_z_model(m1,m2,z,cosmo_conversion,hyper_params_dict,name):
    model=pl_twospin_model_prior(name,hyper_params_dict)
    gamma=hyper_params_dict['gamma']
    kappa=hyper_params_dict['kappa']
    zp=hyper_params_dict['zp']
    return model.mass_prob(m1,m2)*madau(z,gamma,zp,kappa,cosmo_conversion)

def pls_twospin_m_z_model(m1,m2,z,cosmo_conversion,hyper_params_dict,name):
    model=pls_twospin_model_prior(name,hyper_params_dict)
    gamma=hyper_params_dict['gamma']
    kappa=hyper_params_dict['kappa']
    zp=hyper_params_dict['zp']
    return model.mass_prob(m1,m2)*madau(z,gamma,zp,kappa,cosmo_conversion)
    
def pp_twospin_m_z_model(m1,m2,z,cosmo_conversion,hyper_params_dict,name):
    model=pp_twospin_model_prior(name,hyper_params_dict)
    gamma=hyper_params_dict['gamma']
    kappa=hyper_params_dict['kappa']
    zp=hyper_params_dict['zp']
    return model.mass_prob(m1,m2)*madau(z,gamma,zp,kappa,cosmo_conversion)

def LIGO_m_z_model(m1,m2,z,cosmo_conversion,hyper_params_dict,name):
    model=LIGO_model_prior(name,hyper_params_dict)
    gamma=hyper_params_dict['gamma']
    kappa=hyper_params_dict['kappa']
    zp=hyper_params_dict['zp']
    return model.mass_prob(m1,m2)*madau(z,gamma,zp,kappa,cosmo_conversion)

#conversion
from mass_cosmo_spin_models import cosmo_conversion

###########
#hyper_prior
def PP_hyper_prior(dataset,Om0,H0,alpha,beta,mmin,mmax,mu_g,sigma_g,lambda_peak,delta_m,gamma,zp,kappa,mu_a=None,sigma_a=None,sigma_t=None,zeta=None):
    paras=dict(alpha=alpha,beta=beta,mmin=mmin,mmax=mmax,mu_g=mu_g,sigma_g=sigma_g,lambda_peak=lambda_peak,delta_m=delta_m)
    model=LIGO_model_prior('BBH-powerlaw-gaussian',paras)
    conv=cosmo_conversion(Om0=Om0,H0=H0)
    dL = dataset['d']
    M1,M2 = dataset['M1'],dataset['M2']
    a1, a2, ct1, ct2 = dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1, m2, z = conv.detector_frame_to_source_frame(M1,M2,dL)
    if withspin:
        new_prior_term = model.mass_spin_prob(m1, m2, a1, a2, ct1, ct2,mu_a,sigma_a,sigma_t,zeta)*madau(z,gamma,zp,kappa,conv)
    else:
        new_prior_term = model.mass_prob(m1, m2)*madau(z,gamma,zp,kappa,conv)
    jac_prior=np.abs(conv.detector_to_source_jacobian(z))
    original_prior=dL**2
    return new_prior_term/jac_prior/original_prior

def BPL_hyper_prior(dataset,Om0,H0,alpha_1,alpha_2,beta,mmin,mmax,b,delta_m,gamma,zp,kappa,mu_a=None,sigma_a=None,sigma_t=None,zeta=None):
    paras=dict(alpha_1=alpha_1,alpha_2=alpha_2,beta=beta,mmin=mmin,mmax=mmax,b=b,delta_m=delta_m)
    model=LIGO_model_prior('BBH-broken-powerlaw',paras)
    conv=cosmo_conversion(Om0=Om0,H0=H0)
    dL = dataset['d']
    M1,M2 = dataset['M1'],dataset['M2']
    a1, a2, ct1, ct2 = dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1, m2, z = conv.detector_frame_to_source_frame(M1,M2,dL)
    if withspin:
        new_prior_term = model.mass_spin_prob(m1, m2, a1, a2, ct1, ct2,mu_a,sigma_a,sigma_t,zeta)*madau(z,gamma,zp,kappa,conv)
    else:
        new_prior_term = model.mass_prob(m1, m2)*madau(z,gamma,zp,kappa,conv)
    jac_prior=np.abs(conv.detector_to_source_jacobian(z))
    original_prior=dL**2
    return new_prior_term/jac_prior/original_prior

def TPL_hyper_prior(dataset,Om0,H0,alpha,beta,mmin,mmax,gamma,zp,kappa,mu_a=None,sigma_a=None,sigma_t=None,zeta=None):
    paras=dict(alpha=alpha,beta=beta,mmin=mmin,mmax=mmax)
    model=LIGO_model_prior('BBH-powerlaw',paras)
    conv=cosmo_conversion(Om0=Om0,H0=H0)
    dL = dataset['d']
    M1,M2 = dataset['M1'],dataset['M2']
    a1, a2, ct1, ct2 = dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1, m2, z = conv.detector_frame_to_source_frame(M1,M2,dL)
    if withspin:
        new_prior_term = model.mass_spin_prob(m1, m2, a1, a2, ct1, ct2,mu_a,sigma_a,sigma_t,zeta)*madau(z,gamma,zp,kappa,conv)
    else:
        new_prior_term = model.mass_prob(m1, m2)*madau(z,gamma,zp,kappa,conv)
    jac_prior=np.abs(conv.detector_to_source_jacobian(z))
    original_prior=dL**2
    return new_prior_term/jac_prior/original_prior
    
def PLS_hyper_prior(dataset,Om0,H0,alpha,beta,mmin,mmax,delta_m,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,gamma,zp,kappa,mu_a=None,sigma_a=None,sigma_t=None,zeta=None):
    paras=dict(alpha=alpha,beta=beta,mmin=mmin,mmax=mmax,delta_m=delta_m,\
        n1=n1,n2=n2,n3=n3,n4=n4,n5=n5,n6=n6,n7=n7,n8=n8,n9=n9,n10=n10,n11=n11,n12=n12,n13=n13,n14=n14,n15=n15)
    model=LIGO_model_prior('BBH-powerlaw-spline',paras)
    conv=cosmo_conversion(Om0=Om0,H0=H0)
    dL = dataset['d']
    M1,M2 = dataset['M1'],dataset['M2']
    a1, a2, ct1, ct2 = dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1, m2, z = conv.detector_frame_to_source_frame(M1,M2,dL)
    if withspin:
        new_prior_term = model.mass_spin_prob(m1, m2, a1, a2, ct1, ct2,mu_a,sigma_a,sigma_t,zeta)*madau(z,gamma,zp,kappa,conv)
    else:
        new_prior_term = model.mass_prob(m1, m2)*madau(z,gamma,zp,kappa,conv)
    jac_prior=np.abs(conv.detector_to_source_jacobian(z))
    original_prior=dL**2
    return new_prior_term/jac_prior/original_prior
    
def PL_TWOSPIN_hyper_prior(dataset,Om0,H0,alpha1,beta,mmin1,mmax1,alpha_peak,delta_low,delta_high,lambda_peak,mmin2,mmax2,alpha2,mua1,sigmaa1,mua2,sigmaa2,r2,sigmat,zeta,gamma,zp,kappa):
    hyper_params_dict=dict(alpha1=alpha1,beta=beta,mmin1=mmin1,mmax1=mmax1,alpha_peak=alpha_peak,delta_low=delta_low,delta_high=delta_high,\
        lambda_peak=lambda_peak,mmin2=mmin2,mmax2=mmax2,alpha2=alpha2,mua1=mua1,sigmaa1=sigmaa1,mua2=mua2,sigmaa2=sigmaa2,r2=r2,sigmat=sigmat,zeta=zeta)
    model=pl_twospin_model_prior('pl_twospin',hyper_params_dict)
    conv=cosmo_conversion(Om0=Om0,H0=H0)
    dL = dataset['d']
    M1,M2 = dataset['M1'],dataset['M2']
    a1, a2, ct1, ct2 = dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1, m2, z = conv.detector_frame_to_source_frame(M1,M2,dL)
    new_prior_term = model.mass_spin_prob(m1, m2, a1, a2, ct1, ct2)*madau(z,gamma,zp,kappa,conv)
    jac_prior=np.abs(conv.detector_to_source_jacobian(z))
    original_prior=dL**2
    return new_prior_term/jac_prior/original_prior

    
def PL_TWOSPIN_hyper_prior(dataset,Om0,H0,alpha1,beta,mmin1,mmax1,alpha_peak,delta_low,delta_high,lambda_peak,mmin2,mmax2,alpha2,mua1,sigmaa1,mua2,sigmaa2,r2,sigmat,zeta,gamma,zp,kappa):
    hyper_params_dict=dict(alpha1=alpha1,beta=beta,mmin1=mmin1,mmax1=mmax1,alpha_peak=alpha_peak,delta_low=delta_low,delta_high=delta_high,\
        lambda_peak=lambda_peak,mmin2=mmin2,mmax2=mmax2,alpha2=alpha2,mua1=mua1,sigmaa1=sigmaa1,mua2=mua2,sigmaa2=sigmaa2,r2=r2,sigmat=sigmat,zeta=zeta)
    model=pl_twospin_model_prior('pl_twospin',hyper_params_dict)
    conv=cosmo_conversion(Om0=Om0,H0=H0)
    dL = dataset['d']
    M1,M2 = dataset['M1'],dataset['M2']
    a1, a2, ct1, ct2 = dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1, m2, z = conv.detector_frame_to_source_frame(M1,M2,dL)
    new_prior_term = model.mass_spin_prob(m1, m2, a1, a2, ct1, ct2)*madau(z,gamma,zp,kappa,conv)
    jac_prior=np.abs(conv.detector_to_source_jacobian(z))
    original_prior=dL**2
    return new_prior_term/jac_prior/original_prior

def PLS_TWOSPIN_hyper_prior(dataset,Om0,H0,alpha1,beta,mmin1,mmax1,delta_low,delta_high,mmin2,mmax2,alpha2,mua1,sigmaa1,mua2,sigmaa2,r2,sigmat,zeta,n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,gamma,zp,kappa):
    hyper_params_dict=dict(alpha1=alpha1,beta=beta,mmin1=mmin1,mmax1=mmax1,delta_low=delta_low,delta_high=delta_high,\
        mmin2=mmin2,mmax2=mmax2,alpha2=alpha2,mua1=mua1,sigmaa1=sigmaa1,mua2=mua2,sigmaa2=sigmaa2,r2=r2,sigmat=sigmat,zeta=zeta,
        n1=n1,n2=n2,n3=n3,n4=n4,n5=n5,n6=n6,n7=n7,n8=n8,n9=n9,n10=n10,n11=n11,n12=n12,n13=n13,n14=n14,n15=n15)
    model=pls_twospin_model_prior('pls_twospin',hyper_params_dict)
    conv=cosmo_conversion(Om0=Om0,H0=H0)
    dL = dataset['d']
    M1,M2 = dataset['M1'],dataset['M2']
    a1, a2, ct1, ct2 = dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1, m2, z = conv.detector_frame_to_source_frame(M1,M2,dL)
    new_prior_term = model.mass_spin_prob(m1, m2, a1, a2, ct1, ct2)*madau(z,gamma,zp,kappa,conv)
    jac_prior=np.abs(conv.detector_to_source_jacobian(z))
    original_prior=dL**2
    return new_prior_term/jac_prior/original_prior


def PP_TWOSPIN_hyper_prior(dataset,Om0,H0,alpha1,beta,mmin1,mmax1,delta_low,mu_g,sigma_g,lambda_peak,mmin2,mmax2,alpha2,mua1,sigmaa1,mua2,sigmaa2,r2,sigmat,zeta,gamma,zp,kappa):
    hyper_params_dict=dict(alpha1=alpha1,beta=beta,mmin1=mmin1,mmax1=mmax1,delta_low=delta_low,lambda_peak=lambda_peak,mu_g=mu_g,sigma_g=sigma_g,\
        mmin2=mmin2,mmax2=mmax2,alpha2=alpha2,mua1=mua1,sigmaa1=sigmaa1,mua2=mua2,sigmaa2=sigmaa2,r2=r2,sigmat=sigmat,zeta=zeta)
    model=pp_twospin_model_prior('pp_twospin_cut',hyper_params_dict)
    conv=cosmo_conversion(Om0=Om0,H0=H0)
    dL = dataset['d']
    M1,M2 = dataset['M1'],dataset['M2']
    a1, a2, ct1, ct2 = dataset['a1'],dataset['a2'],dataset['cos_tilt_1'],dataset['cos_tilt_2']
    m1, m2, z = conv.detector_frame_to_source_frame(M1,M2,dL)
    new_prior_term = model.mass_spin_prob(m1, m2, a1, a2, ct1, ct2)*madau(z,gamma,zp,kappa,conv)
    jac_prior=np.abs(conv.detector_to_source_jacobian(z))
    original_prior=dL**2
    return new_prior_term/jac_prior/original_prior
    
pp_para_names=['alpha','beta','mmin','mmax','mu_g','sigma_g','lambda_peak','delta_m','gamma','zp','kappa']
bpl_para_names=['alpha_1','alpha_2','beta','mmin','mmax','b','delta_m','gamma','zp','kappa']
tpl_para_names=['alpha','beta','mmin','mmax','gamma','zp','kappa']
pls_para_names=['alpha','beta','mmin','mmax','delta_m','n1','n2','n3','n4','n5','n6','n7','n8','n9','n10','n11','n12','n13','n14','n15','gamma','zp','kappa']
twospin_para_name=['alpha1','beta','mmin1','mmax1','alpha_peak','delta_low','delta_high','lambda_peak','mmin2','mmax2','alpha2','mua1','sigmaa1','mua2','sigmaa2','r2','sigmat','zeta','gamma','zp','kappa']
pls_twospin_para_name=['alpha1','beta','mmin1','mmax1','delta_low','delta_high','mmin2','mmax2','alpha2','r2','gamma','zp','kappa','H0','Om0']

from mass_cosmo_spin_models import default_spin_priors, pp_twospin_priors
if withspin:
    priors=default_spin_priors()
    add_label+='_defaultspin'
else:
    priors=dict(mu_a=0.5,sigma_a=0.5,sigma_t=0.1,zeta=0)

if label=='bpl':
    m_z_model=LIGO_m_z_model
    model_name='BBH-broken-powerlaw'
    hyper_prior=BPL_hyper_prior
    priors.update(BPL_prior())
elif label=='pp':
    m_z_model=LIGO_m_z_model
    model_name='BBH-powerlaw-gaussian'
    hyper_prior=PP_hyper_prior
    priors.update(PP_prior())
elif label=='tpl':
    m_z_model=LIGO_m_z_model
    model_name='BBH-powerlaw'
    hyper_prior=TPL_hyper_prior
    priors.update(TPL_prior())
elif label=='pls':
    m_z_model=LIGO_m_z_model
    model_name='BBH-powerlaw-spline'
    hyper_prior=PLS_hyper_prior
    priors.update(PLS_prior())
elif label=='pl_twospin':
    model_name='pl_twospin'
    m_z_model=pl_twospin_m_z_model
    hyper_prior=PL_TWOSPIN_hyper_prior
    priors=pl_twospin_priors()
elif label=='pls_twospin':
    model_name='pls_twospin'
    m_z_model=pls_twospin_m_z_model
    hyper_prior=PLS_TWOSPIN_hyper_prior
    priors=pls_twospin_priors()
    priors['alpha1']=bilby.core.prior.Uniform(0,8,name='alpha1')
elif label=='pp_twospin':
    model_name='pp_twospin'
    m_z_model=pp_twospin_m_z_model
    hyper_prior=PP_TWOSPIN_hyper_prior
    priors=pp_twospin_priors()

priors.pop('R0')
    

injections = pickle.load(open( "./data/O1_O2_O3_det_frame_SNR9.inj", "rb" ))
injections.update_cut(snr_cut=snr_cut,ifar_cut=ifar_cut)  # We are going to update the injections with the new SNR cut to 12
from scipy.special import logsumexp
import icarogw
class new_injections(icarogw.injections.injections_at_detector):

    def update_VT(self,m_z_prior,paras,model_name):
        """
        This method updates the sensitivity estimations.
        """
        H0,Om0=paras['H0'],paras['Om0']
        self.new_cosmo = cosmo_conversion(H0=H0,Om0=Om0)
        self.z_model = redshift_prior(self.new_cosmo,'madau',paras)
        self.m1s, self.m2s, self.z_samples = self.new_cosmo.detector_frame_to_source_frame(self.m1det,self.m2det,self.dldet)


        # Calculate the weights according to Eq. 18 in the document
        #mz_paras=[paras[key] for key in para_names]
        #log_numer = np.log(m_z_prior(self.m1s,self.m2s,self.z_samples,self.new_cosmo,*mz_paras))
        log_numer = np.log(m_z_prior(self.m1s,self.m2s,self.z_samples,self.new_cosmo,paras,model_name))
        log_jacobian_term = np.log(np.abs(self.new_cosmo.detector_to_source_jacobian(self.z_samples)))
        self.log_weights = log_numer-np.log(self.ini_prior)-log_jacobian_term
        # This is the Volume-Time in which we expect to detect. You can multiply it by R_0 Tobs to get the expected number of detections in Gpc^3 yr
        self.VT_fraction=np.exp(logsumexp(self.log_weights))/self.ntotal
        # This is the fraction of events we expect to detect, a.k.a. the selection effect
        #self.VT_sens=self.VT_fraction*z_prior.norm_fact

    def expected_number_detection(self,R0):
        """
        This method will return the expected number of GW detection given the injection set. Tobs is automatically saved in the class creation

        Parameters
        ----------
        R0 : float
            Merger rate in comoving volume in Gpc-3yr-1
        """
        return self.VT_fraction*R0*self.Tobs*self.z_model.norm_fact

LIGO_injections=new_injections(injections.m1d_original,injections.m2d_original,injections.dl_original,injections.ini_prior_original,\
    injections.snr_original,injections.snr_cut,injections.ifar,injections.ifar_cut,injections.ntotal,injections.Tobs)

############
#loglikelihood
from bilby.hyper.likelihood import HyperparameterLikelihood
class Hyper_selection_with_var(HyperparameterLikelihood):
    
    def likelihood_obs_var(self):
        weights = self.hyper_prior.prob(self.data) 
        #print(self.parameters)
        expectations = np.mean(weights, axis=-1)
        #print('exp:',expectations)
        square_expectations = np.mean(weights**2, axis=-1)
        #print('squ:',square_expectations)
        variances = (square_expectations - expectations**2) / (
            self.samples_per_posterior * expectations**2
        )
        #print('var:',variances)
        variance = np.sum(variances)
        Neffs = expectations**2/square_expectations*self.samples_per_posterior
        Neffmin = np.min(Neffs)
        #print(Neffmin)
        return variance, Neffmin


    def log_likelihood(self):
        self.injections=LIGO_injections
        self.injections.update_VT(m_z_model,self.parameters,model_name)

        Neff=self.injections.calculate_Neff()
        #print(Neff)
        # If the injections are not enough return 0, you cannot go to that point. This is done because the number of injections that you have
        # are not enough to calculate the selection effect
        if Neff<=(4*self.n_posteriors):
            return -1e100
            #float(np.nan_to_num(-np.inf))

        ksi=self.injections.gw_only_selection_effect()
        #print(ksi)

        # Calculate the numerator and denominator in Eq. 7 on the tex document  for each event and multiply them

        # Selection effect, see Eq. 18 on paper
        log_denominator = np.log(ksi)*self.n_posteriors

        #R0=self.parameters['R0']
        #Nexp = self.injections.expected_number_detection(R0)
        #log_poissonian_term = self.n_posteriors*np.log(Nexp)-Nexp
        log_poissonian_term = 0

        log_selection=np.log(ksi)*self.n_posteriors
        obs_vars, obs_Neff = self.likelihood_obs_var()
        #print(obs_Neff)
        if ((obs_Neff>20)):
            #print(self.noise_log_likelihood(),self.log_likelihood_ratio(),log_selection,log_poissonian_term)
            return self.noise_log_likelihood() + self.log_likelihood_ratio() - log_selection + log_poissonian_term
        else:
            return -1e100
            #float(np.nan_to_num(-np.inf))

hp_likelihood = Hyper_selection_with_var(posteriors=samples, hyper_prior=hyper_prior, log_evidences=ln_evidences, max_samples=10000)


bilby.core.utils.setup_logger(outdir=outdir, label=label+add_label)
if run:
    from bilby.core.sampler import run_sampler
    result = run_sampler(likelihood=hp_likelihood, priors=priors, sampler='pymultinest', nlive=500, queue_size=1,
                    use_ratio=False, outdir=outdir, label=label+add_label)
else:
    from bilby.core.result import read_in_result as rr
    result = rr('./{}/{}_result.json'.format(outdir,label+add_label))

plot_paras=[key for key in result.search_parameter_keys if key not in ['n1','n2','n3','n4','n5','n6','n7','n8','n9','n10','n11','n12','n13','n14','n15']]
plot_paras=pls_twospin_para_name
result.plot_corner(quantiles=[0.05, 0.95],parameters=plot_paras,filename='./{}/{}_corner.pdf'.format(outdir,label+add_label),smooth=1.,color='green')
