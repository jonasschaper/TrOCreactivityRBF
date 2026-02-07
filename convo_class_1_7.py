# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 16:07:01 2021

@author: jonas
"""
import ray
from scipy.linalg import toeplitz
from scipy.stats import truncnorm

import numpy as np
import scipy.stats
import os, sys
import json
import scipy.integrate as integrate
#from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from lmfit import minimize, Parameters, report_fit
from scipy import interpolate
import warnings
warnings.simplefilter("always", RuntimeWarning)

@ray.remote(memory=2 * 1024 * 1024 * 1024)
class convolute_trocs:
    """
    Convolution-based reactive transport model for TrOC breakthrough analysis.

    This class implements a one-dimensional convolution framework to simulate
    contaminant transport between a surface-water boundary and groundwater
    observation points using a residence time distribution (RTD). The RTD is
    parameterized via a Peclet-number-based analytical solution and combined
    with optional first-order decay. Model parameters are inferred by minimizing
    a weighted least-squares objective using lmfit, and the class is designed
    to be executed in parallel via Ray as part of a DREAM MCMC workflow.

    The model supports both short (≈17 days) and long (>17 days) input time
    series, dynamically switching between direct boundary forcing and
    upstream-convolved forcing depending on well position and data availability.

    Key features
    ------------
    - Peclet-based residence time distribution (RTD)
    - Time-domain convolution using Toeplitz matrices
    - Optional first-order decay during transport
    - Error-weighted likelihood formulation with nugget term
    - Parallel execution via Ray actors
    - Designed for repeated execution within DREAM MCMC chains

    Parameters
    ----------
    tau_dat : array-like
        Empirical or sampled residence times used to define prior distributions.
    params : lmfit.Parameters
        Parameter container used for optimization (typically includes decay rate k).
    meas_conc : dict
        Dictionary containing observed concentrations and uncertainties:
        {
            'cmea'            : measured concentrations at the well,
            'cmea_err'        : associated measurement errors,
            'cmea_inloc'      : concentrations at the upstream location,
            'cmea_inloc_err'  : errors at upstream location,
            'compriv'         : river boundary concentrations,
            'compriverr'      : river concentration errors,
            't_cmeas'         : measurement times,
            'max_time'        : maximum simulation time
        }
    run_info : dict
        Run configuration dictionary containing:
        {
            'well'         : well identifier,
            'withinrange'  : whether measurements lie within convolution range,
            'attempt_conv' : flag to enable convolution,
            'comp'         : compound name
        }
    hours : int
        Temporal aggregation step (hours per model time step).
    ID : int
        Unique model identifier (used in parallel execution).

    Internal state
    --------------
    The class stores intermediate model states such as:
    - RTD shape and truncation time (t99)
    - Convolved concentration time series
    - Fitted decay rate and RMSE
    - Removal metrics based on mass balance
    - Randomized concentration realizations for uncertainty propagation

    Main methods
    ------------
    update_params(currents, Pe_based, sw_cin_ts)
        Updates transport parameters and boundary conditions.

    convo_run()
        Performs time-domain convolution between input concentrations and RTD.

    model_run(max_time)
        Executes parameter fitting, convolution, and diagnostic calculations.

    get_results()
        Returns model diagnostics and time series required by DREAM.

    write_cmeas(dirName)
        Writes measured concentrations to disk for reproducibility.

    Notes
    -----
    - The model assumes one-dimensional advective–dispersive transport.
    - RTDs are truncated at a high cumulative mass fraction (default 99.9%).
    - Measurement errors are treated as Gaussian with an added nugget term.
    - The class is not intended as a general-purpose transport solver, but as
      a calibrated inversion component within a specific hydrochemical workflow.

    This class represents an archival research implementation and is provided
    for transparency and reproducibility rather than general reuse.
    """

        
    
    def __init__(self, tau_dat,
                       params,
                       meas_conc,
                       run_info,
                       hours,
                       ID):
        
        self.ID = ID
        self.tau = tau_dat 
        self.withinrange = run_info["withinrange"]
        self.well            = run_info["well"]
        self.attempt_conv = run_info["attempt_conv"]
        
        self.comp            = run_info["comp"]


        self.baselen = 17#16.5 # in days
        
        
        self.normal_cmea = True

        
        self.hstp = 1 # days (was hours) # 
        self.shift = 0
        self.t97 = 99999

        self.relc_p  = np.nan
        self.relc_r  = np.nan
        self.relmc_p = np.nan
        self.relms_p = np.nan
        self.rels_p  = np.nan
        self.rels_r  = np.nan

        self.params = params
        
        self.cmea = meas_conc["cmea"]
        self.cmea_err = meas_conc["cmea_err"]

        self.cmea_inloc = meas_conc["cmea_inloc"]
        self.cmea_inloc_err = meas_conc["cmea_inloc_err"]
        
        self.compriv= meas_conc["compriv"]
        self.compriverr= meas_conc["compriverr"]
        
        self.t_cmeas = meas_conc["t_cmeas"]
        self.max_time= meas_conc["max_time"]


        
        self.swts_y = 0
        
        self.mu_tau  = 40
        self.alpha_L = 0.01
        self.xp      = 10
        self.k       = 0.0

        self.extent = False
        self.modtime_stp = np.linspace(1, self.max_time,self.max_time)
        
        swin_it = {'input_ts': self.compriv, 'river_ts': self.compriv}

        self.update_params([self.mu_tau , self.xp/self.alpha_L], Pe_based = True, sw_cin_ts = swin_it)
        
    def update_params(self, currents, Pe_based, sw_cin_ts):
        
        self.mu_tau  = currents[0] /24
        if Pe_based == True:
            self.Pe = currents[1]
        else:
            self.alpha_L = currents[1]
            self.xp      = currents[2]
        
        self.swc_t =  sw_cin_ts['input_ts'][-self.max_time:]
        

        self.model_time = len(self.swc_t)

        # determines 99.9% percentile of g(tau)
        self.t99 = self.get_tau_99()

        self.new_t  = np.arange(1, self.model_time + 1, 1)         
        # bring meas time to new timescale 
        self.tmeas_new = self.t_cmeas+(len(self.new_t) -self.baselen)

        # find indices of cmea and cout vector which are within the conv. range 
        self.inrange_idx = np.where(self.tmeas_new > self.get_tau_99() )



        if len(self.new_t) > 17:
            self.river =  sw_cin_ts['river_ts'][-self.max_time:]
            self.criv_tau = interpolate.PchipInterpolator( self.new_t ,self.river)# need to find closes cin value to that point in time t
        else:
            self.criv_mea = interpolate.PchipInterpolator( self.new_t ,self.compriv)# need to find closes cin value to that point in time t


    def get_tau_99(self, tau_max=None, n_points=10000, mass_frac=0.999):
        # Generate tau values
        if tau_max is None:
            tau_max = 5 * self.mu_tau if self.Pe > 1 else 20 * self.mu_tau
        tau = np.linspace(1e-6, tau_max, n_points)
    
        # Calculate RTD
        g_tau = self.g(tau)
        
        # Normalize to make it a true PDF
        g_tau /= np.trapezoid(g_tau, tau)
    
        # Compute CDF
        cdf = cumulative_trapezoid(g_tau, tau, initial=0)
    
        # Interpolate to find tau where CDF = 0.99
        cdf_interp = interp1d(cdf, tau, bounds_error=False, fill_value="extrapolate")
        tau_cutoff = cdf_interp(mass_frac)
        #, tau, g_tau, cdf
        #print(float(tau_cutoff))
        return float(tau_cutoff)
    

    def convo_run(self):
        
        # time vectors updated update_params() function 
        self.cin_tau = interpolate.PchipInterpolator(self.new_t, self.swc_t)# need to find closes cin value to that point in time t
   
        # create finely scaled time vector für convolution 
        dt = 0.01     # smaller step for better accuracy
        tvec = np.arange(dt, self.model_time + dt, dt)            
        cinfine = self.cin_tau(tvec)

        # create short, finely scales time vector of transfer function 
        # note that transfer function should be shorter than c_in(t)
        tvecs =    tvec[np.where(tvec < self.get_tau_99())]
        
        rtd = self.g((tvecs))+1e-8
        rtd /= np.trapezoid(rtd, tvecs)  # or rtd /= np.sum(rtd) if using dt = 1

        lam = self.k          # decay rate (per day)
        # Apply decay
        tau = np.arange(len(rtd)) * dt
        if self.attempt_conv:
            rtd_decay = rtd * np.exp(-lam * tau)
        else: 
            rtd_decay= rtd
            
        self.rtd = rtd_decay

        N = len(cinfine)
        M = len(self.rtd )

        # Create Toeplitz matrix of shape (N + M - 1) × M
        c_in_pad = np.concatenate([cinfine, np.zeros(M - 1)])
        first_col = c_in_pad[:N + M - 1]
        first_row = np.zeros(M)
        first_row[0] = cinfine[0]
        X = toeplitz(first_col, first_row)

        c_out = X @ self.rtd  # Length N + M - 1

        c_out = c_out* dt
        c_out *=1 
        c_out += 1e-8

        # trimm c_out by cutting away inital and final values 
        c_out_trimmed = c_out[M:N]
        self.cout_int = interpolate.PchipInterpolator(tvec[M:], c_out_trimmed)# need to find closes cin value to that point in time t

        
        self.cout    = self.cout_int(tvec[M:]) 
        self.tf      = tvec[M:]
        self.t_org   = self.new_t
        self.cin     = cinfine[M:]
        self.cmod =self.tmeas_new[self.inrange_idx]
        
        # get cmod at at org time scales for the measurements in convolution range
        self.cmod = self.cout_int(self.tmeas_new[self.inrange_idx])

        # for exporting 

        self.tr1 = self.new_t[self.new_t > self.tf[0]] # find times where convolution makes sense
        self.tr =  self.tr1[self.cout_int(self.tr1)>1e-7]

        self.cr =  self.cout_int(self.tr)
        self.ci =   self.swc_t


   
    def trungauss (self, mu, sdd):
        if sdd < 0.001: sdd = mu*0.05

        a, b = (0 - mu) / sdd, np.inf
        trunc_gauss = truncnorm(a=a, b=b, loc=mu, scale=sdd)
       
        return trunc_gauss.rvs(size=1)[0]

    
    def weighted_mean(self, cmeas, err): 
        
        safe_stderr = np.clip(err, 1e-3, None)
        w = np.ones(len(safe_stderr))
    
        if self.normal_cmea: 

            nugget = 0.025  # in mu g/L
            safe_stderr2 = np.sqrt(safe_stderr**2 + nugget**2)

            w = 1/ safe_stderr2**2

        mea = np.sum(w*cmeas)/np.sum(w)
        sdd = np.sqrt(np.sum(w*(cmeas-mea)**2)/np.sum(w))
        return mea, sdd

    def g(self, tau):
        ' Expressed via Peclet number'
        gt = np.sqrt( (self.mu_tau *self.Pe) / (  np.pi * (tau**3) *4 ) ) * np.exp(- ( self.Pe*(self.mu_tau-tau)**2 )  / (4*self.mu_tau*tau ) )
        self.gt = gt        
        return self.gt

    def objective_fun(self, params):
        
        self.k =  params["k"].value
        self.convo_run()
        res = self.cmea[self.inrange_idx] - self.cmod # np.array(self.cout)[self.mask_obs]
         
        if self.normal_cmea: 

            err = np.clip(self.cmea_err[self.inrange_idx], 1e-3, None)
            # Combine reported error and nugget uncertainty in quadrature
            nugget = 0.025  # in mu g/L
            safe_stderr = np.sqrt(err**2 + nugget**2)
            
            res_normalized = res / safe_stderr            
            res = res_normalized
            
        return res

    def model_run(self,max_time): 


        #if sum(self.mask_obs) == 4 and self.attempt_conv:
        #if len(self.inrange_idx[0]) > 2 and self.attempt_conv:
        if len(self.inrange_idx[0]) >= 2 and self.attempt_conv:

            mini_res = minimize(self.objective_fun, self.params )        
    
            self.k = mini_res.params['k'].value
            self.mini_res = mini_res
            self.convo_run()

            
            res = self.cmea[self.inrange_idx] - self.cmod # np.array(self.cout)[self.mask_obs]
            self.rmse = np.sqrt(np.mean((res)**2 ))
            
            
            rtd_len = self.get_tau_99()/2
            # bring convolution scale to sampling scale 
            tout_sample_ts = self.tf -(len(self.new_t) -self.baselen)
            
            if len(self.new_t) > 17:
                
                keep_times_cout_sts = tout_sample_ts[ (tout_sample_ts > rtd_len) ]
                #keep_times_cout_cts = self.tf[ (tout_sample_ts > rtd_len) ]
                keep_times_cout_cts = self.tf[ (tout_sample_ts > 0) ]
                # keeps values of sampling period (last 17 days)
                
            
                # keep times for cinn if cinn is long 
                cin_keep = self.cin_tau(keep_times_cout_cts-rtd_len)
                criv     = self.criv_tau(keep_times_cout_cts-rtd_len)
                cou_keep = self.cout_int(keep_times_cout_cts) 
                assert len(cin_keep) == len(cou_keep)
            
            else:
                # sampling time scale & convolution time scale identical     
                # keep times for cinn if cinn is short (17d)
                if rtd_len < 7:
                    # keep_times defines times in cin vector to keep 
                    keep_times = self.tf[(tout_sample_ts >= rtd_len) & (self.tf < self.tf[-1]-rtd_len)]
                    keep_times = tout_sample_ts[(tout_sample_ts >= rtd_len) & (self.tf < self.tf[-1]-rtd_len)]
                else:
                    keep_times = self.tf[(tout_sample_ts >= rtd_len) ]

             
                cin_keep = self.cin_tau(keep_times)
                # for cout, shift keep_times so values are kept up untul cout_end
                cou_keep = self.cout_int(keep_times+rtd_len)
                criv     = self.criv_mea(keep_times)
                
                assert len(cin_keep) == len(cou_keep)

            if np.isfinite(np.trapezoid(cin_keep)) and np.trapezoid(cin_keep) != 0:
            
                self.relc_p = 1-(np.trapezoid(cou_keep)/np.trapezoid(cin_keep) )
                self.relc_r = 1-(np.trapezoid(cou_keep)/np.trapezoid(criv) )
            
            else:
                print("Warning: Input integral is zero or invalid; cannot compute removal.")
                print(rtd_len)
                print("Integral:", np.trapezoid(cin_keep))

            #self.rmse = np.sqrt(np.mean((self.cmea- np.array(self.cout)[np.isin( self.tt ,(self.t_cmeas+self.shift-1))])**2 ))
            

            
            self.relmc_p = (np.trapezoid(cin_keep)-(np.trapezoid(cou_keep)))/ np.trapezoid(criv)

        else:
            #print('Warning: River time series too short; estimating simple k only')

            self.k  ,self.relc_p, self.relc_r ,self.rmse = -999, -999, -999,  -999
            self.cr = self.cmea; self.tr = 0
            self.ci = self.cmea_inloc
            self.t99,self.relmc_p = -999, -999



        mea_out, sdd_out  = self.weighted_mean(cmeas = self.cmea, err=  self.cmea_err) 
        mea_inn, sdd_inn  = self.weighted_mean(cmeas = self.cmea_inloc,err  = self.cmea_inloc_err) 
        mea_riv, sdd_riv  = self.weighted_mean(cmeas = self.compriv, err = self.compriverr) 


        
        mean = np.mean(self.cmea)      
        sd = np.std(self.cmea)        
        
        phi = np.sqrt(sd**2 + mean**2)
        mu = np.log(mean**2 / phi)
        sigma = np.sqrt(np.log(phi**2 / mean**2))

        self.cmea_out = np.random.lognormal(mean=mu, sigma=sigma, size=1)[0]
        self.cmea_out = self.trungauss(mea_out, sdd_out)
        
        mean = np.mean(self.cmea_inloc)      
        sd = np.std(self.cmea_inloc)        
        
        phi = np.sqrt(sd**2 + mean**2)
        mu = np.log(mean**2 / phi)
        sigma = np.sqrt(np.log(phi**2 / mean**2))
        
        self.cmea_inn = np.random.lognormal(mean=mu, sigma=sigma, size=1)[0]
        self.cmea_inn = self.trungauss(mea_inn, sdd_inn)


        mean = np.mean(self.compriv)      
        sd = np.std(self.compriv)        
        phi = np.sqrt(sd**2 + mean**2)
        mu = np.log(mean**2 / phi)
        sigma = np.sqrt(np.log(phi**2 / mean**2))
        
        self.compriv_mea = np.random.lognormal(mean=mu, sigma=sigma, size=1)[0]
        self.compriv_mea = self.trungauss(mea_riv, sdd_riv)
        
        
        self.rels_p = 1- self.cmea_out/self.cmea_inn
        self.rels_r = 1- self.cmea_out/self.compriv_mea

        self.relms_p = (self.cmea_inn-self.cmea_out)/self.compriv_mea


        self.k_simple = np.log( self.cmea_inn/self.cmea_out)/self.mu_tau
        
        
    def get_results(self):
       tmeas =  self.t_cmeas+ self.shift

       return    self.rmse,  self.k, self.t99,  np.flip(self.cr), np.flip(self.ci), np.flip(self.tr), tmeas, self.k_simple, self.relc_p, self.relc_r, self.rels_p , self.rels_r, self.relmc_p, self.relms_p
            
       
    def write_cmeas(self, dirName):
        
        tmeas =  self.t_cmeas#+ self.shift
        cmea = self.cmea

        f = open(dirName +'/' + 'cmea' +'.txt','w')
        f.write( 'tmeas')
        f.write(';')   
        f.write( 'cmeas')
        f.write('\n')    
        for j in range(0,len(tmeas)):
                    f.write('%g' % tmeas[j])
                    f.write(';')   
                    f.write('%g' % cmea[j])
                    f.write('\n')    
        f.close()


                
def export_conres(metadir, comp_res , comp):
    
    
    finalres = metadir + '/convolution_res/'

    if not os.path.exists(finalres): os.mkdir(finalres); print("Directory " , finalres ,  " Created ")
    else:  print("Directory " , finalres ,  " already exists")

    f = open(finalres + comp+ '_res' +'.txt','w')
   
        
    for key in comp_res.keys():
        #print(key)
        for well in comp_res[key].keys():
            #print(well)
            f.write(key + " ")
            f.write(well)
            sd = comp_res[key][well]["sd"]
            if comp_res[key][well]["sd"] is None:
                sd = 0.0000
            else:
                sd = comp_res[key][well]["sd"]
            f.write(' %g %g \n' % (round(comp_res[key][well]["k"],5), round(sd,5) ))
    f.close()   

    with open(finalres + comp +'_conres.txt', 'w') as file:
         file.write(json.dumps(comp_res)) # use `json.loads` to do the reverse

