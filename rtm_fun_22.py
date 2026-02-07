# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:39:06 2020

@author: jonas
"""


import numpy as np
import os

import math
import copy
import pandas as pd

import os, sys
import numpy as np

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

import time
import scipy as sc
import scipy.stats
import pandas as pd
from lmfit import minimize, Parameters, report_fit
import psutil


import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import matplotlib.pyplot as plt

from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

def load_troc_strings():

    UT_comps = ["oxypurinol","diclofenac", "olmesartan","candesartan"]#, "trimethoprim", "DEET"]

    bwb_comps = ["sulfamethoxazole",    "benzotriazole",		"carbamazepine",  "gabapentin", "gabapentin-lactam", 
                 "hydrochlorothiazide", "metformin","metoprolol",	 "primidone", "sotalol",
                 "valsartan",     "valsartan acid" , "4-hydroxydiclofenac", "4-formylaminoantipyrine",	 "acesulfame", "bezafibrate" ]

    nonbwb_butrtm = [ "diatrizoic acid","guanylurea", "epoxy-carbamazepine" ,"chlorothiazide",  
                    "irbesartan" , "methylbenzotriazole"  ,  "metoprolol acid" ,
                     "O-desmethylvenlafaxine" ,"sitagliptin"      ,      "tramadol"  ,             "venlafaxine", "iopromid",  "iomeprol"   ]

    nonbwb_butrtm = nonbwb_butrtm #+ debated
    bwb_comps =   UT_comps +bwb_comps
    print(len(nonbwb_butrtm+bwb_comps))
    return bwb_comps, nonbwb_butrtm, UT_comps

def load_paths(workstation, runs ):


    if workstation == False:
        metadir     = "C:/Users/Jonas/OneDrive - UT Cloud/rtm"; os.chdir(metadir)
        multicore = True
        nchains = 8;
        locs = pd.read_excel((metadir + '/rtm2019_geo.xlsx'), sheet_name="xx", engine='openpyxl')  
        #tau_folder      ="C:/Users/jonas/Documents/Projects/rtm/modeling/flopy2020/finished/firstdraft/HRnT_250_6_dis3_Rnin0_Rnz6/dreamres"
        total_runs     = int(800) # 6000 for d = 6 ; but very low acceptance rate so 12k 
        indatfold      = "C:/Users/Jonas/Documents/Projects/rtm/modeling/flopy2025/indata_trocs/" 
    
    else:
        #work station == TRUE
        multicore = True
        # models runs on workstations 
        nchains = 32;# should be multiple of 16
        nchains = 8;
        metadir     = "C:/Users/Jonas/OneDrive - UT Cloud/rtm"; os.chdir(metadir)
    
        indatfold      =  metadir+ "/indata_trocs/"
        locs = pd.read_excel((metadir +'/rtm2019_geo.xlsx'), sheet_name="xx", engine='openpyxl')  
        tau_folder      ="C:/Users/jonas/Documents/rtm/modeling/flopy2020/finished/firstdraft/HRnT_250_6_dis3_Rnin0_Rnz6/dreamres"
        total_runs     = int(runs/nchains) # 6000 for d = 6 ; but very low acceptance rate so 12k 
    indatfold      =  metadir+ "/indata_trocs/"# "C:/Users/Jonas/Documents/rtm/flopy2025/indata_trocs/" 
    
    tau_folder =  metadir + '/HRnT_250_6_dis3_Rnin0_Rnz6/dreamres'
    tau      = np.loadtxt(tau_folder   + '/tau.dat' ) # data = pd.read_csv(filename, sep=" ", header=None)

    return metadir, multicore,locs,tau,nchains, total_runs,indatfold 

def gp_time_series_realization(
    values_early,                   # corresponding early concentrations
    recent_values,         
    dirName_input, 
    comp,
    size=5000,         # corresponding recent concentrations
    start_date="17.01.2019",        # start of full time vector
    end_date="15.06.2019",          # end of full time vector
    n_days_total=148,               # total length of time vector (optional fallback)
    realization_idx=0              # which realization to return
    ):
    
    seed=42                         # for reproducibility

    date_strings = [        "17.01.2019",       "29.01.2019",    "14.02.2019",
        "02.03.2019",        "14.03.2019",        "26.03.2019",        "11.04.2019",        "27.04.2019"    ]
    plot = True
    dates_bwb = pd.to_datetime(date_strings, format="%d.%m.%Y")
    date_strings_early= dates_bwb
            
    
    dates = pd.date_range(start="2019-01-05", end="2019-06-15", freq="D")
    days = (dates - dates[0]).days.to_numpy()  # Time in days from start
    recent_dates = pd.to_datetime(dates[-17:])
    
    if comp in ['oxypurinol',
                'olmesartan',
                'candesartan',
                'trimethoprim',
                'DEET']:
        r1 = dates[-17:-3:]
        r2 = dates[-1:]
        recent_dates = r1.append( r2)        

    start_datetime = pd.to_datetime(start_date, format="%d.%m.%Y")
    end_datetime = pd.to_datetime(end_date, format="%d.%m.%Y")
    full_dates = pd.date_range(start=start_datetime, end=end_datetime, freq="D")
    full_days = (full_dates - full_dates[0]).days.to_numpy()
    
    # Early training dates → day numbers
    dates_early = pd.to_datetime(date_strings_early, format="%d.%m.%Y")
    early_day_numbers = (dates_early - start_datetime).days.to_numpy()
    
    # 2. Prepare observed early data
    values_early_log = np.log(values_early+1e-7)
    
    # 3. Prepare recent fixed values
    last_day_numbers = (recent_dates - start_datetime).days.to_numpy()
    last_17_days = days[-17:]
    
    recent_values_log = np.log(recent_values+1e-7)
    #np.random.seed(seed)

    # 4. Combine all training data
    X_obs = np.concatenate([early_day_numbers, last_day_numbers])[:, None]
    y_obs = np.concatenate([values_early_log, recent_values_log])
    
    print(len(y_obs))

    # Set up and train GP
    ls = 10.0
    if comp in ['oxypurinol']:
        ls = 1.0
    
    
    kernel = ConstantKernel(1.0, (1e-3, 1e3))* RBF(10.0, (1, 300)) + WhiteKernel(0.01, (1e-4, 1))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gp.fit(X_obs, y_obs)
    
    # Predict for only the early part (exclude last 17 days)
    n_total = len(full_dates)
    n_fixed = len(recent_dates)
    full_days = full_days[:, None]
    pred_days = full_days[:-n_fixed]
    
    # Get GP prediction
    pred_days = np.array(pred_days).reshape(-1, 1)  # shape = (n_samples, 1)
    mean_pred, cov_pred = gp.predict(pred_days, return_cov=True)
    
    # Sample  realization
    realizations_early = np.random.multivariate_normal(mean_pred, cov_pred, size=size)
    
    # Reshape recent values
    recent_values_log = recent_values_log.reshape(1, -1)  # shape (1, 17)
    
    # Broadcast to match all realizations
    realization_full = np.hstack([realizations_early, np.tile(recent_values_log, (size, 1))])
    full_real_realizations = np.exp(realization_full)
    
    
    print(gp.kernel_)

  
    if plot:

        kernel_str = str(gp.kernel_)
        import re
        # Extract signal variance
        signal_match = re.search(r'([0-9.]+)\*\*2\s*\*\s*RBF', kernel_str)
        signal_variance = float(signal_match.group(1))**2 if signal_match else np.nan
        
        # Extract length scale
        length_match = re.search(r'RBF\(length_scale=([0-9.]+)\)', kernel_str)
        length_scale = float(length_match.group(1)) if length_match else np.nan
        
        # Extract nugget variance
        nugget_match = re.search(r'WhiteKernel\(noise_level=([0-9.]+)\)', kernel_str)
        nugget_variance = float(nugget_match.group(1)) if nugget_match else np.nan
    


        # 1. Sample one realization
        realization_log = np.random.multivariate_normal(mean_pred, cov_pred, size=size)[realization_idx]
        realization_early = np.exp(realization_log)
        
        # 2. Append fixed tail (untransformed)
        realization_full = np.concatenate([realization_early, recent_values])
        
        # Overwrite for plotting if needed
        realization_full = full_real_realizations[1]
        
        # 3. Compute GP mean and standard deviation
        gp_mean_full = np.concatenate([np.exp(mean_pred), recent_values])
        gp_std_full = np.sqrt(np.concatenate([np.diag(cov_pred), np.zeros(len(recent_values))]))
        
        # 4. Plotting
        plt.figure(figsize=(12, 6))
        
        # GP mean curve
        plt.plot(full_dates, gp_mean_full, color="navy", label="GP mean", linewidth=2)
        
        # One realization
        plt.plot(full_dates, realization_full, color="steelblue", alpha=0.7, label="One realization")
        
        # Observations
        plt.scatter(dates_early, values_early, color="darkorange", label="Early observations", zorder=5)
        plt.scatter(recent_dates, recent_values, color="forestgreen", label="Fixed recent values", zorder=5)
        
        # Labels
        plt.xlabel("Date", fontsize=16)
        plt.ylabel("Concentration (µg L$^{-1}$)", fontsize=16)
        plt.title(f"Gaussian process reconstruction for {comp}", fontsize=16, fontweight="bold")
        
        # Legend with GP kernel parameters (LaTeX formatting)
        textstr = '\n'.join((
            rf"$\sigma_f^2$ = {signal_variance:.3f}",
            rf"$\ell$ = {length_scale:.2f} {' days'}",
            rf"$\sigma_n^2$ = {nugget_variance:.3f}"
        ))
        
        plt.gca().text(
            0.98, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8)
        )
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.tick_params(axis='both', which='minor', labelsize=12)
        # Grid, layout, and save
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.legend(fontsize=14)
        plt.savefig(f"{dirName_input}/{comp}_input.png", dpi=600, bbox_inches='tight')
        plt.close()

    
    # Create DataFrame (each column = one realization)
    df_realizations = pd.DataFrame(full_real_realizations.T)
    # 8. Return as DataFrame
    return df_realizations
    


def export_PP(finaldis, mfmod,PP_kind):

    if PP_kind == "Rn":
        f = open(finaldis +  '/'+ 'pilot_points_Rn'  + '.txt','w')
        dat = mfmod.pilot_points_Rn
    if PP_kind == "PP":
        f = open(finaldis +  '/'+ 'pilot_points'  + '.txt','w')
        dat = mfmod.pilot_points

                
    dim = np.shape(dat)
    for j in range(dim[0]):
        #print(j)
        #f.write('%1g ' % simdata[well][0][j])
        for k in range(dim[1]):
            f.write('%g ' %  float(dat[j][k])       ) 
        f.write('\n')            
    f.close()   

def export_coord(finaldis, mfmod, coor_kind ):
    
    if coor_kind == "yy":
        f = open(finaldis +  '/'+ 'yy'  + '.txt','w')
        dat = mfmod.yy
    if coor_kind == "xx":
        f = open(finaldis +  '/'+ 'xx'  + '.txt','w')
        dat = mfmod.xx
                
    dim = np.shape(dat)
    for j in range(dim[0]):
        #print(j)
        f.write('%g ' % dat[j])
        f.write('\n')            
    f.close()   
    
    
    
def export_ts(finalres, wells, simdata,  label ):

    for well in wells:
        f = open(finalres +  '/'+ well +'_'+ label  + '.txt','w')
        dim = np.shape(simdata[well])
        if dim[1]-1 == 0: # single manual reading 
                f.write('%1g ' % simdata[well][0][0])
                for k in range(dim[0]-1):
                    f.write('%g ' %  simdata[well][k+1][0])        
                f.write('\n')
        else:
            for j in range(dim[1]):
                #print(j)
                f.write('%1g ' % simdata[well][0][j])
                for k in range(dim[0]-1):
                    f.write('%g ' %  simdata[well][k+1][j])        
                f.write('\n')            
        f.close()   
        

def load_input_TrOCs(folder, comp, transient_wells,  ss_locs):
    
    troc_tr = load_trocs(folder  + '/' +  comp  , dattype = '_mea', locs = transient_wells )
    troc_std = load_trocs(folder  + '/' +  comp  , dattype = '_sdd', locs = ss_locs )
    return  troc_tr, troc_std
    


def load_indata(folder, comp, transient_wells, ss_locs):
        
    time, comp_riv  = load_bc_timeseries(folder + 'indata/'+ comp + '/' +  comp + '_SW1'   + ".txt")
    
    # transient calibration data
    troc_tr  = load_trocs(folder + 'indata/'+ comp + '/' +  comp  , locs = transient_wells )

    troc_ghb = load_trocs(folder + 'indata/'+ comp + '/' +  comp  , locs = ['P07'] )
    
    mea= 0
    for i in range(len( troc_ghb['P07'])):
        #print(i)
        mea += troc_ghb['P07'][i][3]
    lowerbc = mea/len(troc_ghb['P07'])



    sstate_troclocs = ss_locs
    
    # mean from transient data 
    troc_stdict = load_trocs(folder + comp + '/' +  comp  , locs = sstate_troclocs )
    sstate_troc = {}
    for key in troc_stdict.keys():
        means = np.mean(troc_stdict[key], axis = 0)
        sstate_troc.update({key: [[0, means[1], means[2], means[3],1]]})

    ini_troclocs = ['P00', 'P01', 'P02','P03' ,'P04', 'P05']

    #only the first element in each transient data series is used is used 
    troc_stdict = load_trocs(folder + comp + '/' +  comp  , locs = ini_troclocs )
    ini_troc = {}
    for key in troc_stdict.keys():
        #print(key)
        means = troc_stdict[key][0]
        ini_troc.update({key: [[0, means[1], means[2], means[3],1]]})


    #------------------------------------------    
    return comp_riv, lowerbc,  ini_troc, troc_tr,  sstate_troc
    

def load_input_heads(folder):
    
    # Boundaries and Cali heads  -------- (are loaded from recovery file upon Restart)
    time, head_ghb  = load_bc_timeseries(folder + 'heads/' +  "watlvl_P07"   + ".txt")
    time, head_riv  = load_bc_timeseries(folder + 'heads/' +  "watlvl_SW1"   + ".txt")
    
    # steady calibration data
    calh_st = load_cali_heads(folder + 'heads/' +  "watlvl_ini" + ".txt"  ,steady= True, locs = [ "P01", "P02", "P03"] )
    # transient calibration data
    calh_tr = load_cali_heads(folder + 'heads/' +  "watlvl_calitrans"  ,steady= False, locs = ['P00', 'P01', 'P03', 'P02', 'P04', 'P05', 'P06', 'P07'] )
    #------------------------------------------
    
    return time, head_riv, head_ghb, calh_st, calh_tr

def load_input_temps(folder, transtemps):

    time, temp_riv  = load_bc_timeseries(folder + 'temp/' +  "Temp_C_SW1"   + ".txt")
    time, temp_ghb  = load_bc_timeseries(folder + 'temp/' +  "Temp_C_P07"   + ".txt")
    
    calT_st = load_cali_heads(folder + 'temp/' +  "Temp_C_ini" + ".txt"  ,steady= True, locs =  ['P00', 'P01', 'P03'] )
    calT_tr = load_cali_heads(folder + 'temp/' +  "Temp_C_calitrans"           ,steady= False, locs = transtemps )
    
    #if 'P00' in transtemps:
    
    calT_st['P00'][0][3] = 18.6
    
    return time, temp_riv, temp_ghb, calT_st, calT_tr
        
def load_input_Rn(folder, well_list ):

    calRn_st_mea = load_cali_heads(folder + 'Rn/' +  "Rn_sstate_mea" + ".txt"  ,steady= True, locs = well_list )
    calRn_st_sdd = load_cali_heads(folder + 'Rn/' +  "Rn_sstate_sdd" + ".txt"  ,steady= True, locs = well_list )

    return calRn_st_mea, calRn_st_sdd
       


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def load_bc_timeseries(datapath):
    f = open(datapath,'r') #opens the parameter file - r = read
    line = f.readlines()
    data = []
    for l in line:
        if l[0] != "#":
            data.append(l)
        #while l[0] != '#':
        #    l=f.readline()
    f.close() 
    
    line = 1
    NBOUND = int((data[0].strip().split())[0])
    
    USTIME = np.zeros(NBOUND)
    USBC = np.zeros(NBOUND)

    for i in range(NBOUND) :
                USTIME[i] = float((data[line+i].strip().split())[0])     
                USBC[i] = float((data[line+i].strip().split())[1]) 
    return USTIME, USBC


def load_bc_timeseries_all(datapath):
    
    '''
    f = open(datapath,'r') #opens the parameter file - r = read
    line = f.readlines()
    data = []
    for l in line:
        if l[0] != "#":
            data.append(l)
        #while l[0] != '#':
        #    l=f.readline()
    f.close() 
    '''
    
    data = pd.read_table(datapath,  sep= " ", header=None)
    
    return data



def load_trocs(datapath, locs, dattype):
    
    trans_heads = {}; 
    for well in locs: 

        f = open(datapath + '_' + str(well) + dattype + '.txt','r') #opens the parameter file - r = read
        line = f.readlines()
        data = []
        for l in line:
            if l[0] != "#":
                data.append(l)
            #while l[0] != '#':
            #    l=f.readline()
        f.close() 
        
        line = 1
        NBOUND = int((data[0].strip().split())[0])

    
        #current_loc = locs[ii]
        head_data = []
        # ii += 1
        for i in range(int(NBOUND)) :
            #iii = i + (int(NBOUND))*int(np.where(well == wellnames)[0])
            #print(iii)
            if (data[line+i].strip().split())[2]  == 'NA': 
                head = 0.01
            else: 
                head  = float( (data[line+i].strip().split())[2]   )   
            depth = float( (data[line+i].strip().split())[3]   )   
            xx    = float( (data[line+i].strip().split())[4]   )   
            time  = float( (data[line+i].strip().split())[1]   )   
            head_data.append([ time, xx, depth, head, NBOUND])            
             
        trans_heads.update({ well :  head_data    } )

    
    return trans_heads


def load_cali_heads(datapath, steady, locs):
    

    #locs = ['P00', 'P07', 'P01', 'P03'] #     locs = [ "P01", "P02", "P03", "P07"] 

    #nlocs = len(locs)
    
    if steady == True:
        all_locs = {}; 

        f = open(datapath,'r') #opens the parameter file - r = read
        line = f.readlines()
        data = []
        for l in line:
            if l[0] != "#":
                data.append(l)
            #while l[0] != '#':
            #    l=f.readline()
        f.close() 
        
        line = 1
        NBOUND = int((data[0].strip().split())[0])
        
        wells = []
        for i in range(int(NBOUND)) :
                wells.append(str( (data[line+i].strip().split())[0]   )   )
    
        wellnames, indices = np.unique(wells, return_inverse=True)
        #locs = wellnames
        nlocs = len(wellnames)
        nt = 1 # number of time steps 
        # ii += 1
        for i in range(int(nlocs)) :
            head_data = []

            well  = str( (data[line+i].strip().split())[0]   )   

            head  = float( (data[line+i].strip().split())[2]   )   
            depth = float( (data[line+i].strip().split())[3]   )   
            xx    = float( (data[line+i].strip().split())[4]   )   
            time  = float( (data[line+i].strip().split())[1]   )   
            
            head_data.append([ time, xx, depth, head, nt])            
             
            all_locs.update({ well :  head_data    } )


        trans_heads = { your_key: all_locs[your_key] for your_key in locs }        
        #nlocs = len(wellnames)
        #NBOUND = 4

    if steady == False:
        trans_heads = {}; 
        for well in locs: 

            f = open(datapath + '_' + str(well) + '.txt','r') #opens the parameter file - r = read
            line = f.readlines()
            data = []
            for l in line:
                if l[0] != "#":
                    data.append(l)

            f.close() 
            
            line = 1
            NBOUND = int((data[0].strip().split())[0])

            head_data = []
            # ii += 1
            for i in range(int(NBOUND)) :
                #iii = i + (int(NBOUND))*int(np.where(well == wellnames)[0])
                #print(iii)
                head  = float( (data[line+i].strip().split())[2]   )   
                depth = float( (data[line+i].strip().split())[3]   )   
                xx    = float( (data[line+i].strip().split())[4]   )   
                time  = float( (data[line+i].strip().split())[1]   )   
                head_data.append([ time, xx, depth, head, NBOUND])            
                 
            trans_heads.update({ well :  head_data    } )


        
    return trans_heads


def logLike_priors(chain, parfac):


    '''
    mlog(2π)/2-∑_(t=1)^m    - log⁡(σ_d ) -∑_(t=1)^m    - ((〖ỹ〗_t-y_t (x))/σ_d )^2 
    
    L += -1./(2.*err**2) * ((cdata[i]-cmod[loc])**2.)

    '''
    loglike = 0
    # like contribution from gaussian priors 
    for n in range(chain.npars):     
        if chain.uniform[n] == False:
            #loglike          += (-1./(2.*chain.width[n]**2) * ((chain.mean[n]-chain.proposal[n])**2.))*parfac
            loglike          += (-1/2 * np.log(2* np.pi) - np.nansum(np.log(chain.width[n])) - 0.5 *  ((chain.mean[n]-chain.proposal[n]) / chain.width[n]) ** 2  )  
            #print(loglike)
    return loglike



def logLike(model, t_cmeas, cmea, measerr):
#weights = None,  measerror= None
    
    
    '''
    model = convo_models[i]
    t_cmeas =  ttt
    
    '''

    if isinstance(measerr, float):
        err = cmea*measerr

    cmod =  np.array(model.cout)[ np.isin(model.tt ,t_cmeas*6+0.01)]
    likefac = 1

    loglike_fit = ( -1/2*np.log(2*np.pi)*len(cmod)  - np.nansum(np.log(err)) - 0.5*np.sum( ((cmod - cmea) / err) ** 2  )  ) * likefac
   
    rmse2 = np.sqrt(  np.mean((np.array(cmod)-np.array(cmea))**2))  

    return loglike_fit, rmse2 


def param_loglike(chain):
            #chain = D.chains[i] 
    
    loglike = 0
    for i in range(chain.npars):
        if chain.uniform[i] == False:
            loglike += (-1./(2.*chain.width[i]**2) * ((chain.mean[i]-chain.proposal[i])**2.))

    return(loglike)





def load_priors_convo(tau_dat, locs,inloc, well, Pe_based):
    
    Log   = [];   Unif  = [];     Mean  = []
    width = [];   pmin  = [];     pmax  = []
    # unifrom conservative parameter priors 
    

    #x_min = locs["xRp"][np.where(locs["piezo"] == well)[0][0]]
    xx = locs["xRp"][np.where(locs["piezo"] == well)[0][0]]
    
    if inloc not in ["sw"]:
        x_inloc = locs["xRp"][np.where(locs["piezo"] == inloc)[0][0]]
        x_well  = locs["xRp"][np.where(locs["piezo"] == well)[0][0]]

        xx = x_well - x_inloc
        #x_max = x_well - x_inloc
    
    aL_min = 0.01
    aL_max = 0.25

    inloc
    well
    
    priors_al_x =   {  'k':   {'min':0.001,  'max': 1,  'uniform': True, 'log': True },
                 #'xp': {'min': x_min,  'max': x_max,  'uniform': True, 'log': False },
                 'aL':     {'min': 0.01,   'max': 0.25,  'uniform': True, 'log': False  },
                 'mutau':   {'min': np.min(tau_dat)/24,   'max': np.max(tau_dat)/24,  'uniform': True, 'log': False   }
                   }
    
    Pe_min = xx / aL_max
    Pe_max = xx / aL_min
    
    
    priors_Pe=   { 
                  'Pe':    {'min': Pe_min,  'max': Pe_max,  'uniform': True, 'log': False },
                  'mutau': {'min': np.min(tau_dat)/24,   'max': np.max(tau_dat)/24,  'uniform': True, 'log': False   },
                  'k':     {'min':0.001,  'max': 1,  'uniform': True, 'log': True },
                  'rmse':   {'min':0.001,  'max': 1,  'uniform': True, 'log': True },
                  't97':   {'min':0.001,  'max': 1,  'uniform': True, 'log': True },
                   }
      
    if Pe_based == True: 
        priors = priors_Pe
        Log,Unif,Mean,width,pmin,pmax = prior_make(priors,'mutau',Log,Unif,Mean,width,pmin,pmax) # dummy tau 
        Log,Unif,Mean,width,pmin,pmax = prior_make(priors,'Pe',Log,Unif,Mean,width,pmin,pmax)
        Log,Unif,Mean,width,pmin,pmax = prior_make(priors,'k',Log,Unif,Mean,width,pmin,pmax)
        Log,Unif,Mean,width,pmin,pmax = prior_make(priors,'rmse',Log,Unif,Mean,width,pmin,pmax)
        Log,Unif,Mean,width,pmin,pmax = prior_make(priors,'t97',Log,Unif,Mean,width,pmin,pmax)
        Log,Unif,Mean,width,pmin,pmax = prior_make(priors,'k',Log,Unif,Mean,width,pmin,pmax)
        
        Log,Unif,Mean,width,pmin,pmax = prior_make(priors,'k',Log,Unif,Mean,width,pmin,pmax)
        Log,Unif,Mean,width,pmin,pmax = prior_make(priors,'k',Log,Unif,Mean,width,pmin,pmax)
        Log,Unif,Mean,width,pmin,pmax = prior_make(priors,'k',Log,Unif,Mean,width,pmin,pmax)
        Log,Unif,Mean,width,pmin,pmax = prior_make(priors,'k',Log,Unif,Mean,width,pmin,pmax)
        Log,Unif,Mean,width,pmin,pmax = prior_make(priors,'k',Log,Unif,Mean,width,pmin,pmax)
        Log,Unif,Mean,width,pmin,pmax = prior_make(priors,'k',Log,Unif,Mean,width,pmin,pmax)
        #Log,Unif,Mean,width,pmin,pmax = prior_make(priors,'k',Log,Unif,Mean,width,pmin,pmax)


    else:
        priors = priors_al_x
        Log,Unif,Mean,width,pmin,pmax = prior_make(priors,'mutau',Log,Unif,Mean,width,pmin,pmax)
        Log,Unif,Mean,width,pmin,pmax = prior_make(priors,'aL',Log,Unif,Mean,width,pmin,pmax)
        Log,Unif,Mean,width,pmin,pmax = prior_make(priors,'xp',Log,Unif,Mean,width,pmin,pmax)
        Log,Unif,Mean,width,pmin,pmax = prior_make(priors,'k',Log,Unif,Mean,width,pmin,pmax)
    


    # MF Parameters -------------------------------------------------------------------------------
    # a) hydraulic conductivity K 

    #print(priors)
    return Log, Unif, Mean, width, pmin, pmax


def prior_make(priors, param,Log, Unif, Mean, width, pmin, pmax): 
    Log.append(priors[param]['log'] )

    Unif.append(priors[param]['uniform'])
    
    if priors[param]['uniform'] == True:
        Mean.append(priors[param]['min']+ ((priors[param]['max']-priors[param]['min'])/2))
        width.append((priors[param]['max']-priors[param]['min'])/2) 
        pmin.append(priors[param]['min'])
        pmax.append(priors[param]['max'])   

    if priors[param]['uniform'] == False:
        Mean.append(priors[param]['mean'])
        width.append(priors[param]['sd']) 
        pmin.append(priors[param]['mean'] - 2*priors[param]['sd'])
        pmax.append(priors[param]['mean'] + 2*priors[param]['sd'])   

    return Log, Unif, Mean, width, pmin, pmax

def update_params_final(dirName, mf_mod, mode, steady):
    
    """
    Calculates mean parameters from posterior samples and updates the model accordingly
    
    Parameters
    ----------
    dirName : STRING
        directory of .
    model : TYPE
        DESCRIPTION.
    seclocs_array : TYPE
        DESCRIPTION.
    consonly : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    mf_mod = mfmodeltr[0]


    """
    
    filename =  dirName +  '/'+'postpars.dat' 
    
    if mode == 'mfonly':
        
        k0_me = [];k1_me = [];k2_me = []; ss_me = []; sy_mea = [] 
        k0 = [];k1 = [];k2 = []; ss = []; sy = [] 
   
                            
        f = open(filename,'r')
        line = f.readlines()
        data = []
        hk = []
        hypo_K = []   
        for l in line:
            if l[0] != "#":
                data = []
                data.append(l)
                k0.append(float(data[0].strip().split()[0+(1*0)+1]))
                k1.append(float(data[0].strip().split()[0+(1*1)+1]))
                k2.append(float(data[0].strip().split()[0+(1*2)+1]))
                ss.append(float(data[0].strip().split()[0+(1*3)+1]))
                sy.append(float(data[0].strip().split()[0+(1*4)+1]))

        f.close()
        
        res = { 'K0': {'mean': np.mean(k0), 'sd': np.std(k0) , 'posterior': k0 },
                'K1': {'mean': np.mean(k1), 'sd': np.std(k1) , 'posterior': k1  },
                'K2': {'mean': np.mean(k2), 'sd': np.std(k2) , 'posterior': k2  },
                'ss': {'mean': np.mean(ss), 'sd': np.std(ss)  , 'posterior': ss },
                'sy': {'mean': np.mean(sy), 'sd': np.std(sy)  , 'posterior': sy }}
        
        parlist = []

        for key in res.keys():
            
            print(key)

            plt.figure(figsize=(10,8))
            plt.hist(np.array(res[key]['posterior']),bins=10)
            plt.title( (key + ' =  '+ str(round(float(res[key]['mean']),4)) + '+-' 
                                  + str(round(float(res[key]['sd']),4))  )
                                  , fontsize=12, transform=  plt.gca().transAxes)          
            plt.savefig( (dirName + "/" + 'posterior_' + key+  '.png'))
            parlist.append(float(res[key]['mean']))

        

        mf_mod.update_params(    np.array(parlist ), steady = steady)
        mf_mod.model_run()
    return np.array(parlist )





def vstack_tau(iteration,restime, tau,  wells= ['P00', 'P01', 'P02', 'P03', 'P04' , 'P05'] ):
    #wells = ['P00', 'P01', 'P02', 'P03', 'P04' , 'P05']; tau = []
    #restime
    taul = []
    for well in wells:
        taul.append(restime[well][0][2])


    if iteration == 1:
        tau =   taul # keep old values
    else:
        #print('here')
        tau =   np.vstack((tau,taul)) # keep old values

    return tau 


def print_tau( dirName,tau ):
    f = open(dirName +  '/'+'tau.dat','w')
    dim = np.shape(tau)
    for j in range(dim[0]):       
        for k in range(dim[1]):
            f.write('%g ' % tau[j][k])
        f.write('\n')
    f.close()   


