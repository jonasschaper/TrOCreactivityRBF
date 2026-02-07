# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 12:32:30 2018

@author: schaper
"""

import numpy as np
import os
import scipy as sp
import random as rand
import sys
import multiprocessing as mp
import random
# following Jims McCallums Script, DREAM is set up around two classes, DREAM and CHAIN
# The class CHAIN is attached to a DREAM instance 

class P(mp.Process):
    def __init__(self,task_queue, result_queue):
        mp.Process.__init__(self)
        self.tasks = task_queue
        self.results = result_queue
    
    def run(self):
        #loop = True
        while True:
            dum = self.tasks.get() #Gets the model object from the tasks Queue
            if dum is None: #This uses a poison pill to break the process
                break
            dum.model_run() #Solves the model
            self.results.put(dum) #Puts the model back into the results queue
        return



    
class DREAM:
    def __init__(self, nchains, npars, maxsetback, beta0, boundary, ncr = 3, delta = 3, prior = False):
        self.nc = nchains
        self.npars = npars
        self.chains = []
        self.delta = delta 
        self.maxsetback = maxsetback
        self.setbackcount = 0   
        self.nb    = 0
        self.beta0 = beta0
        self.boundary = boundary 
        self.tc     = 0
        self.ct     = 0
        self.c      = 0.1
        self.c_star = 1e-12
        self.ncr    = ncr
        self.nacc   = 0
        self.burn   = True
        self.pCR = np.ones(ncr)/ncr # pCR = selection probabilities
        self.Xstd = np.zeros(npars)
        self.J, self.n_id = [np.zeros(ncr) for _ in range(2)] 
        self.R   = np.ones(npars)*100 # starting R_hats
        for i in range(nchains):
            self.chains.append(chain(npars,prior = prior))

    def par_set(self,log,uniform,mean,width,pmin,pmax):
        for i in range(self.nc):
            self.chains[i].par_set(log,uniform,mean,width,pmin,pmax,rstart = True)
            
            
    def par_reset(self):
        for i in range(self.nc):
            self.chains[i].pars = self.chains[i].current       


     
    def propgen(self):

        for i in range(self.nc): # looping through all chains 

            
            # 1. Identify random chains to use to get chain-dependent part of jump rate dx
            dum = list(range(self.nc)) # eg dum = [0, 1, 2, 3, 4]
            dum.remove(i) # if i = 1 dum will be [0, 2, 3, 4]
            ll = self.nc-1
            
            jump    = np.zeros(self.npars) # dx will be the jump to be added to current state 
            zetas = np.zeros(self.npars) + self.c_star*np.random.randn(self.npars)


            '''
            if self.delta < 0:
                npairs = np.random.randint(1,abs(self.delta)+1) # draws random integer, e.g. 1,2,3 for delta = 3
            else:
                npairs = self.delta
            '''
            # One of Multiple chain pairs are used to create jump  
            npairs = int(rand.sample(range(1,abs(self.delta)+1),1)[0])# how many pairs of chains will be used to generate jump?
            #print(npairs)
            for k in range(npairs): #adds and substracts random chain parameter values npairs times 
                #dum2  = np.random.randint(ll) # randomly selects a chain but not chain i
                #ll = ll-1
                #jump += self.chains[dum[dum2]].current # slects parameters from randomly chosen chain (a) and adds to dx
                #dum.remove(dum[dum2]) # removes the chain a 

                
                dum2  = random.choice(dum) #  and selects another one which cannot be a and i 
                dum.remove(dum2)
                jump += self.chains[dum2].current # now substract chain b parameters from chain a parameters
                #print(dum2)
                
                                         
                #dum3  = np.random.randint(ll) # and selects another one which cannot be a and i
                #ll = ll-1 # now ll = nc -3! one for chains i,a and b
                #jump += -self.chains[dum[dum3]].current # now substract chain b parameters from chain a parameters
                #dum.remove(dum[dum3]) # and remove b from all chains left, now the loop starts again if npairs was chosen to be larger 

                dum3  = random.choice(dum) #  and selects another one which cannot be a and i 
                dum.remove(dum3)
                jump += -self.chains[dum3].current # now substract chain b parameters from chain a parameters
                #print(dum3)
            
            
            if  np.count_nonzero(jump) < self.npars : print("jumpchain=", jump)
            #print("jumpchain=", jump)
            
            # Subspace sampling block ----------------------------------------------------
            CR =  []; cr_list = []
            for ii in range(1, self.ncr+1): # ii will be 1,2,3
                CR.append(ii/self.ncr)
                cr_list.append(ii-1)
            self.chains[i].idd = np.random.choice(cr_list, 1,   p= self.pCR)[0] # choose a CR value (idd) with pCR probability
            #z   = sp.rand(self.npars) # choose random numbers between 0 and 1 and store in vector of length npars
            z = np.random.rand(self.npars)
            A    = np.where( z < CR[ self.chains[i].idd] )  # all values of z <= CR are kept to form the subspace A 
            zero = np.where( z >= CR[ self.chains[i].idd] ) # in code, dx is set to zero for the pars that arent kept 
            d_star = len(A[0]); 
            
            if d_star == 0:  
                d_star = 1   
                jump[ np.where( z == min(z)) ] = 0
                zetas[np.where( z == min(z)) ] = 0
            else: 
                jump[zero] = 0
                zetas[zero] = 0

            # smaller idd vlaues mean smaller threshold CR values and thus a larger subset of samples (less pars to be kept)


            if  np.count_nonzero(jump) == 0: Warning("jump empty")#sys.exit('jump empty')

            # Jump rate block ----------------------------------------------------

            gamma    = 2.38/np.sqrt(2.*npairs*d_star)*self.beta0 # 1.68 for npairs =1, d=1; 1.2 for d=2 and < 1 for any d = 3
            g        = np.random.choice([gamma, 1 ], 1,   p=[0.8, 0.2])
            jumprate = g * (1 + np.random.uniform(-self.c,self.c,1) ) ; #print(jumprate)

            chain_jump = jump * jumprate  # chain dependent part of jump 

            jumpjump = chain_jump + zetas # zetas added 

            if  np.count_nonzero(jumpjump) == 0: sys.exit('jump empty')

            self.chains[i].proposal = np.copy(self.chains[i].current) + jumpjump
            self.chains[i].jump     =  np.copy(jumpjump  )  
            
     
            #print('reltive jump =',self.chains[i].jump/self.chains[i].current*100)
            #print('Proposal =', self.chains[i].proposal[0:6])
            #print('current =', self.chains[i].current)

            # How to handel Boundary crossing issues - section  
            
            # folding and eflect methods 
            # step 1) proposal is so small/large that absolute distance to min/max is larger than range of prior
            # if so, set proposal to mean of the range + plus some random number                             
            # option 2) proposal-min/max distance is smaller than width and we can reflect the proposal

            
            for k in range(self.npars):
                #if self.chains[i].uniform[k] == True:
                    if self.boundary == "folding":


                        if self.chains[i].proposal[k] < self.chains[i].pmin[k]: 
                           if     np.abs(self.chains[i].pmin[k]-self.chains[i].proposal[k]) <  np.abs( self.chains[i].pmax[k]-self.chains[i].pmin[k]):
                                  # absolute amount of over-shooting is smaller than prior range (= max - min)
                                  self.chains[i].proposal[k]         = self.chains[i].pmax[k] - np.abs(self.chains[i].pmin[k]-self.chains[i].proposal[k])
                        
                           elif   np.abs(self.chains[i].pmin[k] - self.chains[i].proposal[k]) ==  np.abs( self.chains[i].pmax[k]-self.chains[i].pmin[k] ):            
                                  # absolute amount of over-shooting is smaller than prior range (= max - min)
                                  self.chains[i].proposal[k]         = self.chains[i].pmin[k] 
                        
                           elif   np.abs(self.chains[i].pmin[k] - self.chains[i].proposal[k]) >  np.abs( self.chains[i].pmax[k]-self.chains[i].pmin[k] ):
                                  # absolute amount of over-shooting is larger than prior range (= max - min)! How much larger?
                                  if np.abs(self.chains[i].pmin[k] - self.chains[i].proposal[k])- np.abs(self.chains[i].pmax[k]-self.chains[i].pmin[k]) > np.abs(self.chains[i].pmax[k]-self.chains[i].pmin[k]):
                                      # absolute amount of over-shooting is more than twice as large as the prior range
                                      self.chains[i].proposal[k]= self.chains[i].pmax[k] - np.abs(self.chains[i].pmin[k] - self.chains[i].proposal[k]) + 2*np.abs(self.chains[i].pmax[k]- self.chains[i].pmin[k] )
                                  else:
                                      # absolute amount of over-shooting is less than twice as large as the prior range
                                      self.chains[i].proposal[k]= self.chains[i].pmax[k] - np.abs(self.chains[i].pmin[k] - self.chains[i].proposal[k]) + 1*np.abs(self.chains[i].pmax[k]- self.chains[i].pmin[k] )
                        
                        elif self.chains[i].proposal[k] > self.chains[i].pmax[k]:
                            
                            if    np.abs(self.chains[i].proposal[k] - self.chains[i].pmax[k]) <  np.abs( self.chains[i].pmax[k]- self.chains[i].pmin[k] ) :
                                  self.chains[i].proposal[k]         = self.chains[i].pmin[k] + np.abs(self.chains[i].pmax[k] - self.chains[i].proposal[k])
                             
                            elif  np.abs(self.chains[i].proposal[k] - self.chains[i].pmax[k]) ==  np.abs( self.chains[i].pmax[k]- self.chains[i].pmin[k] ) :                            
                                  self.chains[i].proposal[k]         = self.chains[i].pmax[k] 
                             
                            elif  np.abs(self.chains[i].proposal[k] - self.chains[i].pmax[k]) >  np.abs( self.chains[i].pmax[k]- self.chains[i].pmin[k] ) :
                                
                                if np.abs(self.chains[i].proposal[k]- self.chains[i].pmax[k])  > 2* np.abs(self.chains[i].pmax[k]- self.chains[i].pmin[k] ):
                                  self.chains[i].proposal[k]= self.chains[i].pmin[k] + np.abs(self.chains[i].proposal[k]- self.chains[i].pmax[k]) - 2*np.abs(self.chains[i].pmax[k]- self.chains[i].pmin[k] )
                                else:
                                  self.chains[i].proposal[k]= self.chains[i].pmin[k] + np.abs(self.chains[i].proposal[k]- self.chains[i].pmax[k]) - 1*np.abs(self.chains[i].pmax[k]- self.chains[i].pmin[k]  )
                                                                   

                        '''
                        if self.chains[i].proposal[k] < self.chains[i].pmin[k]:
                            
                            if  np.abs(self.chains[i].pmin[k] - self.chains[i].proposal[k]) >=  np.abs( self.chains[i].pmax[k]- self.chains[i].pmin[k] ):
                                #while np.abs(self.chains[i].pmin[k] - self.chains[i].proposal[k]) >=  np.abs( self.chains[i].pmax[k]- self.chains[i].pmin[k] ):
                                    self.chains[i].proposal[k] = self.chains[i].mean[k]*np.random.normal(1, 0.1)
                            else:                            
                                self.chains[i].old_proposal_min[k] = self.chains[i].proposal[k]
                                self.chains[i].proposal[k]         = self.chains[i].pmax[k] - np.abs(self.chains[i].pmin[k] - self.chains[i].proposal[k])

                        elif self.chains[i].proposal[k] > self.chains[i].pmax[k]:
                            
                             if  np.abs(self.chains[i].proposal[k] - self.chains[i].pmax[k]) >=  np.abs( self.chains[i].pmax[k]- self.chains[i].pmin[k] ) :
                                 #while np.abs(self.chains[i].proposal[k] - self.chains[i].pmax[k]) >=  np.abs( self.chains[i].pmax[k]- self.chains[i].pmin[k] ) :
                                     self.chains[i].proposal[k] = self.chains[i].mean[k]*np.random.normal(1, 0.1)
                             else:                            
                                 self.chains[i].old_proposal_max[k] = self.chains[i].proposal[k]
                                 self.chains[i].proposal[k]         = self.chains[i].pmin[k] + np.abs(self.chains[i].pmax[k] - self.chains[i].proposal[k])

                        '''
                    elif self.boundary == "eflect":                      

                        if  self.chains[i].proposal[k] < self.chains[i].pmin[k]:
                            if  np.abs(self.chains[i].pmin[k] - self.chains[i].proposal[k]) <  np.abs(self.chains[i].pmax[k]- self.chains[i].pmin[k] ):
                                self.chains[i].proposal[k]         = self.chains[i].pmin[k] + np.abs(self.chains[i].pmin[k] - self.chains[i].proposal[k])
                                
                            elif np.abs(self.chains[i].pmin[k] - self.chains[i].proposal[k]) ==  np.abs(self.chains[i].pmax[k]- self.chains[i].pmin[k] ):
                                self.chains[i].proposal[k] = self.chains[i].pmax[k]
                            
                            elif np.abs(self.chains[i].pmin[k] - self.chains[i].proposal[k]) >  np.abs(self.chains[i].pmax[k]- self.chains[i].pmin[k] ):
                                print("min double", round(self.chains[i].proposal[k],2), 'chain:', str(i), '  parameter:', str(k) )

                                if np.abs(self.chains[i].pmin[k] - self.chains[i].proposal[k])- np.abs(self.chains[i].pmax[k]- self.chains[i].pmin[k] ) > np.abs(self.chains[i].pmax[k]- self.chains[i].pmin[k] ):
                                    self.chains[i].proposal[k] = self.chains[i].pmin[k] + np.abs(self.chains[i].pmin[k] - self.chains[i].proposal[k])- 2*np.abs(self.chains[i].pmax[k]- self.chains[i].pmin[k] )
                                else:
                                    self.chains[i].proposal[k] = self.chains[i].pmin[k] +  np.abs(self.chains[i].pmin[k] - self.chains[i].proposal[k])- np.abs(self.chains[i].pmax[k]- self.chains[i].pmin[k] )
                                
                                print("resolved to", round(self.chains[i].proposal[k],3) )
                          
                        if  self.chains[i].proposal[k] > self.chains[i].pmax[k]:
                            if  np.abs(self.chains[i].proposal[k] - self.chains[i].pmax[k]) <  np.abs(self.chains[i].pmax[k]- self.chains[i].pmin[k] ):
                                self.chains[i].proposal[k]         = self.chains[i].pmax[k] - np.abs(self.chains[i].proposal[k] - self.chains[i].pmax[k])
                                
                            elif np.abs(self.chains[i].proposal[k] - self.chains[i].pmax[k])  ==  np.abs(self.chains[i].pmax[k]- self.chains[i].pmin[k] ):
                                self.chains[i].proposal[k] = self.chains[i].pmin[k]
                            
                            elif np.abs(self.chains[i].proposal[k] - self.chains[i].pmax[k]) >  np.abs(self.chains[i].pmax[k]- self.chains[i].pmin[k] ):
                                print("mxx double", round(self.chains[i].proposal[k],3), 'chain:', str(i), '  parameter:', str(k) )
                                if np.abs(self.chains[i].proposal[k] - self.chains[i].pmax[k]) - np.abs(self.chains[i].pmax[k]- self.chains[i].pmin[k] ) > np.abs(self.chains[i].pmax[k]- self.chains[i].pmin[k] ):
                                    self.chains[i].proposal[k] = self.chains[i].pmax[k] - np.abs(self.chains[i].proposal[k] - self.chains[i].pmax[k]) + 2*np.abs(self.chains[i].pmax[k]- self.chains[i].pmin[k] )
                                else:
                                    self.chains[i].proposal[k] = self.chains[i].pmax[k] -  np.abs(self.chains[i].proposal[k] - self.chains[i].pmax[k]) + np.abs(self.chains[i].pmax[k]- self.chains[i].pmin[k] )
                                print("resolved to", round(self.chains[i].proposal[k],3) )
                        '''   
                        if  self.chains[i].proposal[k] < self.chains[i].pmin[k]:
                            if  np.abs(self.chains[i].pmin[k] - self.chains[i].proposal[k]) >=  np.abs(self.chains[i].pmax[k]- self.chains[i].pmin[k] ) :
                                while np.abs(self.chains[i].pmin[k] - self.chains[i].proposal[k]) >=  np.abs(self.chains[i].pmax[k]- self.chains[i].pmin[k] ) :
                                    self.chains[i].proposal[k] = self.chains[i].mean[k]*np.random.normal(1, 0.1)
                                    #self.chains[i].proposal[k] = self.chains[i].pmin[k] + np.abs(self.chains[i].pmin[k] - self.chains[i].proposal[k]*sp.rand())
                            else:
                                self.chains[i].old_proposal_min[k] = self.chains[i].proposal[k]
                                self.chains[i].proposal[k]         = self.chains[i].pmin[k] + np.abs(self.chains[i].pmin[k] - self.chains[i].proposal[k])
                    
                        elif self.chains[i].proposal[k] > self.chains[i].pmax[k]:
                            if np.abs(self.chains[i].proposal[k] - self.chains[i].pmax[k]) >= np.abs( self.chains[i].pmax[k]- self.chains[i].pmin[k] )  :
                                while np.abs(self.chains[i].proposal[k] - self.chains[i].pmax[k]) >= np.abs( self.chains[i].pmax[k]- self.chains[i].pmin[k] )  :
                                    self.chains[i].proposal[k] = self.chains[i].mean[k]*np.random.normal(1, 0.1)
                            else: 
                                self.chains[i].old_proposal_max[k] = self.chains[i].proposal[k]
                                self.chains[i].proposal[k]         = self.chains[i].pmax[k] - np.abs(self.chains[i].proposal[k] - self.chains[i].pmax[k])
                        '''

                    elif self.boundary == "bound":
                        if self.chains[i].proposal[k] < self.chains[i].pmin[k]:
                            self.chains[i].proposal[k] = self.chains[i].pmin[k]
                        
                        elif self.chains[i].proposal[k] > self.chains[i].pmax[k]:
                            self.chains[i].proposal[k] = self.chains[i].pmax[k] 
                                          
                    elif self.boundary == "random":
                        if self.chains[i].proposal[k] < self.chains[i].pmin[k]:
                             self.chains[i].proposal[k] =  self.chains[i].mean[k] - self.chains[i].width[k] *  np.random.rand() # sp.rand()  
                             #print(round(self.chains[i].proposal[k],6), 'chain', i, "par", k, 'gaussian smaller than min' )
                       
                        elif self.chains[i].proposal[k] > self.chains[i].pmax[k]:
                             self.chains[i].proposal[k] =  self.chains[i].mean[k] + self.chains[i].width[k] *np.random.rand() # sp.rand()
                             #print(round(self.chains[i].proposal[k],6), 'chain', i,"par", k, 'unif larger than max' )
                                        
                    np.random.rand()
                 
                 
        #'''         
        for i in range(self.nc):
            for k in range(self.npars):
                if self.chains[i].proposal[k] < self.chains[i].pmin[k]:
                    print(k,"chain = ", i, self.chains[i].proposal[k], "min crossed")                        
                    print( self.chains[i].old_proposal_min[k] )                    
                    #print(lalala+1)
                if self.chains[i].proposal[k] > self.chains[i].pmax[k]: 
                    print(k,"chain = ",i, self.chains[i].proposal[k], "max crossed")                        
                    print( self.chains[i].old_proposal_max[k] )                    
                    #print(lalala+1)
        #'''                
        self.ct+=1; 
        self.tc+=1;
        print(self.tc, self.ct)

        #for i in range(self.nc): # looping through all chains 
        #    print('Proposal final =', self.chains[i].proposal[0:6])
            

    
    def delm_update(self):
        self.delm = np.zeros(self.ncr)
        self.crct=  np.zeros(self.ncr)
        for i in range(self.nc):
            for j in range(self.npars):
                self.delm[self.dumloc] += (self.chains[i].pars[-1,j] - self.chains[i].pars[-2,j]) **2. /self.Var[j]                
                self.crct[self.dumloc] +=1
                
    def Chain_removal(self,  reset_removal, counts):
        '''
        if reset_removal == True:
            counts = self.ct # counts of current burn-in 
        else:
            counts = self.tc # total number of counts 
        '''

        self.reset = False
        Omega = np.zeros((self.nc))
        n = int(counts/2)
        last = np.zeros((self.nc))
        for i in range(self.nc):
            Omega[i] = np.average(np.array(self.chains[i].likelihood[n:]))
            last[i] = self.chains[i].likelihood[-1]
        best = np.argmax(last)
        IQRmin = np.percentile(Omega,25) - 2 * (np.percentile(Omega,75) - np.percentile(Omega,25))        

        for i in range(self.nc):      
            if self.ct != 0 and self.ct%10 == 0: print("Omega =", Omega[i],"IQR = ",IQRmin)            
            if Omega[i] < IQRmin*1.: # added downscaling factor 
                #print(Omega[i],IQRmin)
                self.chains[i].current = np.copy(self.chains[best].current)
                self.chains[i].Lold    = float(np.copy(self.chains[best].Lold))
                self.reset = True
                self.chains[i].likelihood =  np.copy(self.chains[best].likelihood).tolist()   
                #D.chains[i].likelihood = np.copy(D.chains[i].likelihood)    
       
        if (self.reset == True and  self.setbackcount < self.maxsetback):
            self.reset = False
                             
            self.ct = 0
            self.nacc = 0   
                
            self.setbackcount += 1; print('setback number' , self.setbackcount)
            
            if reset_removal == True:
                # resetting pCR adaption part    
                
                '''
                # resetting pCR adaption stats
                self.pCR = np.ones(self.ncr)/self.ncr # pCR = selection probabilities
                self.std_cal()                
                self.J, self.n_id = [np.zeros(self.ncr) for _ in range(2)] 
                '''                 
                
                # discarding all stored parameters and liklihoods
                for i in range(self.nc):
                    self.chains[i].likelihood = [float(self.chains[i].Lold)]
                    self.chains[i].pars       = self.chains[i].pars[-1,:]
                    #self.chains[i].postpars   = self.chains[i].postpars[-1,:]
            
    def Rget(self):
        
        mean = np.zeros((self.nc,self.npars))
        var = np.zeros((self.nc,self.npars))
        for i in range(self.nc):
            mean[i,:], var[i,:], n = self.chains[i].varget() 
        
        W = np.average(var,axis=0)
        B = np.var(mean,axis=0) * n
        self.Var = (1. -1./n) * W + 1. /n * B
        self.R = np.sqrt(self.Var/W)
        
    def std_cal(self):
        stdmat = np.zeros((self.nc, self.npars))
        for i in range(self.nc):    
            stdmat[i,:] = self.chains[i].current               
        self.Xstd = stdmat.std(0)   
    
    def pCR_update(self):
        self.pCR = (self.J/self.n_id) / np.nansum (self.J / self.n_id)         
        '''        
        D.pCR = (D.J/D.n_id) / np.nansum (D.J / D.n_id)         
        '''
    def restart(self, dirName, Restart):
        if Restart:
            rst = np.loadtxt(dirName+ '/'+'current.dat')
            for i in range(self.nc):
                    self.chains[i].Lold = rst[i,0]
                    self.chains[i].current = rst[i,1:]
        #if recovery:
            f = open(dirName +  '/'+'recovery.dat','r') #opens the parameter file - r = read
            line = f.readlines()
            indata = []
            for l in line:
                if l[0] != "#":
                    indata.append(l)
                #while l[0] != '#':
                #    l=f.readline()
            f.close() 
        
            self.nc = int((indata[0].strip().split())[1])
            self.npars = int((indata[1].strip().split())[1])
            self.delta = int((indata[2].strip().split())[1])
            self.ncr = int((indata[3].strip().split())[1])
            self.maxsetback = int((indata[4].strip().split())[1])
            self.ct = int((indata[5].strip().split())[1])
            self.tc = int((indata[6].strip().split())[2])
            for jj in range(1,self.ncr+1):
                self.n_id[jj-1] = float(indata[7].strip().split()[jj] )
            print("loaded n_id", self.n_id)
            for j in range(1,self.ncr+1):
                self.pCR[j-1] = float(indata[8].strip().split()[j] )
            print("loaded pCR", self.pCR)
            for j in range(1,self.ncr+1):
                self.J[j-1] = float(indata[9].strip().split()[j] )
            print("loaded J", self.J)
            self.c_star = float((indata[10].strip().split())[1])
            self.c      = float((indata[11].strip().split())[1])
            self.setbackcount  = int((indata[12].strip().split())[1])
            self.boundary      = str((indata[13].strip().split())[1])
            #initial = str((data[11].strip().split())[0])

            # chain.likelihood and pars are required to calculate omegas for chain removal
            # and rhats respectively 

            for i in range(self.nc):
                f = open(dirName +  '/'+'pars' + '_' + str(i) +  '.dat','r')
                line = f.readlines()
                indata = []
                for l in line:
                    if l[0] != "#":
                        indata.append(l)
                f.close() 
                # recovering likelihoods
                for kk in range(0,len(indata)):
                    #print(indata[kk].strip().split()[0])
                    self.chains[i].likelihood.append(float(indata[kk].strip().split()[0]))
                
                # recovering parameters
                rst = np.loadtxt(dirName +  '/'+'pars' + '_' + str(i) +  '.dat')
                #print(rst)
                for jj in range( np.shape(rst)[0]):
                    #print(jj)
                    if jj == 0:
                        #D.chains[i].pars = np.copy(rst[jj,1:])

                        self.chains[i].pars = np.copy(rst[jj,1:])
                    else:
                        self.chains[i].pars = np.vstack((self.chains[i].pars , rst[jj,1:])) 
                        
                #print('recovered pars:' ,np.shape(self.chains[i].pars))
 
            

    
    
    def update_R_hats(self, dirName,  R_hat):
        for i in range(self.nc):
            self.chains[i].varget()
        
        self.Rget()
        
        if (np.isnan(self.R[0]) or np.isnan(self.R[0])): self.R[0] = -999
        if (np.isnan(self.R[1]) or np.isnan(self.R[1])): self.R[1] = -999                
        R_hat.append(self.R)
        
        #print(D.R)
        # writing the current Rhats to RunnR file    
        f = open(dirName +  '/'+'RunnR.dat','w')
        f.write('%i \n' % (self.ct))
        for i in range(np.size(self.R)):
            f.write('%10g \n' % self.R[i])
        f.close()   
        
    
    def print_postpars(self, dirName):
        f = open(dirName +  '/'+'postpars.dat','w')
        for i in range(self.nc):
            dim = np.shape(self.chains[i].postpars)
            for j in range(dim[0]):       
                f.write('%1g ' % self.chains[i].postlike[j])
                for k in range(dim[1]):
                    f.write('%g ' % self.chains[i].postpars[j,k])
                f.write('\n')
        f.close()   

    def print_postpars_II(self, dirName):
        f = open(dirName +  '/'+'postparsII.dat','w')
        for i in range(self.nc):
            dim = np.shape(self.chains[i].postparsII)
            for j in range(dim[0]):       
                f.write('%1g ' % self.chains[i].postlike[j])
                for k in range(dim[1]):
                    f.write('%g ' % self.chains[i].postparsII[j,k])
                f.write('\n')
        f.close()   

    def print_postpars_II_lmft(self, dirName):
        f = open(dirName +  '/'+'postparsII.dat','w')
        for i in range(self.nc):
            dim = np.shape(self.chains[i].postparsII)
            for j in range(dim[0]):       
                for k in range(dim[1]):
                    f.write('%g ' % self.chains[i].postparsII[j,k])
                f.write('\n')
        f.close()   

 
                
    def print_pars(self, dirName):
        f = open(dirName +  '/'+'pars.dat','w')
        for i in range(self.nc):
            dim = np.shape(self.chains[i].pars)
            for j in range(dim[0]-1):       
                f.write('%1g ' % self.chains[i].likelihood[j])
                for k in range(dim[1]):
                    f.write('%g ' % self.chains[i].pars[j,k])
                f.write('\n')
        f.close()   
        
    def update_pars(self, dirName):
        for i in range(self.nc):
            f = open(dirName +  '/'+'pars' + '_' + str(i) +  '.dat','w')

            dim = np.shape(self.chains[i].pars)
            for j in range(dim[0]-1):       
                f.write('%1g ' % self.chains[i].likelihood[j])
                for k in range(dim[1]):
                    f.write('%g ' % self.chains[i].pars[j,k])
                f.write('\n')
            f.close()   
                
    def print_currents(self, dirName):
        f = open(dirName +  '/'+'chain_pars_like'  +'.dat','w')
        for i in range(self.nc):
            f.write('%g ' % self.chains[i].Lold)
            for j in range(self.npars):
                f.write('%g ' % self.chains[i].current[j])
        f.write('\n')
        f.close()
    
    def print_Rhats(self, dirName, Rhats):
        
        f = open(dirName +  '/' + 'Rhats'  + '.dat','w')        
        Rdim = np.shape(Rhats)        
        for j in range(Rdim[0]):       
            f.write('%1g ' % 0)
            for k in range(Rdim[1]):
                f.write('%g ' % Rhats[j][k])
            f.write('\n')        

    def write_cmod(self, dirName,  spefolder, seclocs_array, time_inn, modelsim, like):
        
        dirName_cmod = dirName + '/'+'cmod'
        dirName_cmod_type = dirName_cmod + spefolder
        
        if self.ct == 10: 
            if not os.path.exists(dirName_cmod):
                os.mkdir(dirName_cmod)
                print("Directory " , dirName_cmod ,  " Created ")
                
            if not  os.path.exists(dirName_cmod_type):
                os.mkdir(dirName_cmod_type)
                print("Directory " , dirName_cmod_type ,  " Created ")



        for loc in range(0,len(seclocs_array)-1):

            f = open(dirName_cmod_type + '/' + 'cmod'+ '_' + str(loc+1) + '_' + str(self.ct)  +'.dat','w')
            tdum = time_inn
            syze = np.size(modelsim[loc])
            f.write('%i \n' % like)
            for k in range(syze):
                f.write('%g %g \n' % (tdum[k],modelsim[loc][k]))
            f.close()


    def update_current(self, dirName):
        
        f = open(dirName +  '/'+'current.dat','w')
        for i in range(self.nc):     
            f.write('%g ' % self.chains[i].Lold)
            for j in range(self.npars):
                f.write('%g ' % self.chains[i].current[j])
            f.write('\n')
        f.close()
    
    def print_DREAM_settings(self, dirName):
                                  #dirName, initial,  dt, dx, x_max, start_times, seclocs_array,weightsl
        f = open(dirName +  '/'+'dream_settings.dat','w')
        f.write('number of chains: '); f.write('%g ' % self.nc); f.write('\n')
        f.write('number of params: '); f.write('%g ' % self.npars); f.write('\n')
        f.write('delta: '); f.write('%g ' % self.delta); f.write('\n')
        f.write('nCR: '); f.write('%g ' % self.ncr); f.write('\n')
        f.write('maxsetback: '); f.write('%g ' %  self.maxsetback); f.write('\n')
        f.write('total iterations: '); f.write('%g ' % self.tc); f.write('\n')
        f.write('pCR: ');        
        for j in range(0,len(self.pCR)):
            f.write('%g ' % self.pCR[j])
        f.write('\n')
        f.write('c star: '); f.write('%g ' % self.c_star); f.write('\n')
        f.write('Acceptance rate during Posterior-sampling: '); f.write('%g ' % round(self.nacc /((self.ct)* self.nc), 3) ); f.write('\n')

        f.write('c: '); f.write('%g ' % self.c); f.write('\n')
        f.write('burn in iterations: '); f.write('%g ' % self.nb); f.write('\n')
        f.write('iterations after burn-in: '); f.write('%g ' % self.ct); f.write('\n')
        f.write('setbackcount: '); f.write('%g ' % self.setbackcount); f.write('\n')
        f.write('boundary crossing: '); f.write(self.boundary); f.write('\n')
        f.write('min: ');        
        for j in range(len(self.chains[0].pmin)):
            f.write('%g ' % self.chains[0].pmin[j])
        f.write('\n')
        f.write('mean: ');        
        for j in range(len(self.chains[0].mean)):
            f.write('%g ' % self.chains[0].mean[j])
        f.write('\n')
        f.write('max: ');        
        for j in range(len(self.chains[0].pmin)):
            f.write('%g ' % self.chains[0].pmax[j])
        f.write('\n')
        f.close()




    def print_recovery(self, dirName):
        
        f = open(dirName +  '/'+'recovery.dat','w')

        f.write('nchains: '); f.write('%g ' % self.nc); f.write('\n')
        f.write('npars: '); f.write('%g ' % self.npars); f.write('\n')
        f.write('delta: '); f.write('%g ' % self.delta); f.write('\n')
        f.write('nCR: ');   f.write('%g ' % self.ncr); f.write('\n')
        f.write('maxsetback: '); f.write('%g ' %  self.maxsetback); f.write('\n')
        f.write('ct: '); f.write('%g ' % self.ct); f.write('\n')
        f.write('total counts: '); f.write('%g ' % self.tc); f.write('\n')
        f.write('n_id: '); 
        for j in range(0,len(self.n_id)):
            f.write( str(int(self.n_id[j]) )  )
            f.write(' ')   
        f.write('\n')
        f.write('pCR: ');        
        for j in range(0,len(self.pCR)):
            f.write( str(self.pCR[j] )  )
            f.write(' ')
        f.write('\n')
        f.write('J: ');        
        for j in range(0,len(self.J)):
            f.write( str(self.J[j] )  )
            f.write(' ')
        f.write('\n')
        f.write('c_star: ');f.write('%g ' % self.c_star); f.write('\n')
        f.write('c: ');f.write('%g ' % self.c); f.write('\n')
        f.write('setbackcount: ');f.write('%g ' % self.setbackcount); f.write('\n')
        f.write('boundary: ');f.write(self.boundary); f.write('\n')
        #f.write('initial: ');f.write(initial); f.write('\n')       
        f.close()
        
        
class chain:
    def __init__(self, npars, prior = False):
        self.npars = npars
        self.current = np.zeros(npars)
        self.proposal = np.zeros(npars)
        self.jump = np.zeros(npars)
        self.Lold = 0.
        self.Lnew = 0.
        self.pwidth = 0.
        self.prior = prior
        self.idd = 0
        self.likelihood = []
        self.postlike = []
        self.accept = 0
        self.old_proposal_min =  np.zeros(npars)
        self.old_proposal_max = np.zeros(npars)
        self.postparsII   = []
        self.real_current = []
                
    def par_set(self,log,uniform,mean,width,pmin,pmax,rstart = False):
        self.log = log         #logical
        self.uniform = uniform # logical
        self.mean = mean       # mean or centre of distribution (log for log parameters)
        self.width = width     #stabdard deviation or width (;og for log)
        self.rs = rstart       #logica - if you want random generte start point
        self.pwidth = []    
        self.pmin=pmin
        self.pmax=pmax
        for i in range(np.size(self.width)):        
            self.pwidth.append(self.width[i]/25.)
        
        if self.rs == True: # randomly generates start point set to false by default 
            for i in range(self.npars):
                if uniform[i] == True:
                    self.current[i] = self.mean[i] - self.width[i] + np.random.rand()  * 2. *self.width[i]
                else:
                    self.current[i] = np.random.normal(loc = self.mean[i], scale = self.width[i])
        else:
            self.current[i] = np.copy(self.mean[i])
        
        for i in range(self.npars):
            if self.current[i] < self.pmin[i]:
                self.current[i] = 2. * self.pmin[i] - self.current[i]
            if self.current[i] > self.pmax[i]:
                self.current[i] = 2. * self.pmax[i] - self.current[i]                

        self.pars         = np.copy(self.current)
        self.postpars     = np.copy(self.current)

    def prior(self):
        for i in range(self.npars):
            if self.uniform == True:
                if self.proposal[i] < (self.mean[i] - self.width[i]) or self.proposal[i] > (self.mean[i] + self.width[i]):
                    self.Lnew += -1000
            else:
                self.Lnew += ((self.proposal[i]-self.mean[i])**2.)/(2.*self.width[i]**2.)
                
    def tprob(self):
        self.accept = 0 
        if self.Lnew > self.Lold:
            print('accepted Lnew > Lold')
            print('Lnew :', round(float(self.Lnew), 3), 'Lold:', round(float(self.Lold), 3))

            self.current = np.copy(self.proposal)
            self.Lold = np.copy (self.Lnew)
            self.accept += 1
            self.postparsII = np.vstack((self.postparsII, self.real_current)) # keep old values

        else:
            unifrom = np.random.rand() # sp.rand()
            if unifrom < np.exp(self.Lnew - self.Lold): # since loglikes are being used!
                print('accepted Lnew < Lold', unifrom, np.exp(self.Lnew - self.Lold), round(float(self.Lnew), 3), round(float(self.Lold), 3))
             
                self.current = np.copy(self.proposal)
                self.Lold    = np.copy(self.Lnew)
                self.accept += 1
                self.postparsII = np.vstack((self.postparsII, self.real_current)) # keep old values

            else: 
                #print('not accepted Lnew' ,round(float(self.Lnew), 3) )
                #print('not accepted jump' , self.current+self.jump/self.current*100)
                self.jump = np.zeros(self.npars) 
                
                
        self.pars = np.vstack((self.pars,self.current)) # keep old values
        self.postpars = np.vstack((self.postpars, self.current)) # keep old values
        #self.postparsII = np.vstack((self.postparsII, self.real_current)) # keep old values

        self.likelihood.append(float(self.Lold))
        self.postlike.append(float(self.Lold))

        #print( round(float(self.Lold), 3), round(self.Lnew, 3), np.exp(self.Lnew - self.Lold) )

        #print(self.likelihood)
        #if self.Lold == self.Lnew: self.L1old[i] = L1[i]
        
        
        
        
    def varget(self):  
        n = int(np.shape(self.pars)[0]/2)
        s2 = np.var(self.pars[n:,:],axis = 0)
        mn = np.average(self.pars[n:,:], axis=0)
        return(mn,s2,n)
'''            
n = int(np.shape(D.chains[i].pars)[0]/2)               
s2 = np.var(D.chains[i].pars[n:,:],axis = 0)
np.average(D.chains[i].pars[n:,:], axis=0)
D.chains[i].pars[n:,:] 



params = { 'k0': {'mean': 1, 'min': 1, 'max' : 2, 'log' : True, 'uniform': True},
           'k1': {'mean': 1, 'min': 1, 'max' : 2, 'log' : True, 'uniform': True},
           'k2': {'mean': 1, 'min': 1, 'max' : 2, 'log' : True, 'uniform': True},
           'ss': {'mean': 1, 'min': 1, 'max' : 2, 'log' : True, 'uniform': True},
           'sy': {'mean': 1, 'min': 1, 'max' : 2, 'log' : True, 'uniform': True}  }


''' 
