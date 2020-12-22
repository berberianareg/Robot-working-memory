#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Nareg Berberian
@institution: University of Ottawa, Ottawa, Canada
@affiliation: Laboratory for Computational Neurodynamics and Cognition
@project title: Embodied Working Memory During Ongoing Input Streams

"""
#%% General statements ========================================================

# this script requires specific libraries and modules (see Import section below).
# this script has been tested using Python 3.7 Spyder 3.3.4, mac OS Big Surf, Version 11.1.
# before running this script, initialize "Random_Network_connectivity.py"

#%% Steps to follow for running the experiment ================================

# 1: go to Preferences > Run -> enable "Interact with the Python console after execution" > Apply > OK.
# 2: run file.
# 3: let the simulation run for some time, then press the "enter" key at any moment.
# 4: press "left" or "right" key, then let the simulation run for some time.
# 5: press "enter" key at any moment, then return to step 3.
 
#%% Import libraries and modules ==============================================

import numpy as np                                                              # for manipulating multi-dimensional arrays
import matplotlib.pyplot as plt                                                 # for visualizing data
from Random_Network_connectivity import Network                                 # import network connectivity
from pynput import keyboard                                                     # for controlling and monitoring the keyboard 
from concurrent import futures                                                  # for running events over multiple threads
import threading                                                                # for managing events between multiple threads
import time                                                                     # for tracking simulation run time
event = threading.Event()

#%% Network parameters ========================================================

Net = Network()                                                                 # assign "Network" class to Net
[N, StructW, StructEEW, StructIIW, StructIEW, StructEIW, InitialW, k_in_deg,
 k_out_deg, rand_N_E, rand_N_I, J_multiplier] = Net.connectivity()              # call "connectivity" function
W = InitialW.copy()                                                             # make a copy of initialized network connectivity

#%% Temporal parameters =======================================================

dt = 1                                                                          # time-step increments (ms)
binWidth = int(40/dt)                                                           # specify binWidth within which post spike counts will be computed (default: 40 ms)
Tbatch = [0]                                                                    # preallocate timing of reinforcements
TSdetect = []                                                                   # preallocate task switch detections

#%% aEIF parameters ===========================================================

C_m     = 281e-3                                                                # membrane capacitance (pF) (default: 281e-3)
E_L     = -70.6                                                                 # leak reversal potential (mV) (default: -70.6)
V_th    = -50.4                                                                 # spike threshold (mV) (default: -50.4)
V_reset = -70.6                                                                 # reset value after spike (mV) (default: -70.6)
g_L     = 30e-3                                                                 # leak conductance (nS) (default: 30e-3)
V_peak  = 20                                                                    # value to draw a spike to, when cell spikes (mV) (default: 20)
DeltaT  = 2                                                                     # slope factor (default: 2)
tau_w   = 144                                                                   # adaptation time constant (ms) (default: 144)
a       = 4e-15                                                                 # subthreshold adaptation (nS) (default: 4e-15)
b       = 0.0805                                                                # spike-triggered adaptation (nA) (default: 0.0805)
tau_s   = 5                                                                     # decay time constant of postsynaptic current (ms) (default: 5)

#%% STP parameters ============================================================

dep_U   = 0.80                                                                  # baseline release probability (default: 0.80)
tau_d   = 900                                                                   # depression time constant (default: 900)
tau_f   = 100                                                                   # facilitation time constant (default: 100)

#%% STDP parameters ===========================================================

maxNbTrials = 5                                                                 # max number of trials (default: an odd integer)
lambdaplus  = 5e-5*dt                                                           # learning rate for LTP - scales the magnitude of individual weight changes during LTP (default: 5e-5)
lambdaminus = 25e-5*dt                                                          # learning rate for LTD - scales the magnitude of individual weight changes during LTD (default: 25e-5)
mu          = 1                                                                 # set boundary conditions on changes in W (default: 1)
alpha       = 2                                                                 # denotes possible asymmetry between scales of LTP and LTD (default: 2)
tauminus    = 50/dt                                                             # temporal window for LTD (default: 50)
tauplus     = 20/dt                                                             # temporal window for LTP (default: 20)

#%% STDP function =============================================================

def STDP(tdiff, lambdaminus, lambdaplus, tauplus, tauminus, mu, alpha, W):      # arguments for STDP function
    N      = len(tdiff)                                                         # the length along axis 0 of matrix "tdiff" will correspond to the N nb of units in the network  
    deltaW = np.zeros([N,N])                                                    # preallocate changes in connection weights
    LTP    = tuple([tdiff > 0])                                                 # find difference (tpost-tpre) in spike times greater than or equal to 0
    LTD    = tuple([tdiff <= 0])                                                # find difference (tpost-tpre) in spike times less than 0    
    if [tdiff > 0]:                                                             # if the difference (tpost-tpre) in spike times is greater than or equal to 0
        deltaW[LTP] = lambdaplus * (1 - W[LTP]) ** mu *\
        np.exp(-(tdiff[LTP]) / tauplus)                                         # potentiate connections between units that exhibit those time differences
    
    if [tdiff <= 0]:                                                            # if the difference (tpost-tpre) in spike times is less than 0 
        deltaW[LTD] = -lambdaminus * (alpha * W[LTD]) ** mu *\
        np.exp((tdiff[LTD]) / tauminus)                                         # depress connections between units that exhibit those time differences
        
    return deltaW                                                               # return changes in synaptic weights 

#%% Intrinsic Poisson noise ===================================================

def intrinsicNoise(fr, N, dt):                                                  # arguments for intrinsic Poisson spike generation
    intrinsicSpikes = np.random.rand(N) < fr * (dt/1000)                        # generate Poisson spikes to each unit
    return intrinsicSpikes.astype(int)                                          # return intrinsic spikes

#%% External input ============================================================

def ExternalInput(N, kInp, m, sd, Rp, Rn, C):                                   # input arguments 
    
    Rp      = Rp - C                                                            # preferred amplitude peak of double Gaussian function
    Rn      = Rn - C                                                            # non-preferred amplitude peak of double Gaussian function
    R       = np.zeros(N)                                                       # preallocate Gaussian at preferred amplitude peak
    R_plus  = np.zeros(N)                                                       # preallocate Gaussian wrap-around at preferred amplitude peak  
    R_minus = np.zeros(N)                                                       # preallocate Gaussian wrap-around at non-preferred amplitude peak 
    for iN in range(N):        
        R[iN] = Rp * np.exp(-(iN - m) ** 2 / (2 * sd ** 2)) +\
        Rn * np.exp(-(iN + (N/2) - m) ** 2 / (2 * sd ** 2))                     # double gaussian function             
        
        R_plus[iN] = Rp * np.exp(-(iN - m + N) ** 2 / (2 * sd ** 2)) +\
        Rn * np.exp(-(iN + (N/2) - m + N) ** 2 / (2 * sd ** 2))                 # wrap-around double gaussian from the preferred amplitude end 
        
        R_minus[iN] = Rp * np.exp(-(iN - m - N) ** 2 / (2 * sd ** 2)) +\
        Rn * np.exp(-(iN + (N/2) - m - N) ** 2 / (2 * sd ** 2))                 # wrap-around double gaussian from the non-preferred amplitude end 

    return C + (R + R_plus + R_minus)                                           # return double gaussian function with the wrap-around effect 

#%% Input parameters ==========================================================

kInp = N                                                                        # k number of units to receive input (default: N)
m    = N/4                                                                      # unit m receives highest amplitude peak (default: unit N/4)
sd   = 7 * (N/100)                                                              # standard deviation of the tuning curve (default: 7 * (N/100))
Rp   = 2.5                                                                      # amplitude at the highest peak (nA) (default: 2.5)
Rn   = 1                                                                        # amplitude at the lowest peak (nA) (default: 1)
C    = 0.5                                                                      # baseline amplitude (nA) (default: 0.5)    
PreSpikesFR = np.full(N, 10)                                                    # frequency of intrinsic noise during Evoked and Spontaneous activity (Hz) (default: 10)
IextOFF = np.zeros(N)                                                           # stimulus OFF
IextONright = ExternalInput(N, kInp, m, sd, Rp, Rn, C)                          # stimulus ON right
IextONleft = ExternalInput(N, kInp, m, sd, Rn, Rp, C)                           # stimulus ON left

#%% STP_STDP_aEIF variables ===================================================

# STP variables
U      = np.zeros(N) + dep_U                                                    # preallocate baseline release probability
u      = np.zeros(N) + dep_U                                                    # preallocate release probability
x      = np.ones(N)                                                             # preallocate neurotransmitter availability

# STDP variables 
Wnonzero = []                                                                   # preallocate number of nonzero elements
Wtrial = [ [] for x in range(maxNbTrials) ]                                     # preallocate average weights across time
Wtrials      = np.zeros([N,N,maxNbTrials])                                      # preallocate weights stored at every trial
deltaW       = np.zeros([N,N])                                                  # preallocate changes in weights
deltaWtrials = np.zeros([N,N,maxNbTrials])                                      # preallocate changes in weights stored at every trial

# aEIF variables
psthIdx = [ [] for x in range(maxNbTrials) ]                                    # preallocate psthIdx
psth    = [ [] for x in range(maxNbTrials) ]                                    # preallocate psth 
psthLmotor = [ [] for x in range(maxNbTrials) ]                                 # preallocate psthLmotor 
psthRmotor = [ [] for x in range(maxNbTrials) ]                                 # preallocate psthRmotor 
preSTs     = [ [] for x in range(N) ]                                           # preallocate pre spike timing
postSTs    = [ [] for x in range(N) ]                                           # preallocate post spike timing
G          = np.dot(InitialW, u * x)                                            # preallocate total synaptic efficacy
Isyn       = np.zeros(N)                                                        # preallocate synaptic current
V          = np.zeros(N) + E_L                                                  # initialize membrane potential 
i_last     = np.zeros(N)                                                        # preallocate last post spike time
j_last     = np.zeros(N)                                                        # preallocate last pre spike time
w          = np.zeros(N)                                                        # preallocate adaptation variable
LastSpikes = np.zeros([N,binWidth])                                             # initialize LastSpikes 

#%% Key press function ========================================================

def on_press(key):
    global Iext                                                                 # define list of global variables
    if key == keyboard.Key.left:                                                # if the left key is pressed
        Iext = IextONleft                                                       # turn ON left stimulus
        print('{0} pressed'.format(key))                                        # print which key was pressed
        TSdetect.append(-1)                                                     # store -1 if left key is pressed for task switch detection
        return False                                                            
    elif key == keyboard.Key.right:                                             # if the right key is pressed
        Iext = IextONright                                                      # turn ON right stimulus
        print('{0} pressed'.format(key))                                        # print which key was pressed
        TSdetect.append(1)                                                      # store +1 if right key is pressed for task switch detection
        return False    
    
#%% Key capture thread ========================================================

def key_capture_thread():    
    event.set()                                                                 # set the internal flag to true    
    input()                                                                     # accommodate an input key press ("enter" key)
    event.clear()                                                               # reset the internal flag to false

    Tbatch.append(s-1)                                                          # store the timing of reinforcement
    deltaWtrials[:,:,itrial] = deltaW                                           # store changes in weights at every trial
    Wtrials[:,:,itrial] = W                                                     # store weights at every trial
    
    print(['trial # {}'.format(itrial)])                                        # print trial number and baseline release Pr

#%% Driving thread ============================================================

def drive_thread():
    global s, u, x, Isyn, v, w, V, W, LastSpikes, deltaW                        # define list of global variables
    event.wait()                                                                # wait until the flag is true
    
    s = 1                                                                       # start simulation at time s = 1
    binCount = 0                                                                # initialize binCount
    LastSpikes = np.zeros([N,binWidth])                                         # initialize LastSpikes 

    while event.is_set():                                                       # while the internal flag is true
        
        if itrial == 0:                                                         # if initial batch
            Wnonzero.append(np.count_nonzero(W))                                # store number of nonzero elements present in weight matrix

        PreSpikes = intrinsicNoise(PreSpikesFR, N, dt)                          # intrinsic Poisson noise
        preSpikers = [x for x,j in enumerate(PreSpikes) if j == 1]              # find pre spikers greater than or equal to 1 
        if preSpikers:                                                          # if there are pre spikers
            j_last[preSpikers] = s-1                                            # store last spike of pre spikers                
            [preSTs[j].append(np.cumsum(Tbatch)[-1] + s-1) for x,j\
             in enumerate(preSpikers)]                                          # store pre spike times

#%% STP mechanism =============================================================
        
        du = (U - u)/tau_f + U * (1 - u) * (PreSpikes/dt)                       # release probability (u)                  
        u = u + du * dt                                                         # increment u by timestep dt via Euler method 
        
        dx = (1 - x)/tau_d - u * x * (PreSpikes/dt)                             # neurotransmitter availability (x)                       
        x = x + dx * dt                                                         # increment x by timestep dt via Euler method    
        
        G = np.dot(W, u * x)                                                    # total synaptic efficacy
        
        dIsyn = -Isyn/tau_s + G * (PreSpikes/dt)                                # total synaptic current                         
        Isyn = Isyn + dIsyn * dt                                                # increment Isyn by timestep dt via Euler method

#%% aEIF model ================================================================               
        
        dV = (-g_L * (V - E_L) + g_L * DeltaT *
              np.exp((V - V_th) / DeltaT) - w + Isyn + Iext) / C_m              # adaptive exponential function
        V = V + dV * dt                                                         # increment V by timestep dt via Euler method
        
        dw = (a * (V - E_L) - w) / tau_w                                        # adaptation function
        w = w + dw * dt                                                         # increment w by timestep dt via Euler method

        if s-1 - (binCount * binWidth) == binWidth:                             # if s-1 increments binWidth times
            psth[itrial].append(np.sum(LastSpikes))                             # compute spike count of last bin
            psthLmotor[itrial].append(np.sum(LastSpikes[:int((N/2))]))          # compute spike count of last bin for left motor
            psthRmotor[itrial].append(np.sum(LastSpikes[int((N/2)):]))          # compute spike count of last bin for right motor
            psthIdx[itrial].append(len(np.hstack(psth)))                        # store bin index
            LastSpikes = np.zeros([N,binWidth])                                 # initialize LastSpikes 
            binCount += 1                                                       # increment binCount

        postSpikers = [y for y,i in enumerate(V) if i > V_peak]                 # find post spikers greater than V_peak
        if postSpikers:                                                         # if there are post spikers
            i_last[postSpikers] = s-1                                           # store last spike of post spikers
            V[postSpikers] = V_reset                                            # reset potential of post spikers
            w[postSpikers] = w[postSpikers] + b                                 # add constant to adaptation variable        
            LastSpikes[postSpikers,s-1 - (binCount * binWidth)] = 1             # assign spike to postSpikers in LastSpikes
            [postSTs[i].append(np.cumsum(Tbatch)[-1] + s-1) for y,i\
             in enumerate(postSpikers)]                                         # store post spike times
        
#%% STDP mechanism ============================================================                    
        
        diffSTs = i_last.reshape(N, 1) - j_last                                 # compute differences in spike timing                 
        
        deltaW = STDP(diffSTs, lambdaminus, lambdaplus, tauplus, tauminus, mu,
                      alpha, W)                                                 # compute changes in synaptic weights via STDP learning   
                       
        Wtrial[itrial].append(np.mean(W))                                       # store mean weights    
                
        W = W + deltaW                                                          # update weights  
                
        print(round(s*dt,1))         
        s += 1                                                                  # increment timestep by 1

#%% Recruit threads ===========================================================

def model(): 
    
    global Iext, nbTrials, itrial                                               # define list of global variables
    itrial = 0                                                                  # initialise itrial
    nbEvokTrials = 0                                                            # initialise nb of Evoked trials
    nbSponTrials = 0                                                            # initialise nb of Spontaneous trials
    nbTrials     = 0                                                            # initialise total nb of trials
    
    while itrial < maxNbTrials:
    
        if itrial % 2 == 1:                                                     # if itrial is an odd number
            print('left.key<--or-->right.key')
            with keyboard.Listener(on_press=on_press) as listener:                       
                listener.join()                                                 # wait for the keypress listener to listen            
            print('IextON')                                                     
            nbEvokTrials = int(itrial+1 - nbSponTrials)                         # store nb of evoked trials
                
        elif itrial % 2 == 0:                                                   # if itrial is an even number
            Iext = IextOFF                                                      # turn stimulus OFF
            print('IextOFF')
            nbSponTrials = int(itrial+1 - nbEvokTrials)                         # store nb of spontaneous trials
            TSdetect.append(0)                                                  # store 0 if left nor right keys are pressed for task switch detection
            
        nbTrials = nbEvokTrials + nbSponTrials                                  # store total nb of trials

        with futures.ThreadPoolExecutor() as executer:
            t1 = executer.submit(key_capture_thread)                            # recruit key_capture_thread
            t2 = executer.submit(drive_thread)                                  # recruit drive_thread
            t1.result()                                                         # start activity of key_capture_thread
            t2.result()                                                         # start activity of drive_thread
            
        itrial += 1                                                             # increment itrial        

#%% Run model =================================================================

start = time.time()                                                             # start of simulation time 
model()                                                                         # run the model
end = time.time()                                                               # end of simulation time 
print("[Finished in %f seconds]" % (end - start))                               # display simulation runtime (s)

del u, w, x, s
#%% Detect batches (i.e. trials) where task has switched ======================

signchange = ((np.roll(np.insert(np.diff(TSdetect),0,0), 1) -\
               TSdetect) !=0).astype(int)                                       # detect sign change for task switch detection
task_switch_batch_idx = np.where(signchange == 0)[0]                            # specify batch indices where the task is switched

#%% Spike raster and PSTH =====================================================

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
fig, ax1 = plt.subplots(figsize=(15,10))  
plt.eventplot(np.flipud(postSTs), color ='k', linestyles ='solid')                                                 
ax1.set_xlim(np.cumsum(Tbatch)[0],np.cumsum(Tbatch)[-1] + 1)     
ax1.set_ylim(0,N)  
ax1.set_xlabel('Time (s)', fontsize = 25)                                                                                                
ax1.set_ylabel('Unit #', fontsize = 25) 
ax1.set_xticklabels(ax1.get_xticks()/1000*dt, fontsize=25)
ax1.set_yticklabels(np.flipud(ax1.get_yticks().astype(int)),fontsize=25) 
[ax1.spines[axis].set_linewidth(4) for axis in ['top','bottom','left','right']]
ax2 = ax1.twinx()                                                              
plt.plot(np.hstack(Wtrial), color = 'saddlebrown', linewidth = 8, alpha = 0.6)
ax2.set_ylabel('Weight', fontsize = 25, color = 'saddlebrown')                                        
ax2.tick_params('y', colors = 'saddlebrown') 
plt.yticks(fontsize=25) 
plt.tight_layout()                                                                                              
plt.show()  

#%% Batch duration ============================================================

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
fig, ax = plt.subplots(figsize=(15,10)) 
ONbatch = plt.bar(np.arange(1,nbTrials,2),np.array(Tbatch[2::2])/1000*dt,
             color = 'powderblue')                                                   
OFFbatch = plt.bar(np.arange(0,nbTrials,2),np.array(Tbatch[1::2])/1000*dt,
              color = 'k')                                                  
if len(task_switch_batch_idx) > 0:
    for ibatch in range(len(task_switch_batch_idx)):
        TSbatch = plt.axvline(x = task_switch_batch_idx[ibatch]-0.5, color = 'orange',linewidth = 5.0,linestyle = '--',alpha = 0.6)                 
    ax.legend([ONbatch, OFFbatch, TSbatch], ['Stimulus ON','Stimulus OFF','task switch'], fontsize=25, framealpha=0, loc='upper center', bbox_to_anchor=(0.5, 1.125), ncol=3)                                                                
else:
    ax.legend([ONbatch, OFFbatch],['Stimulus ON','Stimulus OFF'], fontsize=25,framealpha=0, loc='upper left',bbox_to_anchor=(0.15, 1.125), ncol=2)                           
ax.set_xlabel('Batch #', fontsize = 25, fontweight='bold')                                         
ax.set_ylabel('Batch duration (s)', color = 'k', fontsize = 25)                 
plt.xlim(-1,nbTrials)                                                           
plt.xticks(range(0,nbTrials,1),fontsize = 25)                                   
plt.yticks(fontsize = 25)                                                       
ax.set_xticklabels(ax.get_xticks()+1)
ax.tick_params(axis='y', colors ='k', direction='in',length=6, which='major', right=True, color='k', width = 4)                    
ax.tick_params(axis='x', colors = 'k', direction='in', length=6, color='white', width = 4) 
[ax.spines[axis].set_linewidth(4) for axis in ['top','bottom','left','right']]
plt.tight_layout()                                                                                              
plt.show()             
                                                        
del ONbatch, OFFbatch

#%% External input current ====================================================

#Iexts = IextONleft,IextONright,IextOFF
#Iexts_titles = ['IextON left','IextON right','IextOFF']
#
#cols = np.arange(3)+1
#fig, ax = plt.subplots(1, len(cols), figsize=(15,5))         
#for k in range(len(cols)):
#    if k == len(cols)-1:
#        ax[cols[k]-1].plot(Iexts[k],color='k',linewidth=10)
#    else:
#        ax[cols[k]-1].plot(Iexts[k],color='powderblue',linewidth=10)
#    ax[cols[k]-1].tick_params('y', colors = 'k', direction='in', length=4, labelsize=15)                     
#    ax[cols[k]-1].tick_params('x', colors = 'k', direction='in', length=4, labelsize=15)                     
#    ax[cols[k]-1].tick_params(axis='y', colors ='k', direction='in',length=4, which='major', right=True, color='k', width = 4)                    
#    ax[cols[k]-1].tick_params(axis='x', colors = 'k', direction='in', length=4, color='white', width = 4) 
#    [ax[cols[k]-1].spines[axis].set_linewidth(4) for axis in ['top','bottom','left','right']]
#    ax[cols[k]-1].set_title(Iexts_titles[k],fontweight='bold',fontsize=20)
#    ax[cols[k]-1].set_ylim(-0.2,max(IextONright)+0.25)
#    ax[cols[k]-1].axhline(y=0,linewidth=4,linestyle='--',color='orange',alpha=0.75)
#ax[1].set_xlabel('Unit #',fontweight='bold',fontsize=15)
#ax[0].set_ylabel('Amplitude (nA)',fontweight='bold',fontsize=15)
#plt.tight_layout()
#plt.show()
#
#del Iexts, Iexts_titles, cols

#%% Network connectivity during the end of every batch ========================

#plt.rcParams["font.weight"] = "bold"
#plt.rcParams["axes.labelweight"] = "bold"
#cmap = plt.cm.BrBG
#for ibatch in range(len(Tbatch)-1):
#    fig, ax = plt.subplots(figsize=(10,10))                                           
#    plt.imshow(Wtrials[:,:,ibatch],cmap=cmap)
#    plt.title("$\it{t}$ $=$ $%.1f$ sec " % (np.cumsum(Tbatch)[ibatch + 1]/1000),
#              fontsize = 25, fontweight='bold')
#    plt.xlabel('Sending unit #', fontsize = 25)
#    plt.ylabel('Receiving unit #', fontsize = 25)
#    plt.xticks(range(0,N,int(N/5)), fontsize = 25)
#    plt.yticks(range(0,N,int(N/5)), fontsize = 25)
#    ax.tick_params('x', colors = 'k', direction='in', length=6, top=True, width=4)
#    ax.tick_params('y', colors = 'k', direction='in', length=6, right=True, width=4)               
#    cbar = plt.colorbar()
#    cbar.ax.set_title('Weight',fontsize = 25,fontweight='bold')
#    cbar.ax.tick_params(labelsize=25)
#    [ax.spines[axis].set_linewidth(4) for axis in ['top','bottom','left','right']]
#    plt.tight_layout()                                                                                              
#    plt.show()
    
#%% Average of incoming connection weights during the end of every recall =====
    
#plt.rcParams["font.weight"] = "bold"
#plt.rcParams["axes.labelweight"] = "bold"
#fig, ax = plt.subplots(figsize=(15,10))                                           
#for ibatch in range(len(Tbatch[0::2])):
#    plt.plot(np.mean(Wtrials[:,:,0::2][:,:,ibatch],axis=1),
#             label="$\it{t}$ = %.1f sec" % (np.cumsum(Tbatch)[1::2][ibatch]/1000),
#             color = [np.linspace(1,0.5,len(Tbatch[0::2]))[ibatch],
#                      np.linspace(0.8,0.3,len(Tbatch[0::2]))[ibatch],
#                      np.linspace(0.4,0,len(Tbatch[0::2]))[ibatch]],
#                      linewidth=4)   
#plt.legend(fontsize = 14,framealpha = 0,loc='upper left')
#plt.xlim(0, N)
#plt.xticks(fontsize = 30)
#plt.yticks(fontsize = 30)
#ax.tick_params('x', colors = 'k', direction='in', length=6, top=True, width=4)
#ax.tick_params('y', colors = 'k', direction='in', length=6, right=True, width=4)               
#plt.xlabel('Receiving unit #', fontsize = 30)
#plt.ylabel('Weight (a.u.)', fontsize = 30)
#[ax.spines[axis].set_linewidth(4) for axis in ['top','bottom','left','right']]
#plt.tight_layout()
#plt.show()
                                                                                                                 
#%% Spike count evolution for left and right motors ===========================

#if any(np.array(Tbatch[1:]) < binWidth):
#    raise ValueError('Attention! The duration of one or more batches are shorter than the minimum binWidth required to appropriately produce this figure.')
#else:
#    plt.rcParams['font.weight'] = 'bold'
#    plt.rcParams['axes.labelweight'] = 'bold'
#    fig, ax = plt.subplots(figsize=(15,10))
#    barwidth = 1
#    ONbatchL = plt.bar(np.hstack(psthIdx[1::2]),np.hstack(psthLmotor[1::2]), barwidth, color='powderblue')
#    ONbatchR = plt.bar(np.hstack(psthIdx[1::2]), -1 * np.hstack(psthRmotor[1::2]), barwidth, color='powderblue')
#    OFFbatchL = plt.bar(np.hstack(psthIdx[0::2]), np.hstack(psthLmotor[0::2]), barwidth, color='k')
#    OFFbatchR = plt.bar(np.hstack(psthIdx[0::2]), -1 * np.hstack(psthRmotor[0::2]), barwidth, color='k')
#    ax.set_xlabel('Time (s)', fontsize = 30, fontweight = 'bold') 
#    ax.set_ylabel('Rotating speed (mm/s)', fontsize = 30, fontweight = 'bold')                                         
#    ax.set_xlim(1, len(np.hstack(psthIdx)))
#    ax.set_ylim(-1*max(np.hstack(psth))+275, max(np.hstack(psth))-275)
#    ax.set_xticklabels(ax.get_xticks()[:-1]*binWidth/1000*dt)
#    ax.set_yticklabels(abs(ax.get_yticks().astype(int)))
#    ax.axhline(y = 0, color = 'white', linewidth = 4, linestyle = '-')
#    ax.tick_params('x', colors = 'k', direction='in', length=6,  width=4, top=True)                     
#    ax.tick_params('y', colors = 'k', direction='in', length=6,  width=4)                     
#    plt.xticks(fontsize=25)
#    plt.yticks(fontsize=25)
#    [ax.spines[axis].set_linewidth(4) for axis in ['top','bottom','left','right']]
#    if len(task_switch_batch_idx) > 0:
#        for ibatch in range(len(task_switch_batch_idx)):
#            TSbatch = plt.axvline(x = psthIdx[task_switch_batch_idx[ibatch]][0]-0.5,color ='orange',linewidth = 5.0,linestyle='--',alpha=0.6)                 
#        ax.legend([ONbatchL, OFFbatchL, TSbatch],['Stimulus ON','Stimulus OFF','task switch'], fontsize=25, framealpha = 0, loc='upper center',bbox_to_anchor=(0.5, 1.125), ncol=3)                                                                
#    else:
#        ax.legend([ONbatchL, OFFbatchL],['Stimulus ON','Stimulus OFF'],fontsize=25,framealpha = 0,loc='upper left',bbox_to_anchor=(0.15, 1.125), ncol=2)   
#    plt.tight_layout()
#    plt.show()
#    
#    del barwidth,ONbatchL,ONbatchR,OFFbatchL,OFFbatchR

#%% Percentage of connectivity across time ====================================

#plt.rcParams["font.weight"] = "bold"
#plt.rcParams["axes.labelweight"] = "bold"                                        
#fig, ax = plt.subplots(figsize=(15,10))  
#plt.plot(np.array(Wnonzero)/(N*N)*100, linewidth = 8, color = 'saddlebrown')
#plt.xlim(0, len(Wnonzero))
#plt.xlabel('Time (s)', fontsize = 20)                                           
#plt.ylabel('Proportion of connections (%)', fontsize = 20)
#plt.xticks(fontsize = 20)                                                       
#plt.yticks(fontsize = 20)                                                       
#plt.axhline(y = Wnonzero[-1]/(N*N)*100, linestyle ='--', color ='k', linewidth=3, alpha=1)
#ax.set_xticklabels(np.round(ax.get_xticks()/1000*dt,1))
#[ax.spines[axis].set_linewidth(4) for axis in ['top','bottom','left','right']]
#ax.tick_params('x', colors = 'k', direction='in', length=6, top=True, width=4)
#ax.tick_params('y', colors = 'k', direction='in', length=6, right=True, width=4)    
#plt.tight_layout()
#plt.show()

#%% Initial network connectivity matrix =======================================

#plt.rcParams['font.weight'] = 'bold'
#plt.rcParams['axes.labelweight'] = 'bold'
#cmap = plt.cm.BrBG
#fig, ax = plt.subplots(figsize=(9,9))                                           
#plt.imshow(InitialW,cmap=cmap)
#plt.title('Network connectivity',fontsize = 25, fontweight='bold')
#plt.xlabel('Sending unit #', fontsize = 25)
#plt.ylabel('Receiving unit #', fontsize = 25)
#plt.xticks(range(0,N,int(N/5)), fontsize = 25)
#plt.yticks(range(0,N,int(N/5)), fontsize = 25)
#ax.tick_params('x', colors = 'k', direction='in', length=6, top=True, width=4)
#ax.tick_params('y', colors = 'k', direction='in', length=6, right=True, width=4)               
#cbar = plt.colorbar()
#cbar.ax.set_title('Weight',fontsize = 25,fontweight='bold')
#cbar.ax.tick_params(labelsize=25)
#[ax.spines[axis].set_linewidth(4) for axis in ['top','bottom','left','right']]
#plt.tight_layout()                                                                                              
#plt.show()

#%% Percentage of E and I units ===============================================

#plt.rcParams['font.weight'] = 'bold'
#plt.rcParams['axes.labelweight'] = 'bold'
#percEunits = np.array(len(rand_N_E))/N*100
#percIunits = np.array(len(rand_N_I))/N*100
#
#fig, ax = plt.subplots(figsize=(15,10))             
#plt.bar((0,1), [percEunits, percIunits], color = 'saddlebrown')
#plt.xticks((0,1), ('Excitatory', 'Inhibitory'), fontsize = 20)
#ax.tick_params('y', colors = 'k', direction='in', length=6)                     
#ax.tick_params('x', colors = 'k', direction='in', length=6)                     
#ax.tick_params(axis='y', colors ='k', direction='in',length=6, which='major', right=True, color='k', width = 4)                    
#ax.tick_params(axis='x', colors = 'k', direction='in', length=6, color='white', width = 4) 
#plt.yticks(fontsize = 20)    
#[ax.spines[axis].set_linewidth(4) for axis in ['top','bottom','left','right']]                                                 
#plt.xlabel('Unit type', fontsize=20)
#plt.ylabel('Percentage of units (%)', fontsize=20)
#plt.show()

#%% In-degree and Out-degree connectivity =====================================

#plt.rcParams['font.weight'] = 'bold'
#plt.rcParams['axes.labelweight'] = 'bold'
#fig, ax = plt.subplots(figsize=(15,10))                                          
#s1 = plt.scatter(range(N),k_in_deg/N*100, 80, color = 'saddlebrown', marker = 'o')
#s2 = plt.scatter(range(N),k_out_deg/N*100, 80, color = 'saddlebrown', marker = '+')
#plt.legend([s1,s2],['incoming','outgoing'],fontsize=20)
#plt.xlabel('Unit #', fontsize = 20)                                           
#plt.ylabel('Percentage of connections (%)', fontsize = 20)
#plt.xticks(fontsize = 20)                                                       
#plt.yticks(fontsize = 20)   
#ax.set_yticklabels(ax.get_yticks().astype(int))
#plt.xlim(0, N) 
#ax.tick_params('y', colors = 'k', direction='in', length=6)                     
#ax.tick_params('x', colors = 'k', direction='in', length=6, top=True)                     
#ax.tick_params(axis='y', colors ='k', direction='in',length=6, which='major', right=True, color='k', width = 4)                    
#ax.tick_params(axis='x', colors = 'k', direction='in', length=6, color='k', width = 4) 
#[ax.spines[axis].set_linewidth(4) for axis in ['top','bottom','left','right']]
#plt.show()

#%% Spike-timing-dependent plasticity (STDP) profile ==========================

#T = int(600/dt)
#tpre = np.zeros([N,T]) + (T/2)                                                
#tpost = np.tile(list(range(0,T)),(N,1))                                        
#w = np.zeros([N,N]) + 0.5                                                 
#delta_w = []
#for s in range(T):
#    tdiff = tpost[:,s-1].reshape(N,1) - tpre[:,s-1]                             
#    delta_w.append(np.mean(STDP(tdiff, lambdaminus, lambdaplus, tauplus, tauminus, mu, alpha, w)))
#
#plt.rcParams["font.weight"] = "bold"
#plt.rcParams["axes.labelweight"] = "bold"
#fig, ax = plt.subplots(figsize=(10,7))
#plt.plot(range(T),delta_w,'saddlebrown',linewidth=8)
#plt.xlabel('\u0394t (ms)',fontsize=20)
#plt.ylabel('\u0394W (a.u.)',fontsize=20)
#plt.xticks(fontsize=20)
#plt.yticks(fontsize=20)
#plt.xlim(0,T)
#ax.set_xticklabels((ax.get_xticks() - tpre[0][0]).astype(int)*dt)
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#ax.tick_params('x', direction='in', length=4, top=True, width=3)
#ax.tick_params('y', direction='in', length=4, right=True, width=3)  
#[ax.spines[axis].set_linewidth(4) for axis in ['top','bottom','left','right']]
#[ax.spines[axis].set_color("k") for axis in ['top','bottom','left','right']]
#plt.title('STDP function', fontsize =25, fontweight='bold',)
#plt.axhline(y=0, linewidth=4, alpha=0.5, linestyle='--', color='k')
#plt.axvline(x=T/2, linewidth=4, alpha=0.5, linestyle='--', color='k')
#del T, tpre, tpost, tdiff, w, delta_w
#plt.tight_layout()
#plt.show()

#%% Short-term plasticity (STP) profile =======================================

#T = int(300/dt)
#u = U[0]
#x = 1
#u_store = np.zeros(T)
#u_store[0] = u
#x_store = np.zeros(T)
#x_store[0] = x
#Spikes = np.zeros(T)
#Spikes[int(50/dt)::int(50/dt)] = 1/dt
#
#for s in range(1,T):
#    du = (U[0] - u)/tau_f + U[0] * (1 - u) * Spikes[s-1]                                   
#    u += du * dt                                                            
#    
#    dx = (1 - x)/tau_d - u * x * Spikes[s-1]                                                  
#    x += dx * dt
#    
#    u_store[s] = u
#    x_store[s] = x
#
#plt.rcParams["font.weight"] = "bold"
#plt.rcParams["axes.labelweight"] = "bold"
#fig, ax = plt.subplots(figsize=(8,6))
#ax1 = plt.subplot(211)
#plt.yticks([])
#plt.xlim(0, T)
#plt.xticks(fontsize=20)
#plt.yticks(fontsize=20)
#ax1.set_xticklabels(ax1.get_xticks().astype(int)*dt)
#plt.plot(range(T),Spikes.transpose(),'k',linewidth=5)
#plt.title('Presynaptic spikes', fontsize = 25, fontweight='bold')
#ax1.tick_params(axis='x', direction='in', length=0, width = 4) 
#[ax1.spines[axis].set_visible(False) for axis in ['right','left','top']]
#[ax1.spines[axis].set_linewidth(4) for axis in ['top','bottom','left','right']]
#
#ax2 = plt.subplot(212)
#plt.plot(range(T), u_store, 'orange', linewidth=5)
#plt.plot(range(T), x_store, 'blue', linewidth=5, alpha=0.5)
#plt.xlim(0, T)
#plt.xticks(fontsize=20)
#plt.yticks(fontsize=20)
#plt.xlabel('Time (ms)',fontsize=20)
#plt.ylabel('Probability',fontsize=20)
#plt.legend(['u','x'],fontsize=25, framealpha = 0)
#ax2.set_xticklabels(ax2.get_xticks().astype(int)*dt)
#plt.title('STP function',fontsize = 25,fontweight='bold')
#[ax2.spines[axis].set_visible(False) for axis in ['right','top']]
#[ax2.spines[axis].set_color("k") for axis in ['top','bottom','left','right']]
#[ax2.spines[axis].set_linewidth(4) for axis in ['top','bottom','left','right']]
#ax2.tick_params(axis='x', colors = 'k', direction='in', length=0, width = 4) 
#ax2.tick_params(axis='y', direction='in', length=0, which='major', width = 4)
#del Spikes, T, u_store, x_store             
#plt.tight_layout()
#plt.show()



