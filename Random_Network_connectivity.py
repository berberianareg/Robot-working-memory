#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Nareg Berberian
@institution: University of Ottawa, Ottawa, Canada
@affiliation: Laboratory for Computational Neurodynamics and Cognition
@project title: Embodied Working Memory During Ongoing Input Streams

"""
#%% Import libraries and modules ==============================================

import numpy as np
import random

#%% Network class =============================================================

class Network:
    def __init__(self):
        # network parameters
        self.c = 0.2                                                            # probability of synaptic contact (default: 0.2)
        self.N_E = 400                                                          # nb of excitatory units (default: 400)
        self.N_I = 100                                                          # nb of inhibitory units (default: 100)
        self.N = self.N_E + self.N_I                                            # total nb of units
        
        # assigning strength of units in a given subpopulation
        self.J_multiplier = 1                                                   # multiplicative factor (default: 1)
        self.J_IE = 0.65 * self.J_multiplier                                    # excitatory to inhibitory synaptic strength (default: 0.65)
        self.J_EI = -1 * self.J_multiplier                                      # inhibitory to excitatory synaptic strength (default: -1)   
        self.J_II = -1 * self.J_multiplier                                      # inhibitory to inhibitory synaptic strength (default: -1)
        self.J_EE = 0.65 * self.J_multiplier                                    # excitatory to excitatory synaptic strength (default: 0.65)
        self.connectivity()

#%% Connectivity function =====================================================
        
    # random assignment of E and I units in the network
    def connectivity(self):
        self.rand_N = np.random.permutation(self.N)                             # randomize all cells
        self.rand_N_E = self.rand_N[:self.N_E]                                  # randomly choose excitatory cells
        self.rand_N_I = self.rand_N[self.N_E:]                                  # randomly choose inhibitory cells
        self.nb_excit = round(self.c * self.N_E)                                # random nb of connections from the excitatory population
        self.nb_inhib = round(self.c * self.N_I)                                # random nb of connections from the inhibitory population
        
        self.InitialEEW = np.zeros([self.N,self.N])                             # preallocate connectivity matrix 
        self.InitialIEW = np.zeros([self.N,self.N])                             # preallocate connectivity matrix 
        self.InitialIIW = np.zeros([self.N,self.N])                             # preallocate connectivity matrix 
        self.InitialEIW = np.zeros([self.N,self.N])                             # preallocate connectivity matrix 
        
        for i_excit in list(range(0,len(self.rand_N_E))):
            
            self.InitialEEW[random.sample([x for i,x in enumerate(self.rand_N_E) if i!=i_excit],self.nb_excit),self.rand_N_E[i_excit]] = self.J_EE
            self.InitialEIW[random.sample([x for i,x in enumerate(self.rand_N_I) if i!=i_excit],self.nb_inhib),self.rand_N_E[i_excit]] = self.J_EI
            
        for i_inhib in list(range(0,len(self.rand_N_I))):
            
            self.InitialIIW[random.sample([x for i,x in enumerate(self.rand_N_I) if i!=i_inhib],self.nb_inhib),self.rand_N_I[i_inhib]] = self.J_II
            self.InitialIEW[random.sample([x for i,x in enumerate(self.rand_N_E) if i!=i_inhib],self.nb_excit),self.rand_N_I[i_inhib]] = self.J_IE
            
        self.InitialEEW = self.InitialEEW.transpose()                           # transpose weight matrix such that sending units (axis=1) and receiving units (axis=0)
        self.InitialIEW = self.InitialIEW.transpose()                           # transpose weight matrix such that sending units (axis=1) and receiving units (axis=0)
        self.InitialIIW = self.InitialIIW.transpose()                           # transpose weight matrix such that sending units (axis=1) and receiving units (axis=0)
        self.InitialEIW = self.InitialEIW.transpose()                           # transpose weight matrix such that sending units (axis=1) and receiving units (axis=0)
        
        self.InitialW = self.InitialEEW + self.InitialEIW + self.InitialIIW + self.InitialIEW

#%% Extract network structure =================================================
            
        # find number of IN-degree and OUT-degree connections
        [self.k_in_deg,self.k_out_deg]  = np.count_nonzero(self.InitialW,axis=1),np.count_nonzero(self.InitialW,axis=0)
         
        # find the structure of the network
        self.StructW = np.zeros([self.N,self.N])                                # preallocate the structure of all network connections
        self.StructW[self.InitialW != 0] = 1                                    # find the structure of all network connections
        
        # find the structure of EE connections
        self.StructEEW = np.zeros([self.N,self.N])                              # preallocate the structure of EE connections
        self.StructEEW[self.InitialEEW != 0] = 1                                # find the structure of EE connections
        
        # find the structure of II connections
        self.StructIIW = np.zeros([self.N,self.N])                              # preallocate the structure of II connections 
        self.StructIIW[self.InitialIIW != 0] = 1                                # find the structure of II connections
        
        # find the structure of IE connections
        self.StructIEW = np.zeros([self.N,self.N])                              # preallocate the structure of IE connections 
        self.StructIEW[self.InitialIEW != 0] = 1                                # find the structure of IE connections
        
        # find the structure of EI connections
        self.StructEIW = np.zeros([self.N,self.N])                              # preallocate the structure of EI connections 
        self.StructEIW[self.InitialEIW != 0] = 1                                # find the structure of EI connections
        
        return [self.N, self.StructW, self.StructEEW, self.StructIIW, self.StructIEW,
                self.StructEIW, self.InitialW, self.k_in_deg, self.k_out_deg,
                self.rand_N_E, self.rand_N_I, self.J_multiplier]

