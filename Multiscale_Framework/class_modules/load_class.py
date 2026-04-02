#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 12:00:48 2025

@author: bastien.sauty

A file for defining the loading classes used in the multiscale model. Allows easy and versatile definitions for multiple steps loading cycles.

Artery_load : simulating artery as an axisymmetrical cylinder.
"""

import numpy as np

class Artery_load:
    """
    Class for simulation the artery as an axisymmetrical cylinder. Three simulation parameters : time, axial displacement and pressure.
    See the main files for geometry and BC application.
    
    Parameters:
    -----------
    load_phase : list
        list of list defining the different loading phases. each element contains [nb steps, end value for time, end value for uz, end value for pressure]
        
    Attributes:
    -----------
    list_t : array of size (sum(number steps) + 1)
        contain the evolution of time for each simulation step
    list_uz : array of size (sum(number steps) + 1)
        contain the evolution of axial displacement for each simulation step
    list_P : array of size (sum(number steps) + 1)
        contain the evolution of internal pressure for each simulation step
        
    list_dt : array of size (sum(number steps))
        contain each increment of time
    list_duz : array of size (sum(number steps))
        contain each increment of axial displacement
    list_dP : array of size (sum(number steps))
        contain each increment of pressure
    """
    
    def __init__(self, load_phase):
        
        self.list_t = np.array([0.])
        self.list_uz = np.array([0.])
        self.list_P = np.array([0.])
        self.index_phase = []
        for i, phase in enumerate(load_phase):
            n_phase, t_phase, uz_phase, P_phase = phase
            P_phase *= 0.000133322 # from mmHg to MPa
            list_t_phase = np.linspace(self.list_t[-1], t_phase, n_phase+1)
            list_uz_phase = np.linspace(self.list_uz[-1], uz_phase, n_phase+1)
            list_P_phase = np.linspace(self.list_P[-1], P_phase, n_phase+1)
            
            id_start = len(self.list_t)-1
            self.list_t = np.concatenate((self.list_t, list_t_phase[1:]))
            self.list_uz = np.concatenate((self.list_uz, list_uz_phase[1:]))
            self.list_P = np.concatenate((self.list_P, list_P_phase[1:]))
            id_end = len(self.list_t)-1
            
            self.index_phase.append([id_start, id_end])
            
        self.list_dt = self.list_t[1:] - self.list_t[0:-1]
        self.list_duz = self.list_uz[1:] - self.list_uz[0:-1]
        self.list_dP = self.list_P[1:] - self.list_P[0:-1]
        
    