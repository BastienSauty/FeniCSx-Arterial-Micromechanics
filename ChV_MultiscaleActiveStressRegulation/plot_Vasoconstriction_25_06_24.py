#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 15:55:46 2025

@author: bastien.sauty

Run the sensitivity analysis on the Lin2002 combined tensile-shear test.

Use the main file 'main_Lin2002_0407.py' to run one simulation. 
Run a series of simulation with different parameters, only if not already done.
Then plot the results
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


import matplotlib as mpl
from matplotlib.patches import ConnectionPatch

mpl.rcParams['text.usetex'] = True


#-----------------------------------------------------------------------------#
# Custom modules
#-----------------------------------------------------------------------------#

from Multiscale_Framework.class_modules.load_class import Artery_load

from ChV_MultiscaleActiveStressRegulation.main_Vasoconstriction_25_11_24 import (
    run_simulation,
    load_JSON,
)
#-----------------------------------------------------------------------------#
# Load or Run
#-----------------------------------------------------------------------------#

folder_name = 'Vasoconstriction/'
name = 'Vcn_11_27'
namefile = name +'_scal.pkl'


simu_card_name = 'json_cards/simu_card_Vcn.json'
media_card_name ='json_cards/media_card_Vcn.json'
adventitia_card_name ='json_cards/adventitia_card_Vcn.json'
# Load material card        
simu_card = load_JSON(simu_card_name)
adventitia_card = load_JSON(adventitia_card_name)
media_card = load_JSON(media_card_name)

step_load = Artery_load(simu_card['load_phase'])

try:
    file = open('outputs/'+folder_name+'/'+namefile, 'rb')

    result = pickle.load(file) # load and store pck file
    file.close()
except:
    print(f'Running simulation {namefile}')
    
    result, mech = run_simulation(name, folder_name, simu_card, adventitia_card, media_card)
    
    
#%%
# Phase indices

id_noVcn = step_load.index_phase[1]
id_Vcn = step_load.index_phase[-1]

id_noVcn = np.arange(id_noVcn[0],id_noVcn[1]+1)
id_Vcn = np.arange(id_Vcn[0],id_Vcn[1]+1)

#%%
#-----------------------------------------------------------------------------#
# Plots
#-----------------------------------------------------------------------------#

# Load Control
time_list = step_load.list_t #result.outputs['time'][:]
time_list /= time_list[-1]
press_list = 7500.62*step_load.list_P #result.outputs['press'][:]
uz_list = 1000*step_load.list_uz

fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(8,6), sharex='all')
ax1.plot(uz_list)
ax1.set_ylabel('Axial strain')


ax2.plot(press_list)
ax2.set_ylabel('Pressure [mmHg]')


ax3.plot(time_list)
ax3.set_ylabel('Normalized time')

ax3.set_xlabel('Step number')

for step in step_load.index_phase:
    con = ConnectionPatch(
        (step[0], max(uz_list)),
        (step[0], 0),
        "data",
        "data",
        axesA=ax1,
        axesB=ax3,
        color="black",
        ls="dotted",
    )
    fig.add_artist(con)
    
con = ConnectionPatch(
    (step[1], max(uz_list)),
    (step[1], 0),
    "data",
    "data",
    axesA=ax1,
    axesB=ax3,
    color="black",
    ls="dotted",
)
fig.add_artist(con)

fig.tight_layout()
plt.savefig('images_output/Vasoconstriction/successive_load.pdf')
plt.show()

#%%
# Pressure Radius
plt.figure(figsize=(4,3))  

ri_d = result.outputs['ri_d'][:]
re_d = result.outputs['re_d'][:]
area =  result.outputs['area'][:]
plt.plot(ri_d[id_noVcn], press_list[id_noVcn], color='tab:blue', label=r'$R_i$, no Vcn')
plt.plot(re_d[id_noVcn], press_list[id_noVcn], color='tab:orange', label=r'$R_e$, no Vcn')
plt.plot(ri_d[id_Vcn], press_list[id_Vcn], color='tab:blue', linestyle='dashed', label=r'$R_i$, Vcn')
plt.plot(re_d[id_Vcn], press_list[id_Vcn], color='tab:orange', linestyle='dashed',  label=r'$R_e$, Vcn')
plt.grid()
plt.ylabel('Pressure [mmHg]')
plt.xlabel('Radius [mm]')
plt.legend()
plt.tight_layout()
plt.savefig('images_output/Vasoconstriction/pressure_radius.pdf')
plt.show()

# Radius - time : 
plt.figure(figsize=(4,3))  
plt.plot(time_list/time_list[-1], ri_d, label=r'$R_i$')
plt.plot(time_list/time_list[-1], re_d, label=r'$R_e$')
plt.grid()
plt.xlabel('Normalized time')
plt.ylabel('Radius [mm]')
plt.legend()
plt.tight_layout()
plt.savefig('images_output/Vasoconstriction/time_radius.pdf')
plt.show()
    

#-----------------------------------------------------------------------------#
# Local value alongside position
#-----------------------------------------------------------------------------#

#%%
def find_index_press(press_list, value):
    list_indices = []
    for i, p in enumerate(press_list):
        if np.isclose(p, value):
            list_indices.append(i)
    return(list_indices)


press_100 = find_index_press(press_list, 100)
index_relaxed = press_100[0]
index_basal = press_100[-1]

#%%
r_pos = result.dict_outputs['S_yy']['points']

S_yy_noVcn = result.outputs['S_yy'][index_relaxed, :]
S_yy_Vcn = result.outputs['S_yy'][index_basal, :]


plt.figure(figsize=(4,3))
plt.plot(r_pos, S_yy_noVcn, label='Before Vasoconstriction')
plt.plot(r_pos, S_yy_Vcn, label='After Vasoconstriction')
plt.grid()
plt.xlabel('Radial position [mm]')
plt.ylabel('Circumferential stress [MPa]')
plt.legend()
plt.tight_layout()
plt.savefig('images_output/Vasoconstriction/circ_stress.pdf')
plt.show()

#%%
sig_er_noVcn = result.outputs['s_yy_cell'][index_relaxed, :]
sig_er_Vcn = result.outputs['s_yy_cell'][index_basal, :]


plt.figure(figsize=(4,3))
plt.plot(r_pos, sig_er_noVcn, label='Before Vasoconstriction')
plt.plot(r_pos, sig_er_Vcn, label='After Vasoconstriction')
plt.grid()
plt.xlabel('Radial position [mm]')
plt.ylabel('Axial cellular stress [MPa]')
plt.legend()
plt.tight_layout()
plt.savefig('images_output/Vasoconstriction/cell_stress.pdf')
plt.show()

la_noVcn = result.outputs['lambda_cell_in'][index_relaxed, :]
la_Vcn = result.outputs['lambda_cell_in'][index_basal, :]


plt.figure(figsize=(4,3))
plt.plot(r_pos, la_noVcn, label='Before Vasoconstriction')
plt.plot(r_pos, la_Vcn, label='After Vasoconstriction')
plt.grid()
plt.xlabel('Radial position [mm]')
plt.ylabel('Inelastic stretch in the cells')
plt.legend()
plt.tight_layout()
plt.savefig('images_output/Vasoconstriction/cell_inel_stretch.pdf')
plt.show()


