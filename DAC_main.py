# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
import pdb
import random
import time
from DAC_utils import *

num_states=5
num_actions=2
num_features=5
num_agents=6

network=['ring','full'][1]

if network is 'ring':
    W=get_W_3diags(d=num_agents,p_central=0.4)
else:
    W=get_W_diagmain(d=num_agents,p_central=0.4)

expr_num=10
T=500
Tr=5
Tc=50
Tc_prime=10
beta=0.5
Nc=10
noise_std=np.repeat(0.1,num_agents)




folder='Results/simulation_'+network

set_seed(5)
omega0=[np.random.normal(size=(num_states,num_actions)) for m in range(num_agents)]
h0=[np.zeros((num_states,num_actions)) for m in range(num_agents)]

alg='AC'
N_set=[100,500,2000]
alpha_set=[10,50,200]
legend_set=['Alg.1,N='+str(n) for n in N_set]
hyps=[{'alg':alg,'T':T,'Tc':Tc,'Tc_prime':Tc_prime,'Tr':Tr,'N':N_set[i],'Nc':Nc,\
       'alpha':alpha_set[i],'beta':beta,'noise_std':noise_std,'s0_DAC':0,'s0_DTD':0,\
       'omega0':omega0,'theta0':None,'seed_sim_DAC':None,'seed_sim_DTD':None,'is_print':False,'is_save':False,\
       'getRvg_err_every_numiter':1,'getDTD_err_every_numiter':1,\
       'data_folder':folder+'/ourAC_N'+str(N_set[i])+'_alpha'+str(alpha_set[i]),'communication_dx':1,'sample_dx':1,\
       'color':'red','marker':'','legend':legend_set[i]} for i in range(len(N_set))]

alg='ACH'
N_set=[100,500,2000]
alpha_set=[10,50,200]
alpha_h=0.04
legend_set=['Alg.2,N='+str(n) for n in N_set]
hyps+=[{'alg':alg,'T':T,'Tc':Tc,'Tc_prime':Tc_prime,'Tr':Tr,'N':N_set[i],'Nc':Nc,\
       'alpha':alpha_set[i],'alpha_h':alpha_h,'beta':beta,'noise_std':noise_std,'s0_DAC':0,'s0_DTD':0,\
       'omega0':omega0,'theta0':None,'seed_sim_DAC':None,'seed_sim_DTD':None,'is_print':False,'is_save':False,\
       'getRvg_err_every_numiter':1,'getHvg_err_every_numiter':1,'getDTD_err_every_numiter':1,\
       'data_folder':folder+'/ourACH_N'+str(N_set[i])+'_alpha'+str(alpha_set[i]),'communication_dx':1,'sample_dx':1,\
       'color':'red','marker':'','legend':legend_set[i]} for i in range(len(N_set))]
'''
alg='NAC'
T=2000
Tz=5
N_set=[100,500,2000]
K_set=[50,100,200]
Nk_set=[[int(N_set[i]/K_set[i])]*K_set[i] for i in range(len(K_set))]
alpha_set=[1.0/10,1/2,2]
eta_set=[0.04,0.2,0.8]
legend_set=['Alg.3,N='+str(n) for n in N_set]

hyps+=[{'alg':alg,'T':T,'Tc':Tc,'Tc_prime':Tc_prime,'Tr':Tr,'Nk':Nk_set[i],'Nc':Nc,'Tz':Tz,'K':K_set[i],\
    'alpha':alpha_set[i],'beta':beta,'eta':eta_set[i],'noise_std':noise_std,'s0_DAC':0,'s0_DTD':0,'omega0':omega0,'h0':h0,\
    'theta0':None,'seed_sim_DAC':None,'seed_sim_DTD':None,'is_print':False,'is_save':False,\
    'getRvg_err_every_numiter':1,'getDTD_err_every_numiter':1,\
    'data_folder':folder+'/ourNAC_N'+str(N_set[i])+'_alpha'+str(alpha_set[i])+'_eta'+str(eta_set[i])+'_K'+str(K_set[i]),\
    'communication_dx':1,'sample_dx':1,'color':'red','marker':'','legend':legend_set[i]} for i in range(len(N_set))]

alg='DAC-RP'
T=2500*250

betav=lambda t:5*((t+1)**(-0.8))
beta_theta=lambda t:2*((t+1)**(-0.9))

set_seed(6)
v0=np.random.normal(size=(num_agents,num_states))
num_R_features=num_states*num_states*(num_actions**num_agents)
lambda0=np.random.normal(size=(num_agents,num_R_features))

hyps+=[{'alg':alg,'T':T,'v0':v0,'lambda0':lambda0,'omega0':omega0,'theta0':None,'s0_DTD':0,'s0_DAC':0,'seed':None,\
        'beta_v':betav,'beta_theta':beta_theta,'is_print':False,'is_save':False,'data_folder':folder+'/DACRP1',\
        'communication_dx':1,'sample_dx':500,'color':'red','marker':'','legend':'DAC-RP1',\
        'is_exact_Ravg':False,'getRvg_err_every_numiter':10000,'getDTD_err_every_numiter':10000}]


alg='DAC-RP-batch-InexactRavg'
N_DTD=10
N_DAC=100
T=15000
betav=0.5
beta_theta=10

# if not is_try:
hyps+=[{'alg':alg,'T':T,'N_DTD':N_DTD,'N_DAC':N_DAC,'v0':v0,'lambda0':lambda0,'omega0':omega0,'theta0':None,'s0_DTD':0,'s0_DAC':0,'seed':None,\
            'beta_v':betav,'beta_theta':beta_theta,'is_print':False,'is_save':False,'data_folder':folder+'/DACRP100',\
            'communication_dx':1,'sample_dx':1,'color':'purple','marker':'','legend':'DAC-RP100',\
            'is_exact_Ravg':False,'getRvg_err_every_numiter':100,'getDTD_err_every_numiter':100}]
'''
    
#Colors for our AC
hyps[0]['color']='red'
hyps[1]['color']='green'
hyps[2]['color']='black'

#Colors for our ACH
hyps[3]['color']='cyan'
hyps[4]['color']='blue'
hyps[5]['color']='yellow'
'''
#Colors for DAC-RPs
hyps[6]['color']='cyan'
hyps[7]['color']='blue'

#Colors for our NAC
hyps[3]['color']='red'
hyps[4]['color']='green'
hyps[5]['color']='black'
'''

DAC_dict1=init\
(seed_init=0,state_space=range(num_states),action_spaces=None,init_xi=None,\
 transP=None,reward=None,gamma=0.95,W=W,V_features=None,R_features=None)

results0, Jmax=runs(expr_num=expr_num,num_agents=num_agents,DAC_dict=DAC_dict1,hyps=hyps,folder=folder)

fontsize=20
lgdsize=16 

bottom_loc=0.2
left_loc=0.22
fig_width=6
fig_height=6

indexes=[0,1,2,3,4,5]
plots(results=[results0[k] for k in indexes],hyps=[hyps[k] for k in indexes],Jmax=Jmax,color_Jmax='',marker_Jmax='',\
      percentile=95,fontsize=fontsize,lgdsize=lgdsize,bottom_loc=bottom_loc,left_loc=left_loc,J_legend_loc=7,dJ_legend_loc=1,\
      err_legend_loc=1,fig_width=fig_width,fig_height=fig_height,plot_folder=folder+'/Figures_AC',is_plotJgap=True)
'''
indexes=[3,4,5]
plots(results=[results0[k] for k in indexes],hyps=[hyps[k] for k in indexes],Jmax=Jmax,color_Jmax='',marker_Jmax='',\
      percentile=95,fontsize=fontsize,lgdsize=lgdsize,bottom_loc=bottom_loc,left_loc=left_loc,J_legend_loc=7,dJ_legend_loc=1,\
      err_legend_loc=1,fig_width=fig_width,fig_height=fig_height,plot_folder=folder+'/Figures_ACH',is_plotJgap=True)

indexes=[3,4,5]
plots(results=[results0[k] for k in indexes],hyps=[hyps[k] for k in indexes],Jmax=Jmax,color_Jmax='',marker_Jmax='',\
      percentile=95,fontsize=fontsize,lgdsize=lgdsize,bottom_loc=bottom_loc,left_loc=left_loc,J_legend_loc=1,dJ_legend_loc=1,\
      err_legend_loc=1,fig_width=fig_width,fig_height=fig_height,plot_folder=folder+'/Figures_NAC',is_plotJgap=True)
'''
for hyp_index in range(len(hyps)):
    hyp=hyps[hyp_index].copy()
    if 'relative_Ravg_err' in results0[hyp_index].keys():
        if 'Ideal' not in hyp['legend']:
            Ravg_err=results0[hyp_index]['relative_Ravg_err'].copy()
            Ravg_err=Ravg_err[:,Ravg_err[0]>=0]
            
            if hyp["alg"]=='AC':
                print("The average relative R approximation error for our AC with N="+str(hyp["N"])+": "+str(Ravg_err.mean()))
            elif hyp["alg"]=='ACH':
                print("The average relative R approximation error for our ACH with N="+str(hyp["N"])+": "+str(Ravg_err.mean()))
            else:
                print("The average relative R approximation error for "+str(hyp['legend'])+": "+str(Ravg_err.mean()))

    if 'relative_DTD_err' in results0[hyp_index].keys():
        DTD_err=results0[hyp_index]['relative_DTD_err'].copy()
        DTD_err=DTD_err[:,DTD_err[0]>=0]
        
        if hyp["alg"]=='AC':
            print("The average relative TD error for our AC with N="+str(hyp["N"])+": "+str(DTD_err.mean()))
        elif hyp["alg"]=='ACH':
            print("The average relative TD error for our ACH with N="+str(hyp["N"])+": "+str(DTD_err.mean()))
        else:
            print("The average relative TD error for "+str(hyp['legend'])+": "+str(DTD_err.mean()))
    print()        





    