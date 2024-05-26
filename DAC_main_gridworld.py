#https://papers.nips.cc/paper/2021/file/c2626d850c80ea07e7511bbae4c76f4b-Paper.pdf 

import numpy as np
import matplotlib.pyplot as plt
import os
import pdb
import random
import time
from DAC_utils_gridworld import *

#Cliff env
# height=4
# width=12
height=3
width=4
num_states=(height*width)**2   #Each agent counts its own state from top row (i=0) to bottom row (i=height-1)
num_actions=4     #up,down,left,right
num_features=num_states
num_agents=2
r_cliff=-100  #i=height-1, j not equal to 0 or width-1.
r_goal1=-0.5    #The reward of the agent when it arrives at the destination while the other does not
r_goal2=0     #The reward of both agents when both agents are at the destination
r_rest=-1     #The reward of the agent when it is not in cliff nor destination

def ijij2s(i1,j1,i2,j2,height,width): #return the agents' shared state if the two agents are at (i1,j1),(i2,j2) respectively
    s1=i1*width+j1
    s2=i2*width+j2
    return s1*width*height+s2

def next_ij(i,j,a):
    if i==height-1:
        if j==width-1:   
            return (i,j)  #stay at destination
        elif j!=0:
            return (i,0)  #from cliff to starting point
    
    i_next=i
    j_next=j
    if a==0 and i>0:  #up
        i_next-=1
    elif a==1 and i<height-1:   #down
        i_next+=1
    elif a==2 and j>0:  #left
        j_next-=1
    elif a==3 and j<width-1:   #right
        j_next+=1
    return (i_next,j_next)

transP=np.zeros((num_states,num_actions,num_actions,num_states))
reward=r_rest*np.ones((num_states,num_actions,num_actions,num_states,2))
s=0
for i1 in range(height):
    for j1 in range(width):
        for i2 in range(height):
            for j2 in range(width):
                for a1 in range(num_actions):
                    for a2 in range(num_actions):
                        i1_next,j1_next=next_ij(i1,j1,a1)
                        i2_next,j2_next=next_ij(i2,j2,a2)
                        s_next=ijij2s(i1_next,j1_next,i2_next,j2_next,height,width)
                        transP[s,a1,a2,s_next]=1.0
                        if i1==width and j1>0 and j1<width-1:
                            reward[s,a1,a2,s_next,0]=r_cliff
                        elif i1_next==height-1 and j1_next==width-1:
                            reward[s,a1,a2,s_next,0]=r_goal1
                        
                        if i2==width and j2>0 and j2<width-1:
                            reward[s,a1,a2,s_next,1]=r_cliff
                        elif i2_next==height-1 and j2_next==width-1:
                            reward[s,a1,a2,s_next,1]=r_goal1
                s+=1
                
s_destination=num_states-1
s_pre=ijij2s(height-2,width-1,height-2,width-1,height,width)  #the state when both agents are right above the destination
reward[s_destination,:,:,s_destination,:]=r_goal2
reward[s_pre,:,:,s_destination,:]=r_goal2

init_xi=np.zeros(num_states)
s_init=ijij2s(height-1,0,height-1,0,height,width)
init_xi[s_init]=1.0
#initialize at the location (i=height-1,j=0) for both agents

W=get_W_diagmain(d=num_agents,p_central=0.7)
DAC_dict1=init\
(seed_init=0,state_space=range(num_states),action_spaces=[list(range(num_actions))]*num_agents,init_xi=init_xi,\
 transP=transP,reward=reward,gamma=0.95,W=W,V_features=None,R_features=None)

#hyperparameters
expr_num=1
Tr=5
Tc=50
Tc_prime=10
beta=0.5
Nc=10
noise_std=np.repeat(0.1,num_agents)

folder='Results/Cliff'

set_seed(5)
omega0=[np.random.normal(size=(num_states,num_actions)) for m in range(num_agents)]
h0=[np.zeros((num_states,num_actions)) for m in range(num_agents)]

alg='AC'
T=500
T=1500
N_set=[100,500,2000]
alpha_set=[1,5,20]   
legend_set=['Alg.1,N='+str(n) for n in N_set]
hyps=[{'alg':alg,'T':T,'Tc':Tc,'Tc_prime':Tc_prime,'Tr':Tr,'N':N_set[i],'Nc':Nc,\
       'alpha':alpha_set[i],'beta':beta,'noise_std':noise_std,'s0_DAC':s_init,'s0_DTD':s_init,\
       'omega0':omega0,'theta0':None,'seed_sim_DAC':None,'seed_sim_DTD':None,'is_print':True,'is_save':False,\
       'getRvg_err_every_numiter':15,'getDTD_err_every_numiter':None,\
       'data_folder':folder+'/ourAC_N'+str(N_set[i])+'_alpha'+str(alpha_set[i]),'communication_dx':1,'sample_dx':1,\
       'color':'red','marker':'','legend':legend_set[i]} for i in range(len(N_set))]

alg='NAC'
T=2000
Tz=5
N_set=[100,500,2000]
K_set=[50,100,200]
Nk_set=[[int(N_set[i]/K_set[i])]*K_set[i] for i in range(len(K_set))]
# alpha_set=[0.1,0.5,2]; eta_set=[0.1,0.5,2]
# alpha_set=[0.01,0.05,0.2]; eta_set=[0.01,0.05,0.2]
alpha_set=[0.002,0.01,0.04];eta_set=[0.002,0.01,0.04] 
# alpha_set=[0.004,0.02,0.08];eta_set=[0.004,0.02,0.08] 
legend_set=['Alg.3,N='+str(n) for n in N_set]

hyps+=[{'alg':alg,'T':T,'Tc':Tc,'Tc_prime':Tc_prime,'Tr':Tr,'Nk':Nk_set[i],'Nc':Nc,'Tz':Tz,'K':K_set[i],\
    'alpha':alpha_set[i],'beta':beta,'eta':eta_set[i],'noise_std':noise_std,'s0_DAC':s_init,'s0_DTD':s_init,'omega0':omega0,'h0':h0,\
    'theta0':None,'seed_sim_DAC':None,'seed_sim_DTD':None,'is_print':True,'is_save':False,\
    'getRvg_err_every_numiter':50,'getDTD_err_every_numiter':None,\
    'data_folder':folder+'/ourNAC_N'+str(N_set[i])+'_alpha'+str(alpha_set[i])+'_eta'+str(eta_set[i])+'_K'+str(K_set[i]),\
    'communication_dx':1,'sample_dx':1,'color':'red','marker':'','legend':legend_set[i]} for i in range(len(N_set))]

alg='DAC-RP'
T=50000

betav=lambda t:5*((t+1)**(-0.8))          
beta_theta=lambda t:2*((t+1)**(-0.9))     

set_seed(6)
v0=np.random.normal(size=(num_agents,num_states))
num_R_features=num_states*num_states*(num_actions**num_agents)
lambda0=np.random.normal(size=(num_agents,num_R_features))

hyps+=[{'alg':alg,'T':T,'v0':v0,'lambda0':lambda0,'omega0':omega0,'theta0':None,'s0_DTD':s_init,'s0_DAC':s_init,'seed':None,\
        'beta_v':betav,'beta_theta':beta_theta,'is_print':True,'is_save':False,'data_folder':folder+'/DACRP1',\
        'communication_dx':1,'sample_dx':500,'color':'red','marker':'','legend':'DAC-RP1',\
        'is_exact_Ravg':False,'getRvg_err_every_numiter':100,'getDTD_err_every_numiter':None}]
   

alg='DAC-RP-batch-InexactRavg'
N_DTD=Nc
N_DAC=hyps[0]['N']
T=2000
betav=hyps[0]['beta']; beta_theta=hyps[0]['alpha']  
#betav=0.1; beta_theta=0.1
# betav=0.5; beta_theta=1

# if not is_try:
hyps+=[{'alg':alg,'T':T,'N_DTD':N_DTD,'N_DAC':N_DAC,'v0':v0,'lambda0':lambda0,'omega0':omega0,'theta0':None,'s0_DTD':s_init,'s0_DAC':s_init,'seed':None,\
            'beta_v':betav,'beta_theta':beta_theta,'is_print':True,'is_save':False,'data_folder':folder+'/DACRP100',\
            'communication_dx':1,'sample_dx':1,'color':'purple','marker':'','legend':'DAC-RP100',\
            'is_exact_Ravg':False,'getRvg_err_every_numiter':20,'getDTD_err_every_numiter':None}]

#Colors for our AC
hyps[0]['color']='red'
hyps[1]['color']='green'
hyps[2]['color']='black'

#Colors for DAC-RPs
hyps[6]['color']='cyan'
hyps[7]['color']='blue'
# hyps[8]['color']='blueviolet'
# hyps[9]['color']='darkgoldenrod'

#Colors for our NAC
hyps[3]['color']='red'
hyps[4]['color']='green'
hyps[5]['color']='black'


results0, Jmax=runs(expr_num=expr_num,num_agents=num_agents,DAC_dict=DAC_dict1,hyps=hyps,folder=folder,is_getJmax=True)
Jmax2=r_rest*(1-DAC_dict1['gamma']**width)
print('Jmax='+str(Jmax)+', Jmax2'+str(Jmax2))

fontsize=20
lgdsize=16 

bottom_loc=0.2
left_loc=0.22
fig_width=6
fig_height=6

indexes=[0,1,2,6,7]
plots(results=[results0[k] for k in indexes],hyps=[hyps[k] for k in indexes],Jmax=Jmax,color_Jmax='',marker_Jmax='',\
      percentile=95,fontsize=fontsize,lgdsize=lgdsize,bottom_loc=bottom_loc,left_loc=left_loc,J_legend_loc=7,dJ_legend_loc=1,\
      err_legend_loc=1,fig_width=fig_width,fig_height=fig_height,plot_folder=folder+'/Figures_AC',is_plotJgap=True)
indexes=[3,4,5]
plots(results=[results0[k] for k in indexes],hyps=[hyps[k] for k in indexes],Jmax=Jmax,color_Jmax='',marker_Jmax='',\
      percentile=95,fontsize=fontsize,lgdsize=lgdsize,bottom_loc=bottom_loc,left_loc=left_loc,J_legend_loc=1,dJ_legend_loc=1,\
      err_legend_loc=1,fig_width=fig_width,fig_height=fig_height,plot_folder=folder+'/Figures_NAC',is_plotJgap=True)

for hyp_index in range(len(hyps)):
    hyp=hyps[hyp_index].copy()
    if 'relative_Ravg_err' in results0[hyp_index].keys():
        if 'Ideal' not in hyp['legend']:
            Ravg_err=results0[hyp_index]['relative_Ravg_err'].copy()
            Ravg_err=Ravg_err[:,Ravg_err[0]>=0]
            
            if hyp["alg"]=='AC':
                print("The average relative R approximation error for our AC with N="+str(hyp["N"])+": "+str(Ravg_err.mean()))
            elif hyp["alg"]=='NAC':
                N=int(np.sum(hyp['Nk']))
                print("The average relative R approximation error for our NAC with N="+str(N)+": "+str(Ravg_err.mean()))
            else:
                print("The average relative R approximation error for "+str(hyp['legend'])+": "+str(Ravg_err.mean()))
            
    if 'relative_DTD_err' in results0[hyp_index].keys():
        DTD_err=results0[hyp_index]['relative_DTD_err'].copy()
        DTD_err=DTD_err[:,DTD_err[0]>=0]
        
        if hyp["alg"]=='AC':
            print("The average relative TD error for our AC with N="+str(hyp["N"])+": "+str(DTD_err.mean()))
        elif hyp["alg"]=='NAC':
            N=int(np.sum(hyp['Nk']))
            print("The average relative TD error for our NAC with N="+str(N)+": "+str(DTD_err.mean()))
        else:
            print("The average relative TD error for "+str(hyp['legend'])+": "+str(DTD_err.mean()))
    print()        

ii=6
kk=results0[ii]['Jw'].shape[1]-1
print("The mean J(omega) for DAC-RP with batchsize 1 goes from "+str(results0[ii]['Jw'][:,0].mean())+" to "+str(results0[ii]['Jw'][:,kk].mean()))



    



    