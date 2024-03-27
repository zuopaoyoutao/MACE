import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pdb
import random
import time
import types
import torch
from torch import nn
from torch.optim import Adam

#Default values:
num_states=5
num_actions=2
num_agents=6

def set_hyps(a,a_default): #设置a的参数，如果a为空则返还的默认值，否则返回a
    if(a is None):
        return a_default
    else:
        return a

def set_seed(seed=1): #随机数种子函数，以保证每一次实验复现时产生的随即环境相同
    if seed is not None:
        np.random.seed(seed)    
        random.seed(seed)
    
def get_W_diagmain(d=None,p_central=0.9):
    #Generate dXd communication matrix W with diagonals=p_central and other entries being p_off=(1-p_central)/(d-1)
    if d==1:
        return np.array([1])
    p_off=(1-p_central)/(d-1)
    W=np.array([[p_off]*d]*d)
    np.fill_diagonal(W,p_central)
    return W

def get_W_3diags(d=None,p_central=0.8):  
    #Generate dXd communication matrix W with W[i,i]=p_central and W[i,i+1]=W[i,i-1]=p_off=(1-p_central)/2, 
    # and other entries being zero.
    if d==1:
        return np.array([1])
    W=np.zeros(shape=(d,d))
    p_off=(1-p_central)/2
    if d==2:
        W=np.array([p_central,p_off],[p_off,p_central])
    else:
        W[0,0]=p_central
        W[0,1]=p_off
        W[0,d-1]=p_off
        W[d-1,d-1]=p_central
        W[d-1,d-2]=p_off
        W[d-1,0]=p_off
        for i in range(1,d-1):
            W[i,i]=p_central
            W[i,i-1]=p_off
            W[i,i+1]=p_off
    return W

def init(seed_init=1,state_space=None,action_spaces=None,init_xi=None,transP=None,reward=None,\
         gamma=0.95,W=None,V_features=None,R_features=None):
    DAC_dict={}
    set_seed(seed_init)
    DAC_dict['seed_init']=seed_init

    DAC_dict['state_space']=set_hyps(a=state_space,a_default=range(num_states))   
    DAC_dict['action_spaces']=set_hyps(a=action_spaces,a_default=[list(range(num_actions))]*num_agents)  
    DAC_dict['num_states']=len(DAC_dict['state_space'])
    DAC_dict['num_actions']=[len(tmp) for tmp in DAC_dict['action_spaces']].copy()    #DAC_dict['num_actions'][m] for agent m
    DAC_dict['num_agents']=len(DAC_dict['num_actions'])
    
    if init_xi is None:
        init_xi=np.random.normal(size=(DAC_dict['num_states']))
    else:
        assert init_xi.size==DAC_dict['num_states'], "init_xi should have "+str(DAC_dict['num_states'])+" entries."
    init_xi=np.abs(init_xi).reshape(DAC_dict['num_states'])
    DAC_dict['init_xi']=init_xi/init_xi.sum()
    
    transP_shape=tuple([DAC_dict['num_states']]+DAC_dict['num_actions']+[DAC_dict['num_states']])
    if transP is None:
        DAC_dict['transP']=np.abs(np.random.normal(size=transP_shape))   #P(s,a1,...,aM,s')
        DAC_dict['transP']=DAC_dict['transP']/np.sum(DAC_dict['transP'],axis=DAC_dict['num_agents']+1,keepdims=True)
    else:
        assert transP.shape == transP_shape, \
            "transP should have shape: (num_states,num_actions1,...,num_actionsM,num_states)"
        transP=np.abs(transP)
        DAC_dict['transP']=transP/np.sum(transP,axis=DAC_dict['num_agents']+1,keepdims=True)

    DAC_dict['gamma']=gamma
    newshape=tuple([1,]*(DAC_dict['num_agents']+1)+[DAC_dict['num_states']])
    DAC_dict['P_xi']=DAC_dict['gamma']*DAC_dict['transP']+(1-DAC_dict['gamma'])*DAC_dict['init_xi'].reshape(newshape)
    
    reward_shape=tuple([DAC_dict['num_states']]+DAC_dict['num_actions']+[DAC_dict['num_states']]+[DAC_dict['num_agents']])
    if (reward is not None):
        assert reward.shape==reward_shape,\
            "reward should be either None or an np.array with shape (num_states,num_actions1,...,num_actionsM,num_states,num_agents)"
    DAC_dict['reward'] = set_hyps(a=reward,a_default=np.random.uniform(size=reward_shape)) #获取奖励R
    DAC_dict['reward_agentavg']=DAC_dict['reward'].mean(axis=DAC_dict['num_agents']+2) #获得平均奖励

    if W is None:
        DAC_dict['W']=get_W_3diags(d=DAC_dict['num_agents'],p_central=0.4)
    else:
        assert W.shape==(DAC_dict['num_agents'],DAC_dict['num_agents']),"W should have shape (num_agents,num_agents)"
        DAC_dict['W']=np.abs(W)
    if DAC_dict['num_agents']>1:
        assert np.abs(np.sum(DAC_dict['W'],axis=0)-1).max()<1e-12, "W should be doubly stochastic"
        assert np.abs(np.sum(DAC_dict['W'],axis=1)-1).max()<1e-12, "W should be doubly stochastic"
        assert np.abs(DAC_dict['W']-DAC_dict['W'].T).max()<1e-12, "W should be symmetric"
        u,s,vh=np.linalg.svd(DAC_dict['W'])
        s=s[1]
        assert (s>=0)&(s<1),"The second largest singular value of W should be in [0,1)"
    
    if V_features is None:
        DAC_dict['V_features']=np.diag([1.0]*DAC_dict['num_states'])
    else: 
        assert V_features.shape[1]==DAC_dict['num_states'], "V_features should be a 2d-array with #states columns"
        DAC_dict['V_features']=V_features/np.sqrt(np.sum(V_features**2,axis=0,keepdims=True))
    DAC_dict['num_V_features']=DAC_dict['V_features'].shape[0]
        
    if R_features is None:
        DAC_dict['num_R_features']=DAC_dict['num_states']*DAC_dict['num_states']*np.prod(DAC_dict['num_actions'])
        R_features_shape=(DAC_dict['num_R_features'],DAC_dict['num_states'],)+tuple(DAC_dict['num_actions'])+(DAC_dict['num_states'],)
        DAC_dict['R_features']=np.diag([1.0]*DAC_dict['num_R_features']).reshape(R_features_shape)
    else: 
        DAC_dict['num_R_features']=R_features.shape[0]
        R_features_shape=(DAC_dict['num_R_features'],DAC_dict['num_states'],)+tuple(DAC_dict['num_actions'])+(DAC_dict['num_states'],)
        assert R_features.shape==R_features_shape, 'R_features[k,s,a0,a1,...,a_{M-1},s_next] denotes the k-th feature'
        DAC_dict['R_features']=R_features.copy()
    # R_features[k,s,a0,a1,...,a_{M-1},s_next] denotes the k-th feature
    
    DAC_dict['results']={}
    
    return DAC_dict

def get_hyp_str(hyp):
    hyp1=hyp.copy()
    hyp1.pop('plot_iters')
    hyp1.pop('color')
    hyp1.pop('marker')
    hyp1.pop('legend')
    hyp1.pop('result_dir')
    return str(hyp1)

def get_pim(omega): #这个函数的目的是接受一个包含代理数据的 omega 输入，然后计算每个代理的概率分布，并将这些概率分布存储在一个列表中返回。
    num_agents=len(omega)
    pim=[0]*num_agents
    for m in range(num_agents):
        tmp=np.exp(omega[m])
        pim[m]=tmp/np.sum(tmp,axis=1,keepdims=True)
        #进行归一化处理：计算了 tmp 沿着轴 axis=1（即按行求和）的结果，keepdims=True 选项保持了结果的维度，使其仍然是一个二维数组。最后，将 tmp 中的每一行都除以相应行的总和。这个操作将每个状态或事件的值转化为一个概率，确保所有概率值的总和为1。
    return pim

def get_transP_s2s(pim,transP): #From P(s'|s,a), obtain P(s'|s)，这个函数用于计算在给定代理策略的情况下，从一个状态 s 转移到另一个状态 s' 的概率分布。这个计算涉及将代理的策略概率与状态转移概率矩阵相乘，并对相关轴进行求和，以得到最终的概率分布。
    num_agents=len(pim)
    num_states=transP.shape[0]
    transP_s2s=transP.copy()
    for m in range(num_agents):
        num_actions=pim[m].shape[1]
        newshape=(num_states,)+(1,)*m+(num_actions,)+(1,)*(num_agents-m)
        transP_s2s*=pim[m].reshape(newshape)
    newshape=tuple(range(1,num_agents+1))
    return transP_s2s.sum(axis=newshape)

def stationary_dist(transP_s2s): #Stationary distribution corresponding to pi_omega and transP，从P(s'|s)中获取稳态分布
    evals, evecs = np.linalg.eig(transP_s2s.T) #P.T*evecs=evecs*np.diag(evals) 使用函数计算特征值和特征向量
    evec1 = evecs[:, np.isclose(evals, 1)] #从特征向量中选择特征值接近1的部分。这是通过 np.isclose 函数来实现的，它用于找到与1接近的特征值的索引。
    evec1 = np.abs(evec1[:, 0]) #将选定的特征向量 evec1 的所有元素取绝对值，以确保所有概率值都是正数。
    stationary = evec1 / evec1.sum() #将取绝对值后的特征向量 evec1 归一化，以确保其所有元素之和为1。这样得到的 stationary 即为系统的稳态分布。
    return stationary.real

def get_entropy_agentavg(pim,alpha_h):
    num_agents=len(pim)
    entropy=[] # 用于存储每个代理的信息熵

    for m in range(num_agents): # 计算每个代理的信息熵并添加到列表中
        entropy_agent=-np.sum(pim[m]*np.log(pim[m]+1e-10),axis=1)
        entropy.append(entropy_agent)

    # 将信息熵平均后乘以参数 alpha_h
    entropy_agentavg=alpha_h*np.mean(entropy)

    return entropy_agentavg

def J_h(pim,transP,P_xi,reward_agentavg,alpha_h,nu_omega=None):  # nu_omega是平稳状态分布
    num_agents=len(pim)
    num_states=transP.shape[0]

    if nu_omega is None:
        P_xi_s2s=get_transP_s2s(pim, P_xi)
        nu_omega=stationary_dist(P_xi_s2s)

    entropy_agentavg=get_entropy_agentavg(pim, alpha_h)
    J1=((reward_agentavg+entropy_agentavg)*transP).sum(axis=num_agents+1)
    for m in range(num_agents):
        num_actions=pim[m].shape[1]
        newshape=(num_states,)+(1,)*m+(num_actions,)+(1,)*(num_agents-m-1)
        J1*=pim[m].reshape(newshape)
    newshape=tuple(range(1,num_agents+1))
    J1=J1.sum(axis=newshape)
    J1=(J1*nu_omega).sum()
    return J1


def J(pim, transP, P_xi, reward_agentavg, nu_omega=None):  # nu_omega是平稳状态分布
    num_agents = len(pim)
    num_states = transP.shape[0]

    if nu_omega is None:
        P_xi_s2s = get_transP_s2s(pim, P_xi)
        nu_omega = stationary_dist(P_xi_s2s)

    J1 = (reward_agentavg * transP).sum(axis=num_agents + 1)
    for m in range(num_agents):
        num_actions = pim[m].shape[1]
        newshape = (num_states,) + (1,) * m + (num_actions,) + (1,) * (num_agents - m - 1)
        J1 *= pim[m].reshape(newshape)
    newshape = tuple(range(1, num_agents + 1))
    J1 = J1.sum(axis=newshape)
    J1 = (J1 * nu_omega).sum()
    return J1


def V_vech(pim,transP,reward_agentavg,alpha_h,gamma): #其主要目的是计算每个状态下的价值函数向量。
    num_states=transP.shape[0]
    num_agents=len(pim)
    Vvec=np.zeros(num_states) #创建一个名为 Vvec 的NumPy数组，用于存储每个状态下的价值函数。
    for s in range(num_states):
        P_xi=np.zeros_like(transP)
        exec("P_xi["+":,"*(num_agents+1)+"s]=1-gamma") #使用 exec 函数动态生成代码，目的是将矩阵 P_xi 中状态 s 对应的行设置为 1-gamma，而其他行保持为零。这个操作是为了表示在状态 s 下采取任何动作后的状态保持不变的概率为 1-gamma。
        P_xi+=gamma*transP #将状态转移概率矩阵 transP 中的所有转移概率乘以 gamma 并加到 P_xi 上，以表示采取动作后的状态转移概率。这里假设了一个马尔可夫决策过程中的状态转移模型，其中 (1-gamma) 表示状态保持不变的概率，gamma 表示状态转移的概率。
        Vvec[s]=J_h(pim,transP,P_xi,reward_agentavg,alpha_h,nu_omega=None)/(1-gamma) #使用函数 J 计算在状态 s 下的性能度量，并将其除以 (1-gamma)，得到在状态 s 下的价值函数。
    return Vvec

def V_vec(pim,transP,reward_agentavg,gamma):
    num_states=transP.shape[0]
    num_agents=len(pim)
    Vvec=np.zeros(num_states)
    for s in range(num_states):
        P_xi=np.zeros_like(transP)
        exec("P_xi["+":,"*(num_agents+1)+"s]=1-gamma")
        P_xi+=gamma*transP
        Vvec[s]=J(pim,transP,P_xi,reward_agentavg,nu_omega=None)/(1-gamma)
    return Vvec

def dJ_h(pim,transP,nu_omega,reward_agentavg,alpha_h,gamma):
    num_agents=len(pim)
    num_states=transP.shape[0]
    Vvec=V_vech(pim,transP,reward_agentavg,alpha_h,gamma)
    shape1=(1,)*(num_agents+1)+(num_states,)
    shape2=(num_states,)+(1,)*(num_agents+1)
    entropy_agentavg=get_entropy_agentavg(pim,alpha_h)
    Api=reward_agentavg+entropy_agentavg+gamma*Vvec.reshape(shape1)-Vvec.reshape(shape2) #优势函数
    Api=(Api*transP).sum(axis=num_agents+1)  #Advantage function of (s,a) 求期望
    for m in range(num_agents):
        num_actions=pim[m].shape[1]
        newshape=(num_states,)+(1,)*m+(num_actions,)+(1,)*(num_agents-1-m)
        Api*=pim[m].reshape(newshape)
        #Api=pi(a|s)*Advantage function of (s,a)
    
    dJ_list=[0]*num_agents
    dJ_norm_sq=0
    nu_omega=nu_omega.reshape(num_states,1)
    for m in range(num_agents):
        num_actions=pim[m].shape[1]
        sum_axis=list(range(1,num_agents+1))
        sum_axis.pop(m)
        Am=Api.sum(axis=tuple(sum_axis))  #Am[s,am]=sum_{a^{(-m)}} pi(a|s)*Advantage function of (s,a)
        Am=Am-Am.sum(axis=1).reshape(num_states,1)*pim[m]

        dJ_list[m]=nu_omega*Am
        dJ_norm_sq+=(dJ_list[m]**2).sum()
    return dJ_list, dJ_norm_sq


def dJ(pim, transP, nu_omega, reward_agentavg, gamma):
    num_agents = len(pim)
    num_states = transP.shape[0]
    Vvec = V_vec(pim, transP, reward_agentavg, gamma)
    shape1 = (1,) * (num_agents + 1) + (num_states,)
    shape2 = (num_states,) + (1,) * (num_agents + 1)
    Api = reward_agentavg + +gamma * Vvec.reshape(shape1) - Vvec.reshape(shape2)
    Api = (Api * transP).sum(axis=num_agents + 1)  # Advantage function of (s,a)
    for m in range(num_agents):
        num_actions = pim[m].shape[1]
        newshape = (num_states,) + (1,) * m + (num_actions,) + (1,) * (num_agents - 1 - m)
        Api *= pim[m].reshape(newshape)
        # Api=pi(a|s)*Advantage function of (s,a)

    dJ_list = [0] * num_agents
    dJ_norm_sq = 0
    nu_omega = nu_omega.reshape(num_states, 1)
    for m in range(num_agents):
        num_actions = pim[m].shape[1]
        sum_axis = list(range(1, num_agents + 1))
        sum_axis.pop(m)
        Am = Api.sum(axis=tuple(sum_axis))  # Am[s,am]=sum_{a^{(-m)}} pi(a|s)*Advantage function of (s,a)
        Am = Am - Am.sum(axis=1).reshape(num_states, 1) * pim[m]

        dJ_list[m] = nu_omega * Am
        dJ_norm_sq += (dJ_list[m] ** 2).sum()
    return dJ_list, dJ_norm_sq

def get_optimal_pi(transP, P_xi, reward_agentavg, gamma, eps=1e-7, DAC_dict=None):
    num_states=transP.shape[0]
    num_actions=list(transP.shape)
    num_actions.pop(0)
    num_actions.pop(-1)
    num_agents=len(num_actions)
    optimal_pi=np.array([[0]*num_states]*num_agents)
    for m in range(num_agents):
        num_actionm=num_actions[m]
        optimal_pi[m]=np.random.choice(a=range(num_actionm),size=(num_states),p=[1/num_actionm]*num_actionm) #对于代理 m，在每个状态下随机选择一个动作。这里使用了 np.random.choice 函数，它从给定的动作空间 range(num_actionm) 中以均匀的概率选择一个动作，并将选择的动作赋值给 optimal_pi。
    #Agent m will take the deterministic action am=optimal_pi[m,s] for state s.

    newshape=(1,)*(num_agents+1)+(num_states,)
    
    #Vmax iteration
    dVmax = eps+1
    Vmax = np.random.uniform(size=(num_states))
    while dVmax>=eps:
        Vmax_next=((reward_agentavg+(gamma*Vmax).reshape(newshape))*transP).sum(axis=num_agents+1)\
        .reshape((num_states,-1)).max(axis=1)
        dVmax=np.max(np.abs(Vmax_next-Vmax))
        Vmax=Vmax_next.copy()

    Qmax = ((reward_agentavg+(gamma*Vmax).reshape(newshape))*transP).sum(axis=num_agents+1)
    max_idx = Qmax.reshape(Qmax.shape[0],-1).argmax(1) #找到每个状态下具有最大动作值的动作的索引。这将用于确定最优策略。
    optimal_pi = np.column_stack(np.unravel_index(max_idx, Qmax[0,:,:].shape)) #根据最大动作值的索引，构建最优策略 optimal_pi。这是一个二维数组，其中每行代表一个状态，每列代表一个代理，策略值是代理在给定状态下选择的动作。
    ER = np.zeros((num_states)) #初始化一个数组 ER，用于存储最优策略下的期望累积奖励。
    
    optimal_pim=[0]*num_agents #创建一个列表 optimal_pim，用于存储每个代理的最优策略。
    for m in range(num_agents):
        num_actionm=num_actions[m]
        optimal_pim[m]=np.zeros((num_states,num_actionm))
        optimal_pim[m][range(num_states),optimal_pi[:,m]]=1
    transP_s2s=get_transP_s2s(optimal_pim,transP)
    ER=eval('(reward_agentavg*transP).sum(axis=num_agents+1)[range('+str(num_states)+')'+\
        ''.join([',optimal_pi[:,'+str(m)+']' for m in range(num_agents)])+']') #计算最优策略下的期望累积奖励 ER。这是通过对每个状态下代理采取最优策略后的预期累积奖励求和得到的。
    Vmax=np.linalg.solve(a=np.diag([1.0]*num_states)-gamma*transP_s2s,b=ER) #这个值函数表示在每个状态下采取最优策略后的预期累积奖励。
    
    return optimal_pi,optimal_pim,Vmax

def DTD_h(pim, s0, Tc, W, Wpow_critic, Nc, beta, gamma, transP, reward, alpha_h, features, theta0=None, seed_sim=200, is_print_error=False, theta_star=None,
        DAC_dict=None):
    set_seed(seed_sim)
    num_agents=len(pim)
    num_states=transP.shape[0]
    num_actions=[pim[m].shape[1] for m in range(num_agents)]
    num_features=features.shape[0]
    num_samples=Tc*Nc
    if theta0 is None:
        theta0=np.random.normal(size=(num_features))
        
    if len(theta0.shape)==1:
        Theta=np.ones((num_agents,1)).dot(theta0.reshape(1,-1))  #Each row per agent 
    elif (theta0.shape[0]==1 | theta0.shape[1]==1):
        Theta=np.ones((num_agents,1)).dot(theta0.reshape(1,-1)) 
    else:
        Theta=theta0.copy()
    
    s=np.array([s0]*(num_samples+1))
    a=np.array([[0]*num_agents]*(num_samples)) 
    R=np.zeros((num_samples,num_agents))
    H=np.zeros((num_samples, num_agents))

    for t in range(Tc):
        tN=t*Nc
        i_range=range(tN,tN+Nc)
        for i in i_range:
            #query samples
            s_now=s[i]
            index="[s_now"
            for m in range(num_agents):
                pp=pim[m][s_now]
                a[i,m]=np.random.choice(a=range(num_actions[m]),size=1,p=pp/pp.sum())[0]
                
                index+=","+str(a[i,m])
            Pnow=eval("transP"+index+",:]")
            s_next=np.random.choice(a=range(num_states),size=1,p=Pnow)[0]
            s[i+1]=s_next
            R[i]=eval("reward"+index+",s_next,:]")
            for m in range(num_agents):
                entropy=-np.sum(pim[m]*np.log(pim[m]+1e-10))
                H[i, m]=entropy

        #TD update
        phi_now=features[:,s[i_range]]
        phi_next=features[:,s[range(tN+1,tN+Nc+1)]]
        Bt=phi_now.dot((gamma*phi_next-phi_now).T)
        bt=phi_now.dot(R[i_range,:])
        Ht=phi_now.dot(H[i_range,:])
        Theta=W.dot(Theta)+(beta/Nc)*(Theta.dot(Bt.T)+bt.T+alpha_h*Ht.T) #Theta*Bt的转置
        
        if is_print_error:
            print("t="+str(t)+": error="+str(np.abs(Theta.mean(axis=0)-theta_star.reshape(-1)).max())+"\n")

    return Wpow_critic.dot(Theta)


def DTD(pim, s0, Tc, W, Wpow_critic, Nc, beta, gamma, transP, reward, features, theta0=None, seed_sim=200,
        is_print_error=False, theta_star=None):
    set_seed(seed_sim)
    num_agents = len(pim)
    num_states = transP.shape[0]
    num_actions = [pim[m].shape[1] for m in range(num_agents)]
    num_features = features.shape[0]
    num_samples = Tc * Nc
    if theta0 is None:
        theta0 = np.random.normal(size=(num_features))

    if len(theta0.shape) == 1:
        Theta = np.ones((num_agents, 1)).dot(theta0.reshape(1, -1))  # Each row per agent
    elif (theta0.shape[0] == 1 | theta0.shape[1] == 1):
        Theta = np.ones((num_agents, 1)).dot(theta0.reshape(1, -1))
    else:
        Theta = theta0.copy()

    s = np.array([s0] * (num_samples + 1))
    a = np.array([[0] * num_agents] * (num_samples))
    R = np.zeros((num_samples, num_agents))

    for t in range(Tc):
        tN = t * Nc
        i_range = range(tN, tN + Nc)
        for i in i_range:
            # query samples
            s_now = s[i]
            index = "[s_now"
            for m in range(num_agents):
                pp = pim[m][s_now]
                a[i, m] = np.random.choice(a=range(num_actions[m]), size=1, p=pp / pp.sum())[0]

                index += "," + str(a[i, m])
            Pnow = eval("transP" + index + ",:]")
            s_next = np.random.choice(a=range(num_states), size=1, p=Pnow)[0]
            s[i + 1] = s_next
            R[i] = eval("reward" + index + ",s_next,:]")

        # TD update
        phi_now = features[:, s[i_range]]
        phi_next = features[:, s[range(tN + 1, tN + Nc + 1)]]
        Bt = phi_now.dot((gamma * phi_next - phi_now).T)
        bt = phi_now.dot(R[i_range, :])
        Theta = W.dot(Theta) + (beta / Nc) * (Theta.dot(Bt.T) + bt.T)

        if is_print_error:
            print("t=" + str(t) + ": error=" + str(np.abs(Theta.mean(axis=0) - theta_star.reshape(-1)).max()) + "\n")

    return Wpow_critic.dot(Theta)

def get_optimal_theta_h(pim, reward, alpha_h, features, gamma, transP, transP_s2s=None, mu=None):
    num_agents=len(pim)
    num_states=transP.shape[0]
    if transP_s2s is None:
        transP_s2s=get_transP_s2s(pim=pim,transP=transP)
        
    if mu is None:
        mu=stationary_dist(transP_s2s)
    D=np.diag(mu)
    pim_array = np.array(pim)
    entropy = -np.sum(pim_array * np.log(pim_array + 1e-10))
    B=features.dot(gamma*D.dot(transP_s2s)-D).dot(features.T)
    b=((reward+alpha_h*entropy)*transP.reshape(transP.shape+(1,))).sum(axis=num_agents+1)
    for m in range(num_agents):
        num_actions=pim[m].shape[1]
        b*=pim[m].reshape((num_states,)+(1,)*m+(num_actions,)+(1,)*(num_agents-m))
    b=b.sum(axis=tuple(range(1,num_agents+1)))
    b=features.dot(D).dot(b)
    return -np.linalg.solve(a=B,b=b.mean(axis=1))


def get_optimal_theta(pim, reward, features, gamma, transP, transP_s2s=None, mu=None):
    num_agents = len(pim)
    num_states = transP.shape[0]
    if transP_s2s is None:
        transP_s2s = get_transP_s2s(pim=pim, transP=transP)

    if mu is None:
        mu = stationary_dist(transP_s2s)
    D = np.diag(mu)

    B = features.dot(gamma * D.dot(transP_s2s) - D).dot(features.T)
    b = (reward * transP.reshape(transP.shape + (1,))).sum(axis=num_agents + 1)
    for m in range(num_agents):
        num_actions = pim[m].shape[1]
        b *= pim[m].reshape((num_states,) + (1,) * m + (num_actions,) + (1,) * (num_agents - m))
    b = b.sum(axis=tuple(range(1, num_agents + 1)))
    b = features.dot(D).dot(b)
    return -np.linalg.solve(a=B, b=b.mean(axis=1))


def DAC(DAC_dict, T, Tc, Tc_prime, Tr, N, Nc, alpha, beta, noise_std, s0_DAC=None, s0_DTD=None, omega0=None, \
        theta0=None, h0=None, seed_sim_DAC=100, seed_sim_DTD=200, is_print=False, is_save=False, \
        getRvg_err_every_numiter=None, getDTD_err_every_numiter=None, save_folder="DAC_results/"):
    start_time = time.time()
    DAC_dict = DAC_dict.copy()
    set_seed(seed_sim_DAC)

    num_samples = T * N

    DAC_dict['T'] = T
    DAC_dict['Tc'] = Tc
    DAC_dict['Tc_prime'] = Tc_prime
    DAC_dict['Tr'] = Tr
    DAC_dict['N'] = N
    DAC_dict['Nc'] = Nc
    DAC_dict['alpha'] = alpha
    DAC_dict['beta'] = beta
    DAC_dict['noise_std'] = noise_std
    DAC_dict['seed_sim_DAC'] = seed_sim_DAC
    DAC_dict['seed_sim_DTD'] = seed_sim_DTD
    DAC_dict['is_print'] = is_print
    DAC_dict['is_save'] = is_save

    if s0_DAC is None:
        s0_DAC = np.random.choice(a=range(DAC_dict['num_states']), size=1, p=DAC_dict['init_xi'])[0]

    if s0_DTD is None:
        s0_DTD = np.random.choice(a=range(DAC_dict['num_states']), size=1, p=DAC_dict['init_xi'])[0]

    DAC_dict['s0_DTD'] = s0_DTD

    if (omega0 is None):
        omega0 = [np.random.normal(size=(DAC_dict['num_states'], DAC_dict['num_actions'][m])) for m in
                  range(DAC_dict['num_agents'])]

    DAC_dict['omega'] = [0] * DAC_dict['num_agents']
    for m in range(DAC_dict['num_agents']):
        DAC_dict['omega'][m] = np.zeros((DAC_dict['num_states'], DAC_dict['num_actions'][m], T))
        # DAC_dict['omega'][m][s,am,t]  m-th agent, t-th iteration
        DAC_dict['omega'][m][:, :, 0] = omega0[m].copy()

    del omega0

    if theta0 is None:
        theta0 = np.random.normal(size=DAC_dict['num_V_features'])
    DAC_dict['Theta'] = np.zeros((DAC_dict['num_agents'], DAC_dict['num_V_features'], T))
    # Theta[m,:,t]: theta vector of agent m, at t-th iteration

    DAC_dict['s'] = np.array([s0_DAC] * (num_samples + 1))
    DAC_dict['a'] = np.array([[0] * DAC_dict['num_agents']] * (num_samples))  # DAC_dict['a'][i,m] is a_i of agent m
    DAC_dict['R'] = np.zeros((num_samples, DAC_dict['num_agents']))  # DAC_dict['R'][i,m] is R_i of agent m
    DAC_dict['s_prime'] = np.array([s0_DAC] * (num_samples + 1))  # DAC_dict['s_prime'][i] means s_{i+1}'

    DAC_dict['Jw'] = np.zeros(T)
    DAC_dict['dJ_normsq'] = np.zeros(T)
    DAC_dict['Jw_cummean'] = np.zeros(T)
    DAC_dict['dJ_normsq_cummean'] = np.zeros(T)

    DAC_dict['getRvg_err_every_numiter'] = getRvg_err_every_numiter
    DAC_dict['getDTD_err_every_numiter'] = getDTD_err_every_numiter
    if getRvg_err_every_numiter is not None:
        DAC_dict['absolute_Ravg_err'] = -np.ones(T)
        DAC_dict['relative_Ravg_err'] = -np.ones(T)
    if getDTD_err_every_numiter is not None:
        DAC_dict['absolute_DTD_err'] = -np.ones(T)
        DAC_dict['relative_DTD_err'] = -np.ones(T)

    Wpow_critic = np.linalg.matrix_power(DAC_dict['W'], Tc_prime)
    Wpow_reward = np.linalg.matrix_power(DAC_dict['W'], Tr)

    for t in range(T - 1):
        pim = get_pim([DAC_dict['omega'][m][:, :, t] for m in range(DAC_dict['num_agents'])])
        if t > 0:
            theta0 = DAC_dict['Theta'][:, :, t - 1].copy()
        DAC_dict['Theta'][:, :, t] = DTD \
            (pim=pim, s0=s0_DTD, Tc=Tc, W=DAC_dict['W'], Wpow_critic=Wpow_critic, \
             Nc=Nc, beta=beta, gamma=DAC_dict['gamma'], transP=DAC_dict['transP'], \
             reward=DAC_dict['reward'], features=DAC_dict['V_features'], \
             theta0=theta0, seed_sim=seed_sim_DTD)

        if getDTD_err_every_numiter is not None:
            if t % getDTD_err_every_numiter == 0:
                theta_star = get_optimal_theta(pim, DAC_dict['reward'], DAC_dict['V_features'], DAC_dict['gamma'],
                                               DAC_dict['transP'], None, None)
                DAC_dict['absolute_DTD_err'][t] = np.sum(
                    (DAC_dict['Theta'][:, :, t] - theta_star.reshape((1, -1))) ** 2) / DAC_dict['num_agents']
                DAC_dict['relative_DTD_err'][t] = DAC_dict['absolute_DTD_err'][t] / np.sum(theta_star ** 2)

        tN = t * N
        i_range = range(tN, tN + N)

        for m in range(DAC_dict['num_agents']):
            DAC_dict['omega'][m][:, :, t + 1] = DAC_dict['omega'][m][:, :, t].copy()

        if getRvg_err_every_numiter is not None:
            if t % getRvg_err_every_numiter == 0:
                R_agentavg_sum = 0
                Rhat_sum = 0

        for i in i_range:
            # query samples
            s_now = DAC_dict['s'][i]
            index = "[s_now"
            for m in range(DAC_dict['num_agents']):
                pp = pim[m][s_now]
                DAC_dict['a'][i, m] = np.random.choice(a=range(DAC_dict['num_actions'][m]), size=1, p=pp / pp.sum())[0]
                index += "," + str(DAC_dict['a'][i, m])

            pp = eval("DAC_dict['P_xi']" + index + ",:]")
            s_next = np.random.choice(a=range(DAC_dict['num_states']), size=1, p=pp)[0]
            DAC_dict['s'][i + 1] = s_next

            pp = eval("DAC_dict['transP']" + index + ",:]")
            s_prime = np.random.choice(a=range(DAC_dict['num_states']), size=1, p=pp)[0]
            DAC_dict['s_prime'][i] = s_prime

            DAC_dict['R'][i] = eval("DAC_dict['reward']" + index + ",s_prime,:]")
            R_hat = 1 + np.random.standard_t(df=4, size=(DAC_dict['num_agents'])) * (noise_std / np.sqrt(2))
            R_hat = Wpow_reward.dot(DAC_dict['R'][i] * R_hat)

            if getRvg_err_every_numiter is not None:
                if t % getRvg_err_every_numiter == 0:
                    R_agentavg_sum += DAC_dict['R'][i].mean()
                    Rhat_sum += R_hat

            TD_coeff = DAC_dict['gamma'] * DAC_dict['V_features'][:, s_prime] \
                       - DAC_dict['V_features'][:, s_now]

            for m in range(DAC_dict['num_agents']):
                # Actor update using AC with sample i
                psim_coeff = (R_hat[m] + (TD_coeff * DAC_dict['Theta'][m, :, t]).sum()) * alpha / N
                psim_si = -pim[m][s_now]
                psim_si[DAC_dict['a'][i, m]] += 1
                DAC_dict['omega'][m][s_now, :, t + 1] += psim_coeff * psim_si

        P_xi_s2s = get_transP_s2s(pim, DAC_dict['P_xi'])
        nu_omega = stationary_dist(P_xi_s2s)
        Jw = J(pim, DAC_dict['transP'], DAC_dict['P_xi'], DAC_dict['reward_agentavg'], nu_omega)
        _, dJ_normsq = dJ(pim, DAC_dict['transP'], nu_omega, DAC_dict['reward_agentavg'], DAC_dict['gamma'])
        DAC_dict['Jw'][t] = Jw
        DAC_dict['dJ_normsq'][t] = dJ_normsq
        if t >= 1:
            Jw_cummean = DAC_dict['Jw_cummean'][t - 1] * t / (t + 1) + Jw / (t + 1)
            dJ_normsq_cummean = DAC_dict['dJ_normsq_cummean'][t - 1] * t / (t + 1) + dJ_normsq / (t + 1)
        else:
            Jw_cummean = Jw
            dJ_normsq_cummean = dJ_normsq

        DAC_dict['Jw_cummean'][t] = Jw_cummean
        DAC_dict['dJ_normsq_cummean'][t] = dJ_normsq_cummean

        if getRvg_err_every_numiter is not None:
            if t % getRvg_err_every_numiter == 0:
                R_agentavg_batchavg = R_agentavg_sum / N
                Rhat_batchavg = Rhat_sum / N
                DAC_dict['absolute_Ravg_err'][t] = ((R_agentavg_batchavg - Rhat_batchavg) ** 2).mean()
                DAC_dict['relative_Ravg_err'][t] = DAC_dict['absolute_Ravg_err'][t] / (R_agentavg_batchavg ** 2)

        if is_print:
            print("Iteration " + str(t) + ", J=" + str(Jw) + ", ||dJ||^2=" + str(dJ_normsq))
            print("J_cum_mean=" + str(Jw_cummean) + ", dJ_normsq_cum_mean=" + str(dJ_normsq_cummean) + "\n")

            if getRvg_err_every_numiter is not None:
                if t % getRvg_err_every_numiter == 0:
                    print("avg_{m,s,a,s_next} [R_avg(s,a,s_next)-lambda1[m]*f(s,a,s_next)]^2=" + str(
                        DAC_dict['absolute_Ravg_err'][t]) + "\n")
                    print("relative R estimation error=" + str(DAC_dict['relative_Ravg_err'][t]) + "\n")
                else:
                    print("\n")
            else:
                print("\n")

            if getDTD_err_every_numiter is not None:
                if t % getDTD_err_every_numiter == 0:
                    print("avg_{m} ||theta[m]-theta*||^2=" + str(DAC_dict['absolute_DTD_err'][t]) + "\n")
                    print("avg_{m} ||theta[m]-theta*||^2/||theta*||^2=" + str(DAC_dict['relative_DTD_err'][t]) + "\n")
                else:
                    print("\n")
            else:
                print("\n")

    # Get J and dJ of the final iteration
    t = T - 1
    pim = get_pim([DAC_dict['omega'][m][:, :, t] for m in range(DAC_dict['num_agents'])])
    P_xi_s2s = get_transP_s2s(pim, DAC_dict['P_xi'])
    nu_omega = stationary_dist(P_xi_s2s)
    Jw = J(pim, DAC_dict['transP'], DAC_dict['P_xi'], DAC_dict['reward_agentavg'], nu_omega)
    _, dJ_normsq = dJ(pim, DAC_dict['transP'], nu_omega, DAC_dict['reward_agentavg'], DAC_dict['gamma'])
    DAC_dict['Jw'][t] = Jw
    dJ_normsq_cummean = DAC_dict['dJ_normsq_cummean'][t - 1] * t / (t + 1) + dJ_normsq / (t + 1)

    DAC_dict['Jw_cummean'][t] = Jw_cummean
    DAC_dict['dJ_normsq_cummean'][t] = dJ_normsq_cummean
    DAC_dict['time(s)'] = time.time() - start_time

    if is_print:
        print("Iteration " + str(t) + ", J=" + str(Jw) + ", ||dJ||^2=" + str(dJ_normsq))
        print("J_cum_mean=" + str(Jw_cummean) + ", dJ_normsq_cum_mean=" + str(dJ_normsq_cummean) + "\n")

        if getDTD_err_every_numiter is not None:
            if t % getDTD_err_every_numiter == 0:
                print("avg_{m} ||theta[m]-theta*||^2=" + str(DAC_dict['absolute_DTD_err'][t]) + "\n")
                print("avg_{m} ||theta[m]-theta*||^2/||theta*||^2=" + str(DAC_dict['relative_DTD_err'][t]) + "\n")
            else:
                print("\n")
        else:
            print("\n")

    if is_save:
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        np.save(file=save_folder + '/theta.npy', arr=DAC_dict['Theta'])
        np.save(file=save_folder + '/Jw.npy', arr=DAC_dict['Jw'])
        np.save(file=save_folder + '/dJ_normsq.npy', arr=DAC_dict['dJ_normsq'])
        np.save(file=save_folder + '/Jw_cummean.npy', arr=DAC_dict['Jw_cummean'])
        np.save(file=save_folder + '/dJ_normsq_cummean.npy', arr=DAC_dict['dJ_normsq_cummean'])

        for m in range(DAC_dict['num_agents']):
            np.save(file=save_folder + '/omega_agent' + str(m) + '.npy', arr=DAC_dict['omega'][m])

        hyp_txt = open(save_folder + '/hyperparameters.txt', 'a')
        keys = ['T', 'Tc', 'Tc_prime', 'Tr', 'N', 'Nc', 'alpha', 'beta']
        keys += ['noise_std', 'seed_sim_DAC', 'seed_sim_DTD', 'is_print', 'is_save']
        for key in keys:
            hyp_txt.write(key + '=' + str(DAC_dict[key]) + '\n')
        hyp_txt.write('Time consumption: ' + str(DAC_dict['time(s)'] / 60) + ' minutes\n')
        hyp_txt.close()

    return DAC_dict

def DAC_h(DAC_dict,T,Tc,Tc_prime,Tr,N,Nc,alpha,alpha_h,beta,noise_std,s0_DAC=None,s0_DTD=None,omega0=None,\
        theta0=None,h0=None,seed_sim_DAC=100,seed_sim_DTD=200,is_print=False,is_save=False,\
        getRvg_err_every_numiter=None,getHvg_err_every_numiter=None,getDTD_err_every_numiter=None,save_folder="DAC_results/"):
    
    start_time=time.time()
    DAC_dict=DAC_dict.copy()
    set_seed(seed_sim_DAC)
    
    num_samples=T*N
    
    DAC_dict['T']=T
    DAC_dict['Tc']=Tc
    DAC_dict['Tc_prime']=Tc_prime
    DAC_dict['Tr']=Tr
    DAC_dict['N']=N
    DAC_dict['Nc']=Nc
    DAC_dict['alpha']=alpha
    DAC_dict['alpha_h']=alpha_h
    DAC_dict['beta']=beta
    DAC_dict['noise_std']=noise_std
    DAC_dict['seed_sim_DAC']=seed_sim_DAC
    DAC_dict['seed_sim_DTD']=seed_sim_DTD
    DAC_dict['is_print']=is_print
    DAC_dict['is_save']=is_save
        
    if s0_DAC is None:
        s0_DAC=np.random.choice(a=range(DAC_dict['num_states']),size=1,p=DAC_dict['init_xi'])[0]
    
    if s0_DTD is None:
        s0_DTD=np.random.choice(a=range(DAC_dict['num_states']),size=1,p=DAC_dict['init_xi'])[0]
    
    DAC_dict['s0_DTD']=s0_DTD
    
    if (omega0 is None):
        omega0=[np.random.normal(size=(DAC_dict['num_states'],DAC_dict['num_actions'][m])) for m in range(DAC_dict['num_agents'])]

    DAC_dict['omega']=[0]*DAC_dict['num_agents']
    for m in range(DAC_dict['num_agents']):
        DAC_dict['omega'][m]=np.zeros((DAC_dict['num_states'],DAC_dict['num_actions'][m],T))  
        #DAC_dict['omega'][m][s,am,t]  m-th agent, t-th iteration
        DAC_dict['omega'][m][:,:,0]=omega0[m].copy()
    
    del omega0
    
    if theta0 is None:
        theta0=np.random.normal(size=DAC_dict['num_V_features'])
    DAC_dict['Theta']=np.zeros((DAC_dict['num_agents'], DAC_dict['num_V_features'], T))   
    #Theta[m,:,t]: theta vector of agent m, at t-th iteration
    
    DAC_dict['s']=np.array([s0_DAC]*(num_samples+1))
    DAC_dict['a']=np.array([[0]*DAC_dict['num_agents']]*(num_samples))  #DAC_dict['a'][i,m] is a_i of agent m
    DAC_dict['R']=np.zeros((num_samples,DAC_dict['num_agents']))        #DAC_dict['R'][i,m] is R_i of agent m
    DAC_dict['H']=np.zeros((num_samples,DAC_dict['num_agents']))        #DAC_dict['H'][i,m] is H_i of agent m
    DAC_dict['s_prime']=np.array([s0_DAC]*(num_samples+1))              #DAC_dict['s_prime'][i] means s_{i+1}'
        
    DAC_dict['Jw']=np.zeros(T)
    DAC_dict['dJ_normsq']=np.zeros(T)
    DAC_dict['Jw_cummean']=np.zeros(T)
    DAC_dict['dJ_normsq_cummean']=np.zeros(T)
    
    DAC_dict['getRvg_err_every_numiter']=getRvg_err_every_numiter
    DAC_dict['getDTD_err_every_numiter']=getDTD_err_every_numiter
    DAC_dict['getHvg_err_every_numiter']=getHvg_err_every_numiter

    if getRvg_err_every_numiter is not None:
        DAC_dict['absolute_Ravg_err']=-np.ones(T)
        DAC_dict['relative_Ravg_err']=-np.ones(T)
    if getHvg_err_every_numiter is not None:
        DAC_dict['absolute_Havg_err']=-np.ones(T)
        DAC_dict['relative_Havg_err']=-np.ones(T)
    if getDTD_err_every_numiter is not None:
        DAC_dict['absolute_DTD_err']=-np.ones(T)
        DAC_dict['relative_DTD_err']=-np.ones(T)
    
    Wpow_critic=np.linalg.matrix_power(DAC_dict['W'],Tc_prime)
    Wpow_reward=np.linalg.matrix_power(DAC_dict['W'],Tr)
    Wpow_entropy=np.linalg.matrix_power(DAC_dict['W'],Tr)

    for t in range(T-1):
        pim=get_pim([DAC_dict['omega'][m][:,:,t] for m in range(DAC_dict['num_agents'])])
        if t>0:
            theta0=DAC_dict['Theta'][:,:,t-1].copy()
        DAC_dict['Theta'][:,:,t]=DTD_h\
        (pim=pim,s0=s0_DTD,Tc=Tc,W=DAC_dict['W'],Wpow_critic=Wpow_critic,\
         Nc=Nc,beta=beta,gamma=DAC_dict['gamma'],transP=DAC_dict['transP'],\
         reward=DAC_dict['reward'],alpha_h=DAC_dict['alpha_h'],features=DAC_dict['V_features'],\
         theta0=theta0,seed_sim=seed_sim_DTD)

        if getDTD_err_every_numiter is not None:
            if t%getDTD_err_every_numiter==0:
                theta_star=get_optimal_theta_h(pim,DAC_dict['reward'],DAC_dict['alpha_h'],DAC_dict['V_features'],DAC_dict['gamma'],DAC_dict['transP'],None,None)
                DAC_dict['absolute_DTD_err'][t]=np.sum((DAC_dict['Theta'][:,:,t]-theta_star.reshape((1,-1)))**2)/DAC_dict['num_agents']
                DAC_dict['relative_DTD_err'][t]=DAC_dict['absolute_DTD_err'][t]/np.sum(theta_star**2)
        
        tN=t*N
        i_range=range(tN,tN+N)
        
        for m in range(DAC_dict['num_agents']):
            DAC_dict['omega'][m][:,:,t+1]=DAC_dict['omega'][m][:,:,t].copy()
            
        if getRvg_err_every_numiter is not None:
            if t%getRvg_err_every_numiter==0:
                R_agentavg_sum=0
                Rhat_sum=0

        if getHvg_err_every_numiter is not None:
            if t%getHvg_err_every_numiter==0:
                H_agentavg_sum=0
                Hhat_sum=0
                
        for i in i_range:
            #query samples
            s_now=DAC_dict['s'][i]
            index="[s_now"
            for m in range(DAC_dict['num_agents']):
                pp=pim[m][s_now]
                DAC_dict['a'][i,m]=np.random.choice(a=range(DAC_dict['num_actions'][m]),size=1,p=pp/pp.sum())[0]
                index+=","+str(DAC_dict['a'][i,m])
            
            pp=eval("DAC_dict['P_xi']"+index+",:]") #状态转移概率
            s_next=np.random.choice(a=range(DAC_dict['num_states']),size=1,p=pp)[0]
            DAC_dict['s'][i+1]=s_next
            
            pp=eval("DAC_dict['transP']"+index+",:]")
            s_prime=np.random.choice(a=range(DAC_dict['num_states']),size=1,p=pp)[0]
            DAC_dict['s_prime'][i]=s_prime
            
            DAC_dict['R'][i]=eval("DAC_dict['reward']"+index+",s_prime,:]") #计算奖励 R
            R_hat=1+np.random.standard_t(df=4,size=(DAC_dict['num_agents']))*(noise_std/np.sqrt(2)) #1+e_t^(m)
            R_hat=Wpow_reward.dot(DAC_dict['R'][i]*R_hat)

            for m in range(DAC_dict['num_agents']):
                DAC_dict['H'][i][m]=-tf.reduce_sum(pim[m] * tf.math.log(pim[m] + 1e-10))

            H_hat=1+np.random.standard_t(df=4,size=(DAC_dict['num_agents']))*(noise_std/np.sqrt(2))
            H_hat=Wpow_entropy.dot(DAC_dict['H'][i]*H_hat)

            if getRvg_err_every_numiter is not None:
                if t%getRvg_err_every_numiter==0:      
                    R_agentavg_sum+=DAC_dict['R'][i].mean()  
                    Rhat_sum+=R_hat

            if getHvg_err_every_numiter is not None:
                if t%getHvg_err_every_numiter==0:
                    H_agentavg_sum+=DAC_dict['H'][i].mean()
                    Hhat_sum+=H_hat

            TD_coeff=DAC_dict['gamma']*DAC_dict['V_features'][:,s_prime]\
            -DAC_dict['V_features'][:,s_now]
            
            for m in range(DAC_dict['num_agents']):
                #Actor update using AC with sample i
                psim_coeff=(R_hat[m]-alpha_h*H_hat[m]+(TD_coeff*DAC_dict['Theta'][m,:,t]).sum())*alpha/N
                psim_si=-pim[m][s_now]
                psim_si[DAC_dict['a'][i,m]]+=1
                DAC_dict['omega'][m][s_now,:,t+1]+=psim_coeff*psim_si
            
        P_xi_s2s=get_transP_s2s(pim,DAC_dict['P_xi'])
        nu_omega=stationary_dist(P_xi_s2s)
        Jw=J_h(pim,DAC_dict['transP'],DAC_dict['P_xi'],DAC_dict['reward_agentavg'],DAC_dict['alpha_h'],nu_omega)
        _, dJ_normsq=dJ_h(pim,DAC_dict['transP'],nu_omega,DAC_dict['reward_agentavg'],DAC_dict['alpha_h'],DAC_dict['gamma'])
        DAC_dict['Jw'][t]=Jw
        DAC_dict['dJ_normsq'][t]=dJ_normsq
        if t>=1:
            Jw_cummean=DAC_dict['Jw_cummean'][t-1]*t/(t+1)+Jw/(t+1)
            dJ_normsq_cummean=DAC_dict['dJ_normsq_cummean'][t-1]*t/(t+1)+dJ_normsq/(t+1)
        else:
            Jw_cummean=Jw
            dJ_normsq_cummean=dJ_normsq
            
        DAC_dict['Jw_cummean'][t]=Jw_cummean
        DAC_dict['dJ_normsq_cummean'][t]=dJ_normsq_cummean
        
        if getRvg_err_every_numiter is not None:
            if t%getRvg_err_every_numiter==0:
                R_agentavg_batchavg=R_agentavg_sum/N
                Rhat_batchavg=Rhat_sum/N
                DAC_dict['absolute_Ravg_err'][t]=((R_agentavg_batchavg-Rhat_batchavg)**2).mean()
                DAC_dict['relative_Ravg_err'][t]=DAC_dict['absolute_Ravg_err'][t]/(R_agentavg_batchavg**2)

        if getHvg_err_every_numiter is not None:
            if t%getHvg_err_every_numiter==0:
                H_agentavg_batchavg=H_agentavg_sum/N
                Hhat_batchavg=Hhat_sum/N
                DAC_dict['absolute_Havg_err'][t]=((H_agentavg_batchavg-Hhat_batchavg)**2).mean()
                DAC_dict['relative_Havg_err'][t]=DAC_dict['absolute_Havg_err'][t]/(H_agentavg_batchavg**2)
                
        if is_print:
            print("Iteration "+str(t)+", J="+str(Jw)+", ||dJ||^2="+str(dJ_normsq))
            print("J_cum_mean="+str(Jw_cummean)+", dJ_normsq_cum_mean="+str(dJ_normsq_cummean)+"\n")
            
            if getRvg_err_every_numiter is not None:
                if t%getRvg_err_every_numiter==0:
                    print("avg_{m,s,a,s_next} [R_avg(s,a,s_next)-lambda1[m]*f(s,a,s_next)]^2="+str(DAC_dict['absolute_Ravg_err'][t])+"\n")
                    print("relative R estimation error="+str(DAC_dict['relative_Ravg_err'][t])+"\n")
                else:
                    print("\n")
            else:
                print("\n")

            if getHvg_err_every_numiter is not None:
                if t%getHvg_err_every_numiter==0:
                    print("avg_{m,s,a,s_next} [H_avg(s,a,s_next)-lambda1[m]*f(s,a,s_next)]^2="+str(DAC_dict['absolute_Havg_err'][t])+"\n")
                    print("relative H estimation error="+str(DAC_dict['relative_Havg_err'][t])+"\n")
                else:
                    print("\n")
            else:
                print("\n")
                
            if getDTD_err_every_numiter is not None:
                if t%getDTD_err_every_numiter==0:
                    print("avg_{m} ||theta[m]-theta*||^2="+str(DAC_dict['absolute_DTD_err'][t])+"\n")
                    print("avg_{m} ||theta[m]-theta*||^2/||theta*||^2="+str(DAC_dict['relative_DTD_err'][t])+"\n")
                else:
                    print("\n")
            else:
                print("\n")
    
    # Get J and dJ of the final iteration
    t=T-1
    pim=get_pim([DAC_dict['omega'][m][:,:,t] for m in range(DAC_dict['num_agents'])])
    P_xi_s2s=get_transP_s2s(pim,DAC_dict['P_xi'])
    nu_omega=stationary_dist(P_xi_s2s)
    Jw=J_h(pim,DAC_dict['transP'],DAC_dict['P_xi'],DAC_dict['reward_agentavg'],DAC_dict['alpha_h'],nu_omega)
    _, dJ_normsq=dJ_h(pim,DAC_dict['transP'],nu_omega,DAC_dict['reward_agentavg'],DAC_dict['alpha_h'],DAC_dict['gamma'])
    DAC_dict['Jw'][t]=Jw
    dJ_normsq_cummean=DAC_dict['dJ_normsq_cummean'][t-1]*t/(t+1)+dJ_normsq/(t+1)
        
    DAC_dict['Jw_cummean'][t]=Jw_cummean
    DAC_dict['dJ_normsq_cummean'][t]=dJ_normsq_cummean
    DAC_dict['time(s)']=time.time()-start_time
    
    if is_print:
        print("Iteration "+str(t)+", J="+str(Jw)+", ||dJ||^2="+str(dJ_normsq))
        print("J_cum_mean="+str(Jw_cummean)+", dJ_normsq_cum_mean="+str(dJ_normsq_cummean)+"\n")
            
        if getDTD_err_every_numiter is not None:
            if t%getDTD_err_every_numiter==0:
                print("avg_{m} ||theta[m]-theta*||^2="+str(DAC_dict['absolute_DTD_err'][t])+"\n")
                print("avg_{m} ||theta[m]-theta*||^2/||theta*||^2="+str(DAC_dict['relative_DTD_err'][t])+"\n")
            else:
                print("\n")
        else:
            print("\n")
    
    if is_save:
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        
        np.save(file=save_folder+'/theta.npy',arr=DAC_dict['Theta'])
        np.save(file=save_folder+'/Jw.npy',arr=DAC_dict['Jw'])
        np.save(file=save_folder+'/dJ_normsq.npy',arr=DAC_dict['dJ_normsq'])
        np.save(file=save_folder+'/Jw_cummean.npy',arr=DAC_dict['Jw_cummean'])
        np.save(file=save_folder+'/dJ_normsq_cummean.npy',arr=DAC_dict['dJ_normsq_cummean'])
        
        for m in range(DAC_dict['num_agents']):
            np.save(file=save_folder+'/omega_agent'+str(m)+'.npy',arr=DAC_dict['omega'][m])
    
        hyp_txt=open(save_folder+'/hyperparameters.txt','a')
        keys=['T','Tc','Tc_prime','Tr','N','Nc','alpha','alpha_h','beta']
        keys+=['noise_std','seed_sim_DAC','seed_sim_DTD','is_print','is_save']
        for key in keys:
            hyp_txt.write(key+'='+str(DAC_dict[key])+'\n')
        hyp_txt.write('Time consumption: '+str(DAC_dict['time(s)']/60)+' minutes\n')
        hyp_txt.close()
    
    return DAC_dict
    
def DNAC(DAC_dict,T,Tc,Tc_prime,Tr,Tz,Nk,Nc,alpha,beta,eta,noise_std,s0_DAC=None,s0_DTD=None,omega0=None,\
         theta0=None,h0=None,seed_sim_DAC=100,seed_sim_DTD=200,is_print=False,is_save=False,\
         getRvg_err_every_numiter=None,getDTD_err_every_numiter=None,save_folder="DNAC_results/"):
    start_time=time.time()
    DAC_dict=DAC_dict.copy()
    set_seed(seed_sim_DAC)
    
    DAC_dict['T']=T
    DAC_dict['Tc']=Tc
    DAC_dict['Tc_prime']=Tc_prime
    DAC_dict['Tr']=Tr
    DAC_dict['Tz']=Tz
    DAC_dict['Nk']=Nk

    if type(Nk) is np.ndarray:
        Nk=np.ndarray.tolist(Nk.reshape(-1))
    
    if type(Nk) is list:
        K=len(Nk)
        Nk=[int(Nk[k]) for k in range(K)]
    else:
        K=1
        Nk=[int(Nk)]
    DAC_dict['K']=K
    N=int(np.round(np.sum(Nk)))
    DAC_dict['N']=N
    num_samples=T*N
    DAC_dict['Nc']=Nc
    DAC_dict['alpha']=alpha
    DAC_dict['beta']=beta
    DAC_dict['eta']=eta
    DAC_dict['noise_std']=noise_std
    DAC_dict['seed_sim_DAC']=seed_sim_DAC
    DAC_dict['seed_sim_DTD']=seed_sim_DTD
    DAC_dict['is_print']=is_print
    DAC_dict['is_save']=is_save
        
    if s0_DAC is None:
        s0_DAC=np.random.choice(a=range(DAC_dict['num_states']),size=1,p=DAC_dict['init_xi'])[0]
    
    if s0_DTD is None:
        s0_DTD=np.random.choice(a=range(DAC_dict['num_states']),size=1,p=DAC_dict['init_xi'])[0]
    
    DAC_dict['s0_DTD']=s0_DTD
    
    if omega0 is None:
        omega0=[np.random.normal(size=(DAC_dict['num_states'],DAC_dict['num_actions'][m])) for m in range(DAC_dict['num_agents'])]

    DAC_dict['omega']=[0]*DAC_dict['num_agents']
    for m in range(DAC_dict['num_agents']):
        DAC_dict['omega'][m]=np.zeros((DAC_dict['num_states'],DAC_dict['num_actions'][m],T))  
        #DAC_dict['omega'][m][s,am,t]  m-th agent, t-th iteration
        DAC_dict['omega'][m][:,:,0]=omega0[m].copy()
    
    del omega0
    
    if theta0 is None:
        theta0=np.random.normal(size=DAC_dict['num_V_features'])
    DAC_dict['Theta']=np.zeros((DAC_dict['num_agents'], DAC_dict['num_V_features'], T))   
    #Theta[m,:,t]: theta vector of agent m, at t-th iteration
    
    if h0 is None:
        h0=[np.zeros((DAC_dict['num_states'],DAC_dict['num_actions'][m])) for m in range(DAC_dict['num_agents'])]
    DAC_dict['h0']=h0.copy()
    h=h0.copy()
    del h0
    
    DAC_dict['s']=np.array([s0_DAC]*(num_samples+1))
    DAC_dict['a']=np.array([[0]*DAC_dict['num_agents']]*(num_samples))
    DAC_dict['R']=np.zeros((num_samples,DAC_dict['num_agents']))
    DAC_dict['s_prime']=np.array([s0_DAC]*(num_samples+1))   #DAC_dict['s_prime'][i] means s_{i+1}'
        
    DAC_dict['Jw']=np.zeros(T)
    DAC_dict['dJ_normsq']=np.zeros(T)
    DAC_dict['Jw_cummean']=np.zeros(T)
    DAC_dict['dJ_normsq_cummean']=np.zeros(T)
    
    DAC_dict['getRvg_err_every_numiter']=getRvg_err_every_numiter
    DAC_dict['getDTD_err_every_numiter']=getDTD_err_every_numiter
    if getRvg_err_every_numiter is not None:
        DAC_dict['absolute_Ravg_err']=-np.ones(T)
        DAC_dict['relative_Ravg_err']=-np.ones(T)
    if getDTD_err_every_numiter is not None:
        DAC_dict['absolute_DTD_err']=-np.ones(T)
        DAC_dict['relative_DTD_err']=-np.ones(T)
    
    Wpow_critic=np.linalg.matrix_power(DAC_dict['W'],Tc_prime)
    Wpow_reward=np.linalg.matrix_power(DAC_dict['W'],Tr)
    Wpow_z=np.linalg.matrix_power(DAC_dict['W'],Tz)
    
    psim_si=[0.0]*DAC_dict['num_agents']
    Mz=[0.0]*DAC_dict['num_agents']
    
    for t in range(T-1):
        pim=get_pim([DAC_dict['omega'][m][:,:,t] for m in range(DAC_dict['num_agents'])])
        if t>0:
            theta0=DAC_dict['Theta'][:,:,t-1].copy()
        
        DAC_dict['Theta'][:,:,t]=DTD\
        (pim=pim,s0=s0_DTD,Tc=Tc,W=DAC_dict['W'],Wpow_critic=Wpow_critic,\
         Nc=Nc,beta=beta,gamma=DAC_dict['gamma'],transP=DAC_dict['transP'],\
         reward=DAC_dict['reward'],features=DAC_dict['V_features'],\
         theta0=theta0,seed_sim=seed_sim_DTD)
        
        tN=t*N
        i_range=range(tN,tN+N)
        
        k=0
        batchsize_now=0
        df=[np.zeros((DAC_dict['num_states'],DAC_dict['num_actions'][m])) for m in range(DAC_dict['num_agents'])]
        
        if getRvg_err_every_numiter is not None:
            if t%getRvg_err_every_numiter==0:
                R_agentavg_sum=0
                Rhat_sum=0
        
        for i in i_range:
            #query samples
            s_now=DAC_dict['s'][i]
            index="[s_now"
            for m in range(DAC_dict['num_agents']):
                pp=pim[m][s_now]
                DAC_dict['a'][i,m]=np.random.choice(a=range(DAC_dict['num_actions'][m]),size=1,p=pp/pp.sum())[0]
                index+=","+str(DAC_dict['a'][i,m])
            
            pp=eval("DAC_dict['P_xi']"+index+",:]")
            s_next=np.random.choice(a=range(DAC_dict['num_states']),size=1,p=pp)[0]
            DAC_dict['s'][i+1]=s_next
            
            pp=eval("DAC_dict['transP']"+index+",:]")
            s_prime=np.random.choice(a=range(DAC_dict['num_states']),size=1,p=pp)[0]
            DAC_dict['s_prime'][i]=s_prime
            
            DAC_dict['R'][i]=eval("DAC_dict['reward']"+index+",s_prime,:]")
            R_hat=1+np.random.standard_t(df=4,size=(DAC_dict['num_agents']))*(noise_std/np.sqrt(2))
            R_hat=Wpow_reward.dot(DAC_dict['R'][i]*R_hat)
            
            if getRvg_err_every_numiter is not None:
                if t%getRvg_err_every_numiter==0:      
                    R_agentavg_sum+=DAC_dict['R'][i].mean()  
                    Rhat_sum+=R_hat
                
            TD_coeff=DAC_dict['gamma']*DAC_dict['V_features'][:,s_prime]\
            -DAC_dict['V_features'][:,s_now]
                
            for m in range(DAC_dict['num_agents']):
                psim_si[m]=-pim[m][s_now]         
                psim_si[m][DAC_dict['a'][i,m]]+=1
                Mz[m]=(psim_si[m]*h[m][s_now]).sum()*DAC_dict['num_agents']
            
            Mz=Wpow_z.dot(Mz)
            for m in range(DAC_dict['num_agents']):
                psim_coeff=Mz[m]-R_hat[m]-(TD_coeff*DAC_dict['Theta'][m,:,t]).sum()
                df[m][s_now]+=psim_coeff*psim_si[m]
            
            batchsize_now+=1
            if batchsize_now==Nk[k]:
                h=[h[m]-(eta/Nk[k])*df[m] for m in range(DAC_dict['num_agents'])]
                df=[np.zeros((DAC_dict['num_states'],DAC_dict['num_actions'][m])) for m in range(DAC_dict['num_agents'])]
                k+=1
                batchsize_now=0
        
        for m in range(DAC_dict['num_agents']):
            DAC_dict['omega'][m][:,:,t+1]=DAC_dict['omega'][m][:,:,t]+alpha*h[m]
        P_xi_s2s=get_transP_s2s(pim,DAC_dict['P_xi'])
        nu_omega=stationary_dist(P_xi_s2s)
        Jw=J(pim,DAC_dict['transP'],DAC_dict['P_xi'],DAC_dict['reward_agentavg'],nu_omega)
        _, dJ_normsq=dJ(pim,DAC_dict['transP'],nu_omega,DAC_dict['reward_agentavg'],DAC_dict['gamma'])
        DAC_dict['Jw'][t]=Jw
        DAC_dict['dJ_normsq'][t]=dJ_normsq
        if t>=1:
            Jw_cummean=DAC_dict['Jw_cummean'][t-1]*t/(t+1)+Jw/(t+1)
            dJ_normsq_cummean=DAC_dict['dJ_normsq_cummean'][t-1]*t/(t+1)+dJ_normsq/(t+1)
        else:
            Jw_cummean=Jw
            dJ_normsq_cummean=dJ_normsq
            
        DAC_dict['Jw_cummean'][t]=Jw_cummean
        DAC_dict['dJ_normsq_cummean'][t]=dJ_normsq_cummean
        
        if getRvg_err_every_numiter is not None:
            if t%getRvg_err_every_numiter==0:
                R_agentavg_batchavg=R_agentavg_sum/N
                Rhat_batchavg=Rhat_sum/N
                DAC_dict['absolute_Ravg_err'][t]=((R_agentavg_batchavg-Rhat_batchavg)**2).mean()
                DAC_dict['relative_Ravg_err'][t]=DAC_dict['absolute_Ravg_err'][t]/(R_agentavg_batchavg**2)
                
        if getDTD_err_every_numiter is not None:
            if t%getDTD_err_every_numiter==0:
                theta_star=get_optimal_theta(pim,DAC_dict['reward'],DAC_dict['V_features'],DAC_dict['gamma'],DAC_dict['transP'],None,None)                
                DAC_dict['absolute_DTD_err'][t]=np.sum((DAC_dict['Theta'][:,:,t]-theta_star.reshape((1,-1)))**2)/DAC_dict['num_agents']
                DAC_dict['relative_DTD_err'][t]=DAC_dict['absolute_DTD_err'][t]/np.sum(theta_star**2)
                
        if is_print:
            print("Iteration "+str(t)+", J="+str(Jw)+", ||dJ||^2="+str(dJ_normsq))
            print("J_cum_mean="+str(Jw_cummean)+", dJ_normsq_cum_mean="+str(dJ_normsq_cummean)+"\n")
            
            if getRvg_err_every_numiter is not None:
                if t%getRvg_err_every_numiter==0:
                    print("avg_{m,s,a,s_next} [R_avg(s,a,s_next)-lambda1[m]*f(s,a,s_next)]^2="+str(DAC_dict['absolute_Ravg_err'][t])+"\n")
                    print("relative R estimation error="+str(DAC_dict['relative_Ravg_err'][t])+"\n")
                else:
                    print("\n")
            else:
                print("\n")
                
            if getDTD_err_every_numiter is not None:
                if t%getDTD_err_every_numiter==0:
                    print("avg_{m} ||theta[m]-theta*||^2="+str(DAC_dict['absolute_DTD_err'][t])+"\n")
                    print("avg_{m} ||theta[m]-theta*||^2/||theta*||^2="+str(DAC_dict['relative_DTD_err'][t])+"\n")
                else:
                    print("\n")
            else:
                print("\n")
    # Get J and dJ of the final iteration
    t=T-1
    pim=get_pim([DAC_dict['omega'][m][:,:,t] for m in range(DAC_dict['num_agents'])])
    P_xi_s2s=get_transP_s2s(pim,DAC_dict['P_xi'])
    nu_omega=stationary_dist(P_xi_s2s)
    Jw=J(pim,DAC_dict['transP'],DAC_dict['P_xi'],DAC_dict['reward_agentavg'],nu_omega)
    _, dJ_normsq=dJ(pim,DAC_dict['transP'],nu_omega,DAC_dict['reward_agentavg'],DAC_dict['gamma'])
    DAC_dict['Jw'][t]=Jw
    DAC_dict['dJ_normsq'][t]=dJ_normsq
    Jw_cummean=DAC_dict['Jw_cummean'][t-1]*t/(t+1)+Jw/(t+1)
    dJ_normsq_cummean=DAC_dict['dJ_normsq_cummean'][t-1]*t/(t+1)+dJ_normsq/(t+1)
        
    DAC_dict['Jw_cummean'][t]=Jw_cummean
    DAC_dict['dJ_normsq_cummean'][t]=dJ_normsq_cummean
    DAC_dict['time(s)']=time.time()-start_time
    
    if is_print:
        print("Iteration "+str(t)+", J="+str(Jw)+", ||dJ||^2="+str(dJ_normsq))
        print("J_cum_mean="+str(Jw_cummean)+", dJ_normsq_cum_mean="+str(dJ_normsq_cummean)+"\n")
        if getDTD_err_every_numiter is not None:
            if t%getDTD_err_every_numiter==0:
                print("avg_{m} ||theta[m]-theta*||^2="+str(DAC_dict['absolute_DTD_err'][t])+"\n")
                print("avg_{m} ||theta[m]-theta*||^2/||theta*||^2="+str(DAC_dict['relative_DTD_err'][t])+"\n")
            else:
                print("\n")
        else:
            print("\n")
    
    if is_save:
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        
        np.save(file=save_folder+'/theta.npy',arr=DAC_dict['Theta'])
        np.save(file=save_folder+'/Jw.npy',arr=DAC_dict['Jw'])
        np.save(file=save_folder+'/dJ_normsq.npy',arr=DAC_dict['dJ_normsq'])
        np.save(file=save_folder+'/Jw_cummean.npy',arr=DAC_dict['Jw_cummean'])
        np.save(file=save_folder+'/dJ_normsq_cummean.npy',arr=DAC_dict['dJ_normsq_cummean'])
        
        for m in range(DAC_dict['num_agents']):
            np.save(file=save_folder+'/omega_agent'+str(m)+'.npy',arr=DAC_dict['omega'][m])
    
        hyp_txt=open(save_folder+'/hyperparameters.txt','a')
        keys=['T','Tc','Tc_prime','Tr','N','Nc','alpha','beta']
        keys+=['noise_std','seed_sim_DAC','seed_sim_DTD','is_print','is_save']
        for key in keys:
            hyp_txt.write(key+'='+str(DAC_dict[key])+'\n')
        hyp_txt.write('Time consumption: '+str(DAC_dict['time(s)']/60)+' minutes\n')
        hyp_txt.close()
    
    return DAC_dict

def DAC_Kaiqing_alg2(DAC_dict,T,v0=None,lambda0=None,omega0=None,theta0=None,s0_DTD=None,s0_DAC=None,seed=None,beta_v=0.3,beta_theta=0.1,\
                     is_print=True,is_save=True,save_folder="Kaiqing_alg2_results/",getRvg_err_every_numiter=None,getDTD_err_every_numiter=None,is_exact_Ravg=False):
    start_time=time.time()
    DAC_dict=DAC_dict.copy()
    set_seed(seed)
    DAC_dict['seed']=seed
        
    if v0 is None:
        v0=np.random.normal(size=(DAC_dict['num_agents'],DAC_dict['num_V_features']))
    if lambda0 is None:
        lambda0=np.random.normal(size=(DAC_dict['num_agents'],DAC_dict['num_R_features']))
    
    DAC_dict['v0']=v0.copy()
    v=v0.copy()
    DAC_dict['lambda0']=lambda0.copy()
    lambda1=lambda0.copy()
    
    v_tilde=np.zeros((DAC_dict['num_agents'],DAC_dict['num_V_features']))
    lambda1_tilde=np.zeros((DAC_dict['num_agents'],DAC_dict['num_R_features']))
    
    DAC_dict['T']=T
    DAC_dict['is_print']=is_print
    DAC_dict['is_save']=is_save
    DAC_dict['getRvg_err_every_numiter']=getRvg_err_every_numiter
    DAC_dict['getDTD_err_every_numiter']=getDTD_err_every_numiter
    DAC_dict['is_exact_Ravg']=is_exact_Ravg
    
    if s0_DTD is None:
        s0_DTD=np.random.choice(a=range(DAC_dict['num_states']),size=1,p=DAC_dict['init_xi'])[0]
    if s0_DAC is None:
        s0_DAC=np.random.choice(a=range(DAC_dict['num_states']),size=1,p=DAC_dict['init_xi'])[0]
    
    if omega0 is None:
        omega0=[np.random.normal(size=(DAC_dict['num_states'],DAC_dict['num_actions'][m])) for m in range(DAC_dict['num_agents'])]

    DAC_dict['omega']=[0]*DAC_dict['num_agents']
    for m in range(DAC_dict['num_agents']):
        DAC_dict['omega'][m]=np.zeros((DAC_dict['num_states'],DAC_dict['num_actions'][m],T))  
        #DAC_dict['omega'][m][s,am,t]  m-th agent, t-th iteration
        DAC_dict['omega'][m][:,:,0]=omega0[m].copy()
    
    del omega0
    
    DAC_dict['s_DTD']=np.array([s0_DTD]*(T+1))
    DAC_dict['a_DTD']=np.array([[0]*DAC_dict['num_agents']]*T)
    DAC_dict['s_DAC']=np.array([s0_DAC]*(T+1))
    DAC_dict['a_DAC']=np.array([[0]*DAC_dict['num_agents']]*T)
    DAC_dict['s_prime_DAC']=np.array([-1]*T)
    DAC_dict['R_DTD']=np.zeros((T,DAC_dict['num_agents']))
    
    DAC_dict['Jw']=np.zeros(T)
    DAC_dict['dJ_normsq']=np.zeros(T)
    DAC_dict['Jw_cummean']=np.zeros(T)
    DAC_dict['dJ_normsq_cummean']=np.zeros(T)
    if ((getRvg_err_every_numiter is not None) and (not is_exact_Ravg)):
        DAC_dict['absolute_Ravg_err']=-np.ones(T)
        DAC_dict['relative_Ravg_err']=-np.ones(T)
    if getDTD_err_every_numiter is not None:
        DAC_dict['absolute_DTD_err']=-np.ones(T)
        DAC_dict['relative_DTD_err']=-np.ones(T)
    
    if type(beta_v) is not types.FunctionType:
        beta_vt=beta_v
    DAC_dict['beta_v']=beta_v
    
    if type(beta_theta) is not types.FunctionType:
        beta_thetat=beta_theta
    DAC_dict['beta_theta']=beta_theta
        
    for t in range(T-1):
        pim=get_pim([DAC_dict['omega'][m][:,:,t] for m in range(DAC_dict['num_agents'])])
        
        if type(beta_v) is types.FunctionType:
            beta_vt=beta_v(t)
        
        if type(beta_theta) is types.FunctionType:
            beta_thetat=beta_theta(t)
        
        #query samples
        s_now_DTD=DAC_dict['s_DTD'][t]
        s_now_DAC=DAC_dict['s_DAC'][t]
        index_DTD="s_now_DTD"
        index_DAC="s_now_DAC"
        for m in range(DAC_dict['num_agents']):
            pp=pim[m][s_now_DTD]
            DAC_dict['a_DTD'][t,m]=np.random.choice(a=range(DAC_dict['num_actions'][m]),size=1,p=pp/pp.sum())[0]
            pp=pim[m][s_now_DAC]
            DAC_dict['a_DAC'][t,m]=np.random.choice(a=range(DAC_dict['num_actions'][m]),size=1,p=pp/pp.sum())[0]
            index_DTD+=","+str(DAC_dict['a_DTD'][t,m])
            index_DAC+=","+str(DAC_dict['a_DAC'][t,m])
        
        pp=eval("DAC_dict['transP']["+index_DTD+",:]")  
        s_next_DTD=np.random.choice(a=range(DAC_dict['num_states']),size=1,p=pp)[0]
        DAC_dict['s_DTD'][t+1]=s_next_DTD
        pp=eval("DAC_dict['P_xi']["+index_DAC+",:]")
        s_next_DAC=np.random.choice(a=range(DAC_dict['num_states']),size=1,p=pp)[0]
        DAC_dict['s_DAC'][t+1]=s_next_DAC
        pp=eval("DAC_dict['transP']["+index_DAC+",:]")
        s_prime_DAC=np.random.choice(a=range(DAC_dict['num_states']),size=1,p=pp)[0]
        DAC_dict['s_prime_DAC'][t]=s_prime_DAC
        
        Rnow_DTD=eval("DAC_dict['reward']["+index_DTD+",s_next_DTD,:]")
        DAC_dict['R_DTD'][t]=Rnow_DTD
        
        R_features_DTD=eval("DAC_dict['R_features'][:,"+index_DTD+",s_next_DTD]")
        R_features_DAC=eval("DAC_dict['R_features'][:,"+index_DAC+",s_prime_DAC]") 
        
        if is_exact_Ravg:
            Rnow_DAC=eval("DAC_dict['reward']["+index_DAC+",s_next_DAC,:]")  
            
        for m in range(DAC_dict['num_agents']):
            if is_exact_Ravg:
                R_hat=np.mean(Rnow_DTD) 
            else:
                R_hat=np.sum(R_features_DTD*lambda1[m])   
            lambda1_tilde[m]=lambda1[m]+beta_vt*(Rnow_DTD[m]-R_hat)*R_features_DTD
            # mu_tilde[m]=(1-beta_vt)*mu[m]+beta_vt*Rnow[m]
            delta=Rnow_DTD[m]+np.sum\
                (v[m]*(DAC_dict['gamma']*DAC_dict['V_features'][:,s_next_DTD]-DAC_dict['V_features'][:,s_now_DTD]))
            v_tilde[m]=v[m]+beta_vt*delta*DAC_dict['V_features'][:,s_now_DTD]   #Critic step
            if is_exact_Ravg:
                R_hat=np.mean(Rnow_DAC)  
            else:  
                R_hat=np.sum(R_features_DAC*lambda1[m])   
            delta=R_hat+np.sum\
                (v[m]*(DAC_dict['gamma']*DAC_dict['V_features'][:,s_next_DAC]-DAC_dict['V_features'][:,s_now_DAC]))
            psi=-pim[m][s_now_DAC]         
            psi[DAC_dict['a_DAC'][t,m]]+=1
            DAC_dict['omega'][m][:,:,t+1]=DAC_dict['omega'][m][:,:,t].copy()
            DAC_dict['omega'][m][s_now_DAC,:,t+1]=DAC_dict['omega'][m][s_now_DAC,:,t]+beta_thetat*delta*psi   #Actor step
            
        
        # mu=DAC_dict['W'].dot(mu_tilde)
        lambda1=DAC_dict['W'].dot(lambda1_tilde)
        v=DAC_dict['W'].dot(v_tilde)
        
        P_xi_s2s=get_transP_s2s(pim,DAC_dict['P_xi'])
        nu_omega=stationary_dist(P_xi_s2s)
        Jw=J(pim,DAC_dict['transP'],DAC_dict['P_xi'],DAC_dict['reward_agentavg'],nu_omega)
        _, dJ_normsq=dJ(pim,DAC_dict['transP'],nu_omega,DAC_dict['reward_agentavg'],DAC_dict['gamma'])
        DAC_dict['Jw'][t]=Jw
        DAC_dict['dJ_normsq'][t]=dJ_normsq
        if t>=1:
            Jw_cummean=DAC_dict['Jw_cummean'][t-1]*t/(t+1)+Jw/(t+1)
            dJ_normsq_cummean=DAC_dict['dJ_normsq_cummean'][t-1]*t/(t+1)+dJ_normsq/(t+1)
        else:
            Jw_cummean=Jw
            dJ_normsq_cummean=dJ_normsq
            
        DAC_dict['Jw_cummean'][t]=Jw_cummean
        DAC_dict['dJ_normsq_cummean'][t]=dJ_normsq_cummean
        if ((getRvg_err_every_numiter is not None) and (not is_exact_Ravg)):
            if t%getRvg_err_every_numiter==0:
                tmp=lambda1.reshape(lambda1.shape+(1,)*(DAC_dict['num_agents']+2))*DAC_dict['R_features'].reshape((1,)+DAC_dict['R_features'].shape)
                tmp=((tmp.sum(axis=1)-DAC_dict['reward_agentavg'])**2).mean()
                DAC_dict['absolute_Ravg_err'][t]=tmp
                DAC_dict['relative_Ravg_err'][t]=tmp/np.mean(DAC_dict['reward_agentavg']**2)
        if getDTD_err_every_numiter is not None:
            if t%getDTD_err_every_numiter==0:
                v_star=get_optimal_theta(pim,DAC_dict['reward'],DAC_dict['V_features'],DAC_dict['gamma'],DAC_dict['transP'],None,None)                
                DAC_dict['absolute_DTD_err'][t]=np.sum((v-v_star.reshape((1,-1)))**2)/DAC_dict['num_agents']
                DAC_dict['relative_DTD_err'][t]=DAC_dict['absolute_DTD_err'][t]/np.sum(v_star**2)

        if is_print:
            print("Iteration "+str(t)+", J="+str(Jw)+", ||dJ||^2="+str(dJ_normsq))
            print("J_cum_mean="+str(Jw_cummean)+", dJ_normsq_cum_mean="+str(dJ_normsq_cummean))
            if ((getRvg_err_every_numiter is not None) and (not is_exact_Ravg)):
                if t%getRvg_err_every_numiter==0:
                    print("avg_{m,s,a,s_next} [R_avg(s,a,s_next)-lambda1[m]*f(s,a,s_next)]^2="+str(DAC_dict['absolute_Ravg_err'][t])+"\n")
                    print("relative R estimation error="+str(DAC_dict['relative_Ravg_err'][t])+"\n")
                
            if getDTD_err_every_numiter is not None:
                if t%getDTD_err_every_numiter==0:
                    print("avg_{m} ||v[m]-v*||^2="+str(DAC_dict['absolute_DTD_err'][t])+"\n")
                    print("avg_{m} ||v[m]-v*||^2/||v*||^2="+str(DAC_dict['relative_DTD_err'][t])+"\n")
            print()
            
    # Get J and dJ of the final iteration
    t=T-1
    pim=get_pim([DAC_dict['omega'][m][:,:,t] for m in range(DAC_dict['num_agents'])])
    P_xi_s2s=get_transP_s2s(pim,DAC_dict['P_xi'])
    nu_omega=stationary_dist(P_xi_s2s)
    Jw=J(pim,DAC_dict['transP'],DAC_dict['P_xi'],DAC_dict['reward_agentavg'],nu_omega)
    _, dJ_normsq=dJ(pim,DAC_dict['transP'],nu_omega,DAC_dict['reward_agentavg'],DAC_dict['gamma'])
    DAC_dict['Jw'][t]=Jw
    DAC_dict['dJ_normsq'][t]=dJ_normsq
    Jw_cummean=DAC_dict['Jw_cummean'][t-1]*t/(t+1)+Jw/(t+1)
    dJ_normsq_cummean=DAC_dict['dJ_normsq_cummean'][t-1]*t/(t+1)+dJ_normsq/(t+1)
        
    DAC_dict['Jw_cummean'][t]=Jw_cummean
    DAC_dict['dJ_normsq_cummean'][t]=dJ_normsq_cummean
    if ((getRvg_err_every_numiter is not None) and (not is_exact_Ravg)):
        if t%getRvg_err_every_numiter==0:
            tmp=lambda1.reshape(lambda1.shape+(1,)*(DAC_dict['num_agents']+2))*DAC_dict['R_features'].reshape((1,)+DAC_dict['R_features'].shape)
            tmp=((tmp.sum(axis=1)-DAC_dict['reward_agentavg'])**2).mean()
            DAC_dict['absolute_Ravg_err'][t]=tmp
    if getDTD_err_every_numiter is not None:
        if t%getDTD_err_every_numiter==0:
            v_star=get_optimal_theta(pim,DAC_dict['reward'],DAC_dict['V_features'],DAC_dict['gamma'],DAC_dict['transP'],None,None)                
            DAC_dict['absolute_DTD_err'][t]=np.sum((v-v_star.reshape((1,-1)))**2)/DAC_dict['num_agents']

    if is_print:
        print("Iteration "+str(t)+", J="+str(Jw)+", ||dJ||^2="+str(dJ_normsq))
        print("J_cum_mean="+str(Jw_cummean)+", dJ_normsq_cum_mean="+str(dJ_normsq_cummean))
        if ((getRvg_err_every_numiter is not None) and (not is_exact_Ravg)):
            if t%getRvg_err_every_numiter==0:
                print("avg_{m,s,a,s_next} [R_avg(s,a,s_next)-lambda1[m]*f(s,a,s_next)]^2="+str(DAC_dict['absolute_Ravg_err'][t])+"\n")
                print("relative R estimation error="+str(DAC_dict['relative_Ravg_err'][t])+"\n")
        print()
    
    DAC_dict['time(s)']=time.time()-start_time
    if is_save:
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        
        np.save(file=save_folder+'/Jw.npy',arr=DAC_dict['Jw'])
        np.save(file=save_folder+'/dJ_normsq.npy',arr=DAC_dict['dJ_normsq'])
        np.save(file=save_folder+'/Jw_cummean.npy',arr=DAC_dict['Jw_cummean'])
        np.save(file=save_folder+'/dJ_normsq_cummean.npy',arr=DAC_dict['dJ_normsq_cummean'])
        if ((getRvg_err_every_numiter is not None) and (not is_exact_Ravg)):
            np.save(file=save_folder+'/relative_Ravg_err.npy',arr=DAC_dict['relative_Ravg_err'])
            np.save(file=save_folder+'/absolute_Ravg_err.npy',arr=DAC_dict['absolute_Ravg_err'])
        if getDTD_err_every_numiter is not None:
            np.save(file=save_folder+'/relative_DTD_err.npy',arr=DAC_dict['relative_DTD_err'])
            np.save(file=save_folder+'/absolute_DTD_err.npy',arr=DAC_dict['absolute_DTD_err'])
        
        for m in range(DAC_dict['num_agents']):
            np.save(file=save_folder+'/omega_agent'+str(m)+'.npy',arr=DAC_dict['omega'][m])
    
        hyp_txt=open(save_folder+'/hyperparameters.txt','a')
                
        keys=['T','v0','lambda0','s0_DTD','s0_DAC','seed','is_exact_Ravg','getRvg_err_every_numiter','getDTD_err_every_numiter']
        for key in keys:
            if key in DAC_dict.keys():
                hyp_txt.write(key+'='+str(DAC_dict[key])+'\n')
        
        keys=['beta_v','beta_theta']
        for key in keys:
            if type(DAC_dict[key]) is not types.FunctionType:
                hyp_txt.write(key+'='+str(DAC_dict[key])+'\n')
            else:
                hyp_txt.write(key+': Function \n')
                
        hyp_txt.write('Time consumption: '+str(DAC_dict['time(s)']/60)+' minutes\n')
        hyp_txt.close()
    
    return DAC_dict

def DAC_Kaiqing_alg2_minibatch(DAC_dict,T,N_DTD=1,N_DAC=1,v0=None,lambda0=None,omega0=None,theta0=None,s0_DTD=None,s0_DAC=None,\
                               seed=None,beta_v=0.3,beta_theta=0.1,is_print=True,is_save=True,save_folder="Kaiqing_alg2_results/",\
                               getRvg_err_every_numiter=None,getDTD_err_every_numiter=None,is_exact_Ravg=False):
    start_time=time.time()
    DAC_dict=DAC_dict.copy()
    set_seed(seed)
    DAC_dict['seed']=seed
        
    if v0 is None:
        v0=np.random.normal(size=(DAC_dict['num_agents'],DAC_dict['num_V_features']))
    if lambda0 is None:
        lambda0=np.random.normal(size=(DAC_dict['num_agents'],DAC_dict['num_R_features']))
    
    DAC_dict['v0']=v0.copy()
    v=v0.copy()
    DAC_dict['lambda0']=lambda0.copy()
    lambda1=lambda0.copy()
    
    v_tilde=np.zeros((DAC_dict['num_agents'],DAC_dict['num_V_features']))
    lambda1_tilde=np.zeros((DAC_dict['num_agents'],DAC_dict['num_R_features']))
    
    DAC_dict['T']=T
    DAC_dict['N_DTD']=N_DTD
    DAC_dict['N_DAC']=N_DAC
    DAC_dict['is_print']=is_print
    DAC_dict['is_save']=is_save
    DAC_dict['getRvg_err_every_numiter']=getRvg_err_every_numiter
    DAC_dict['getDTD_err_every_numiter']=getDTD_err_every_numiter
    DAC_dict['is_exact_Ravg']=is_exact_Ravg
    
    if s0_DTD is None:
        s0_DTD=np.random.choice(a=range(DAC_dict['num_states']),size=1,p=DAC_dict['init_xi'])[0]
    if s0_DAC is None:
        s0_DAC=np.random.choice(a=range(DAC_dict['num_states']),size=1,p=DAC_dict['init_xi'])[0]
    
    if omega0 is None:
        omega0=[np.random.normal(size=(DAC_dict['num_states'],DAC_dict['num_actions'][m])) for m in range(DAC_dict['num_agents'])]

    DAC_dict['omega']=[0]*DAC_dict['num_agents']
    for m in range(DAC_dict['num_agents']):
        DAC_dict['omega'][m]=np.zeros((DAC_dict['num_states'],DAC_dict['num_actions'][m],T))  
        #DAC_dict['omega'][m][s,am,t]  m-th agent, t-th iteration
        DAC_dict['omega'][m][:,:,0]=omega0[m].copy()
    
    del omega0
    
    DAC_dict['Jw']=np.zeros(T)
    DAC_dict['dJ_normsq']=np.zeros(T)
    DAC_dict['Jw_cummean']=np.zeros(T)
    DAC_dict['dJ_normsq_cummean']=np.zeros(T)
    if ((getRvg_err_every_numiter is not None) and (not is_exact_Ravg)):
        DAC_dict['absolute_Ravg_err']=-np.ones(T)
        DAC_dict['relative_Ravg_err']=-np.ones(T)
    if getDTD_err_every_numiter is not None:
        DAC_dict['absolute_DTD_err']=-np.ones(T)
        DAC_dict['relative_DTD_err']=-np.ones(T)
    
    if type(beta_v) is not types.FunctionType:
        beta_vt=beta_v
    DAC_dict['beta_v']=beta_v
    
    if type(beta_theta) is not types.FunctionType:
        beta_thetat=beta_theta
    DAC_dict['beta_theta']=beta_theta
    
    s_next_DTD=s0_DTD
    a_now_DTD=np.array([0]*DAC_dict['num_agents'])
    s_next_DAC=s0_DAC
    a_now_DAC=np.array([0]*DAC_dict['num_agents'])
            
    for t in range(T-1):
        pim=get_pim([DAC_dict['omega'][m][:,:,t] for m in range(DAC_dict['num_agents'])])
        
        if type(beta_v) is types.FunctionType:
            beta_vt=beta_v(t)
        
        if type(beta_theta) is types.FunctionType:
            beta_thetat=beta_theta(t)
        
        dlambda1=np.zeros_like(lambda1)
        dv_tilde=np.zeros_like(v_tilde)
        dpolicy=[np.zeros_like(pim[m]) for m in range(DAC_dict['num_agents'])]
        
        for i in range(N_DTD):
            s_now_DTD=s_next_DTD            
            index_DTD="s_now_DTD"
            for m in range(DAC_dict['num_agents']):
                pp=pim[m][s_now_DTD]
                a_now_DTD[m]=np.random.choice(a=range(DAC_dict['num_actions'][m]),size=1,p=pp/pp.sum())[0]
                index_DTD+=","+str(a_now_DTD[m])
            
            pp=eval("DAC_dict['transP']["+index_DTD+",:]")  
            s_next_DTD=np.random.choice(a=range(DAC_dict['num_states']),size=1,p=pp)[0]
            
            Rnow_DTD=eval("DAC_dict['reward']["+index_DTD+",s_next_DTD,:]")
            
            R_features_DTD=eval("DAC_dict['R_features'][:,"+index_DTD+",s_next_DTD]")
            
            for m in range(DAC_dict['num_agents']):
                if is_exact_Ravg:
                    R_hat=np.mean(Rnow_DTD) 
                else:
                    R_hat=np.sum(R_features_DTD*lambda1[m])   
                dlambda1[m]+=(Rnow_DTD[m]-R_hat)*R_features_DTD
                delta=Rnow_DTD[m]+np.sum\
                    (v[m]*(DAC_dict['gamma']*DAC_dict['V_features'][:,s_next_DTD]-DAC_dict['V_features'][:,s_now_DTD]))
                dv_tilde[m]+=delta*DAC_dict['V_features'][:,s_now_DTD]   #Critic step
        
        for i in range(N_DAC):
            s_now_DAC=s_next_DAC
            index_DAC="s_now_DAC"
            for m in range(DAC_dict['num_agents']):
                pp=pim[m][s_now_DAC]
                a_now_DAC[m]=np.random.choice(a=range(DAC_dict['num_actions'][m]),size=1,p=pp/pp.sum())[0]
                index_DAC+=","+str(a_now_DAC[m])
            
            pp=eval("DAC_dict['P_xi']["+index_DAC+",:]")
            s_next_DAC=np.random.choice(a=range(DAC_dict['num_states']),size=1,p=pp)[0]
            pp=eval("DAC_dict['transP']["+index_DAC+",:]")
            s_prime_DAC=np.random.choice(a=range(DAC_dict['num_states']),size=1,p=pp)[0]
            
            R_features_DAC=eval("DAC_dict['R_features'][:,"+index_DAC+",s_prime_DAC]")
        
            if is_exact_Ravg:
                Rnow_DAC=eval("DAC_dict['reward']["+index_DAC+",s_next_DAC,:]")  
            
            for m in range(DAC_dict['num_agents']):
                if is_exact_Ravg:
                    R_hat=np.mean(Rnow_DAC)  
                else:
                    R_hat=np.sum(R_features_DAC*lambda1[m])   
                delta=R_hat+np.sum\
                    (v[m]*(DAC_dict['gamma']*DAC_dict['V_features'][:,s_next_DAC]-DAC_dict['V_features'][:,s_now_DAC]))
                psi=-pim[m][s_now_DAC]         
                psi[a_now_DAC[m]]+=1
                dpolicy[m][s_now_DAC,:]+=delta*psi
                
        for m in range(DAC_dict['num_agents']):
            DAC_dict['omega'][m][:,:,t+1]=DAC_dict['omega'][m][:,:,t]+(beta_thetat/N_DAC)*dpolicy[m]   #Actor step
        v_tilde=v+(beta_vt/N_DTD)*dv_tilde
        lambda1_tilde=lambda1+(beta_vt/N_DTD)*dlambda1
        
        lambda1=DAC_dict['W'].dot(lambda1_tilde)
        v=DAC_dict['W'].dot(v_tilde)
        
        P_xi_s2s=get_transP_s2s(pim,DAC_dict['P_xi'])
        nu_omega=stationary_dist(P_xi_s2s)
        Jw=J(pim,DAC_dict['transP'],DAC_dict['P_xi'],DAC_dict['reward_agentavg'],nu_omega)
        _, dJ_normsq=dJ(pim,DAC_dict['transP'],nu_omega,DAC_dict['reward_agentavg'],DAC_dict['gamma'])
        DAC_dict['Jw'][t]=Jw
        DAC_dict['dJ_normsq'][t]=dJ_normsq
        if t>=1:
            Jw_cummean=DAC_dict['Jw_cummean'][t-1]*t/(t+1)+Jw/(t+1)
            dJ_normsq_cummean=DAC_dict['dJ_normsq_cummean'][t-1]*t/(t+1)+dJ_normsq/(t+1)
        else:
            Jw_cummean=Jw
            dJ_normsq_cummean=dJ_normsq
            
        DAC_dict['Jw_cummean'][t]=Jw_cummean
        DAC_dict['dJ_normsq_cummean'][t]=dJ_normsq_cummean
        
        if ((getRvg_err_every_numiter is not None) and (not is_exact_Ravg)):
            if t%getRvg_err_every_numiter==0:
                tmp=lambda1.reshape(lambda1.shape+(1,)*(DAC_dict['num_agents']+2))*DAC_dict['R_features'].reshape((1,)+DAC_dict['R_features'].shape)
                tmp=((tmp.sum(axis=1)-DAC_dict['reward_agentavg'])**2).mean()
                DAC_dict['absolute_Ravg_err'][t]=tmp
                DAC_dict['relative_Ravg_err'][t]=tmp/np.mean(DAC_dict['reward_agentavg']**2)
        if getDTD_err_every_numiter is not None:
            if t%getDTD_err_every_numiter==0:
                v_star=get_optimal_theta(pim,DAC_dict['reward'],DAC_dict['V_features'],DAC_dict['gamma'],DAC_dict['transP'],None,None)                
                DAC_dict['absolute_DTD_err'][t]=np.sum((v-v_star.reshape((1,-1)))**2)/DAC_dict['num_agents']
                DAC_dict['relative_DTD_err'][t]=DAC_dict['absolute_DTD_err'][t]/np.sum(v_star**2)

        if is_print:
            print("Iteration "+str(t)+", J="+str(Jw)+", ||dJ||^2="+str(dJ_normsq))
            print("J_cum_mean="+str(Jw_cummean)+", dJ_normsq_cum_mean="+str(dJ_normsq_cummean))            
            if ((getRvg_err_every_numiter is not None) and (not is_exact_Ravg)):
                if t%getRvg_err_every_numiter==0:
                    print("avg_{m,s,a,s_next} [R_avg(s,a,s_next)-lambda1[m]*f(s,a,s_next)]^2="+str(DAC_dict['absolute_Ravg_err'][t]))
                    print("relative R estimation error="+str(DAC_dict['relative_Ravg_err'][t]))
                
            if getDTD_err_every_numiter is not None:
                if t%getDTD_err_every_numiter==0:
                    print("avg_{m} ||v[m]-v*||^2="+str(DAC_dict['absolute_DTD_err'][t]))
                    print("avg_{m} ||v[m]-v*||^2/||v*||^2="+str(DAC_dict['relative_DTD_err'][t]))
            print()
            
    # Get J and dJ of the final iteration
    t=T-1
    pim=get_pim([DAC_dict['omega'][m][:,:,t] for m in range(DAC_dict['num_agents'])])
    P_xi_s2s=get_transP_s2s(pim,DAC_dict['P_xi'])
    nu_omega=stationary_dist(P_xi_s2s)
    Jw=J(pim,DAC_dict['transP'],DAC_dict['P_xi'],DAC_dict['reward_agentavg'],nu_omega)
    _, dJ_normsq=dJ(pim,DAC_dict['transP'],nu_omega,DAC_dict['reward_agentavg'],DAC_dict['gamma'])
    DAC_dict['Jw'][t]=Jw
    DAC_dict['dJ_normsq'][t]=dJ_normsq
    Jw_cummean=DAC_dict['Jw_cummean'][t-1]*t/(t+1)+Jw/(t+1)
    dJ_normsq_cummean=DAC_dict['dJ_normsq_cummean'][t-1]*t/(t+1)+dJ_normsq/(t+1)
        
    DAC_dict['Jw_cummean'][t]=Jw_cummean
    DAC_dict['dJ_normsq_cummean'][t]=dJ_normsq_cummean
    if ((getRvg_err_every_numiter is not None) and (not is_exact_Ravg)):
        if t%getRvg_err_every_numiter==0:
            tmp=lambda1.reshape(lambda1.shape+(1,)*(DAC_dict['num_agents']+2))*DAC_dict['R_features'].reshape((1,)+DAC_dict['R_features'].shape)
            tmp=((tmp.sum(axis=1)-DAC_dict['reward_agentavg'])**2).mean()
            DAC_dict['absolute_Ravg_err'][t]=tmp
    if getDTD_err_every_numiter is not None:
        if t%getDTD_err_every_numiter==0:
            v_star=get_optimal_theta(pim,DAC_dict['reward'],DAC_dict['V_features'],DAC_dict['gamma'],DAC_dict['transP'],None,None)                
            DAC_dict['absolute_DTD_err'][t]=np.sum((v-v_star.reshape((1,-1)))**2)/DAC_dict['num_agents']

    if is_print:
        print("Iteration "+str(t)+", J="+str(Jw)+", ||dJ||^2="+str(dJ_normsq))
        print("J_cum_mean="+str(Jw_cummean)+", dJ_normsq_cum_mean="+str(dJ_normsq_cummean))
        if ((getRvg_err_every_numiter is not None) and (not is_exact_Ravg)):
            if t%getRvg_err_every_numiter==0:
                print("avg_{m,s,a,s_next} [R_avg(s,a,s_next)-lambda1[m]*f(s,a,s_next)]^2="+str(DAC_dict['absolute_Ravg_err'][t]))
                print("relative R estimation error="+str(DAC_dict['relative_Ravg_err'][t]))
        if getDTD_err_every_numiter is not None:
            if t%getDTD_err_every_numiter==0:
                print("avg_{m} ||v[m]-v*||^2="+str(DAC_dict['absolute_DTD_err'][t]))
                print("avg_{m} ||v[m]-v*||^2/||v*||^2="+str(DAC_dict['relative_DTD_err'][t]))
        print()
    
    DAC_dict['time(s)']=time.time()-start_time
    if is_save:
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        
        np.save(file=save_folder+'/Jw.npy',arr=DAC_dict['Jw'])
        np.save(file=save_folder+'/dJ_normsq.npy',arr=DAC_dict['dJ_normsq'])
        np.save(file=save_folder+'/Jw_cummean.npy',arr=DAC_dict['Jw_cummean'])
        np.save(file=save_folder+'/dJ_normsq_cummean.npy',arr=DAC_dict['dJ_normsq_cummean'])
        if ((getRvg_err_every_numiter is not None) and (not is_exact_Ravg)):
            np.save(file=save_folder+'/relative_Ravg_err.npy',arr=DAC_dict['relative_Ravg_err'])
            np.save(file=save_folder+'/absolute_Ravg_err.npy',arr=DAC_dict['absolute_Ravg_err'])
        if getDTD_err_every_numiter is not None:
            np.save(file=save_folder+'/relative_DTD_err.npy',arr=DAC_dict['relative_DTD_err'])
            np.save(file=save_folder+'/absolute_DTD_err.npy',arr=DAC_dict['absolute_DTD_err'])
        
        for m in range(DAC_dict['num_agents']):
            np.save(file=save_folder+'/omega_agent'+str(m)+'.npy',arr=DAC_dict['omega'][m])
    
        hyp_txt=open(save_folder+'/hyperparameters.txt','a')
                
        keys=['T','v0','N_DTD','N_DAC','lambda0','s0_DTD','s0_DAC','seed','is_exact_Ravg','getRvg_err_every_numiter','getDTD_err_every_numiter']
        for key in keys:
            if key in DAC_dict.keys():
                hyp_txt.write(key+'='+str(DAC_dict[key])+'\n')
        
        keys=['beta_v','beta_theta']
        for key in keys:
            if type(DAC_dict[key]) is not types.FunctionType:
                hyp_txt.write(key+'='+str(DAC_dict[key])+'\n')
            else:
                hyp_txt.write(key+': Function \n')
                
        hyp_txt.write('Time consumption: '+str(DAC_dict['time(s)']/60)+' minutes\n')
        hyp_txt.close()
    
    return DAC_dict

def runs(expr_num,num_agents,DAC_dict,hyps,is_print=True,is_save_JdJ=True,folder='Figures'):
    hyps=hyps.copy()
    
    err_types=['Jw','dJ_normsq','Jw_cummean','dJ_normsq_cummean']
    err_types+=['absolute_Ravg_err','absolute_DTD_err','relative_Ravg_err','relative_DTD_err']
    hyps_num=len(hyps)
    results=[0]*hyps_num
    
    if folder is not None:   #Then begin to save figure
        if not os.path.isdir(folder):
            os.makedirs(folder)
    for hyp_index in range(hyps_num):
        start_time=time.time()
        hyp=hyps[hyp_index].copy()
        results[hyp_index]={}

        if is_print:
            print('Begin hyperparameter '+str(hyp_index)+':\n')
        
        hyp_txt=open(folder+'/hyperparameters.txt','a')
        
        keys=['alg','T','is_print','is_save','data_folder','color','marker','legend']        
        if hyp['alg']=='AC':            
            keys+=['Tc','Tc_prime','Tr','N','Nc','alpha','beta']
            keys+=['noise_std','seed_sim_DAC','seed_sim_DTD','getDTD_err_every_numiter']
        elif hyp['alg']=='ACH':
            keys+=['Tc','Tc_prime','Tr','N','Nc','alpha','alpha_h','beta']
            keys+=['noise_std','seed_sim_DAC','seed_sim_DTD','getDTD_err_every_numiter']
        elif hyp['alg']=='NAC':
            keys+=['Tc','Tc_prime','Tr','N','Nc','alpha','beta']
            keys+=['noise_std','seed_sim_DAC','seed_sim_DTD']
            keys+=['eta','Nk','K','Tz','getDTD_err_every_numiter']
            
            if type(hyp['Nk']) is list:
                hyp['K']=len(hyp['Nk'])
                hyp['N']=np.sum(hyp['Nk'])
            elif type(hyp['Nk']) is np.ndarray:
                hyp['Nk']=np.ndarray.tolist(hyp['Nk'])
                hyp['K']=len(hyp['Nk'])
                hyp['N']=np.sum(hyp['Nk'])
            else:
                hyp['K']=1
                hyp['N']=hyp['Nk']
        elif hyp['alg']=='DAC-RP':   #DAC-RP with batchsize=1
            keys+=['s0_DTD','s0_DAC','seed','beta_v','beta_theta','is_exact_Ravg','getRvg_err_every_numiter','getDTD_err_every_numiter']
        else:   #DAC-RP with batchsize>1
            keys+=['N_DTD','N_DAC','s0_DTD','s0_DAC','seed','beta_v','beta_theta','is_exact_Ravg','getRvg_err_every_numiter','getDTD_err_every_numiter']
        
        
        toprint='\n Begin hyperparameter set '+str(hyp_index)+'\n'
        hyp_txt.write(toprint)
        toprint='Number of experiments: '+str(expr_num)+'\n'
        hyp_txt.write(toprint)
        if is_print:
            print(toprint)
        toprint='Number of agents: '+str(num_agents)+'\n'
        hyp_txt.write(toprint)
        if is_print:
            print(toprint)
        for key in keys:
            if type(hyp[key]) is list:
                toprint=key+': '+', '.join([str(tmp) for tmp in hyp[key]])
            else:
                toprint=key+': '+str(hyp[key])+'\n'
            hyp_txt.write(toprint)
            if is_print:
                print(toprint)
                
        for k in range(expr_num):
            if is_print:
                print('Begin experiment '+str(k)+'...\n')

            if hyp['alg']=='AC':  #Our AC algorithm
                DAC_dict=\
                DAC(DAC_dict,T=hyp['T'],Tc=hyp['Tc'],Tc_prime=hyp['Tc_prime'],Tr=hyp['Tr'],\
                    N=hyp['N'],Nc=hyp['Nc'],alpha=hyp['alpha'],beta=hyp['beta'],\
                    noise_std=hyp['noise_std'],s0_DAC=hyp['s0_DAC'],s0_DTD=hyp['s0_DTD'],\
                    omega0=hyp['omega0'],theta0=hyp['theta0'],seed_sim_DAC=hyp['seed_sim_DAC'],\
                    seed_sim_DTD=hyp['seed_sim_DTD'],is_print=hyp['is_print'],is_save=hyp['is_save'],\
                    getRvg_err_every_numiter=hyp['getRvg_err_every_numiter'],getDTD_err_every_numiter=hyp['getDTD_err_every_numiter'],\
                    save_folder=hyp['data_folder'])
            elif hyp['alg']=='ACH':  #Our ACH algorithm
                DAC_dict=\
                DAC_h(DAC_dict,T=hyp['T'],Tc=hyp['Tc'],Tc_prime=hyp['Tc_prime'],Tr=hyp['Tr'],\
                    N=hyp['N'],Nc=hyp['Nc'],alpha=hyp['alpha'],alpha_h=hyp['alpha_h'],beta=hyp['beta'],\
                    noise_std=hyp['noise_std'],s0_DAC=hyp['s0_DAC'],s0_DTD=hyp['s0_DTD'],\
                    omega0=hyp['omega0'],theta0=hyp['theta0'],seed_sim_DAC=hyp['seed_sim_DAC'],\
                    seed_sim_DTD=hyp['seed_sim_DTD'],is_print=hyp['is_print'],is_save=hyp['is_save'],\
                    getRvg_err_every_numiter=hyp['getRvg_err_every_numiter'],getDTD_err_every_numiter=hyp['getDTD_err_every_numiter'],\
                    save_folder=hyp['data_folder'])
            elif hyp['alg']=='NAC':   #our NAC algorithm
                DAC_dict=\
                DNAC(DAC_dict,T=hyp['T'],Tc=hyp['Tc'],Tc_prime=hyp['Tc_prime'],Tr=hyp['Tr'],Tz=hyp['Tz'],\
                    Nk=hyp['Nk'],Nc=hyp['Nc'],alpha=hyp['alpha'],beta=hyp['beta'],eta=hyp['eta'],\
                    noise_std=hyp['noise_std'],s0_DAC=hyp['s0_DAC'],s0_DTD=hyp['s0_DTD'],\
                    omega0=hyp['omega0'],theta0=hyp['theta0'],h0=hyp['h0'],seed_sim_DAC=hyp['seed_sim_DAC'],\
                    seed_sim_DTD=hyp['seed_sim_DTD'],is_print=hyp['is_print'],is_save=hyp['is_save'],\
                    getRvg_err_every_numiter=hyp['getRvg_err_every_numiter'],getDTD_err_every_numiter=hyp['getDTD_err_every_numiter'],\
                    save_folder=hyp['data_folder'])
            elif hyp['alg']=='DAC-RP':     #Kaiqing's AC algorithm 2 with batchsize=1
                DAC_dict=DAC_Kaiqing_alg2\
                    (DAC_dict,T=hyp['T'],v0=hyp['v0'],lambda0=hyp['lambda0'],omega0=hyp['omega0'],theta0=hyp['theta0'],\
                     s0_DTD=hyp['s0_DTD'],s0_DAC=hyp['s0_DAC'],seed=hyp['seed'],beta_v=hyp['beta_v'],beta_theta=hyp['beta_theta'],\
                     is_print=hyp['is_print'],is_save=hyp['is_save'],save_folder=hyp['data_folder'],\
                     getRvg_err_every_numiter=hyp['getRvg_err_every_numiter'],getDTD_err_every_numiter=hyp['getDTD_err_every_numiter'],is_exact_Ravg=hyp['is_exact_Ravg'])
            else:    #Kaiqing's AC algorithm 2 with batchsize>1
                DAC_dict=DAC_Kaiqing_alg2_minibatch\
                    (DAC_dict,T=hyp['T'],N_DTD=hyp['N_DTD'],N_DAC=hyp['N_DAC'],v0=hyp['v0'],lambda0=hyp['lambda0'],\
                     omega0=hyp['omega0'],theta0=hyp['theta0'],s0_DTD=hyp['s0_DTD'],s0_DAC=hyp['s0_DAC'],seed=hyp['seed'],\
                     beta_v=hyp['beta_v'],beta_theta=hyp['beta_theta'],is_print=hyp['is_print'],is_save=hyp['is_save'],save_folder=hyp['data_folder'],\
                     getRvg_err_every_numiter=hyp['getRvg_err_every_numiter'],getDTD_err_every_numiter=hyp['getDTD_err_every_numiter'],is_exact_Ravg=hyp['is_exact_Ravg'])
                
            for err_type in err_types:
                if err_type in DAC_dict.keys():
                    if k==0:
                        results[hyp_index][err_type]=np.zeros((expr_num,hyp['T']))
                    results[hyp_index][err_type][k]=DAC_dict[err_type].copy()
        
        if is_save_JdJ:
            if not os.path.isdir(hyp['data_folder']):
                os.makedirs(hyp['data_folder'])
            for err_type in err_types:
                if err_type in results[hyp_index].keys():
                    np.save(hyp['data_folder']+'/all_'+err_type+'.npy',results[hyp_index][err_type])

    if hyp['alg'] == 'AC':
        optimal_pi,optimal_pim,Vmax=get_optimal_pi(transP=DAC_dict['transP'],P_xi=DAC_dict['P_xi'],\
                                                   reward_agentavg=DAC_dict['reward_agentavg'],gamma=DAC_dict['gamma'],eps=1e-7)
        J_max=J(pim=optimal_pim,transP=DAC_dict['transP'],P_xi=DAC_dict['P_xi'],\
                reward_agentavg=DAC_dict['reward_agentavg'],nu_omega=None)
    elif hyp['alg'] == 'ACH':
        optimal_pi,optimal_pim,Vmax=get_optimal_pi(transP=DAC_dict['transP'],P_xi=DAC_dict['P_xi'],\
                                                   reward_agentavg=DAC_dict['reward_agentavg'],gamma=DAC_dict['gamma'],eps=1e-7)
        J_max=J_h(pim=optimal_pim,transP=DAC_dict['transP'],P_xi=DAC_dict['P_xi'],\
                reward_agentavg=DAC_dict['reward_agentavg'],alpha_h=DAC_dict['alpha_h'],nu_omega=None)
    else:
        optimal_pi,optimal_pim,Vmax=get_optimal_pi(transP=DAC_dict['transP'],P_xi=DAC_dict['P_xi'],\
                                                   reward_agentavg=DAC_dict['reward_agentavg'],gamma=DAC_dict['gamma'],eps=1e-7)
        J_max=J(pim=optimal_pim,transP=DAC_dict['transP'],P_xi=DAC_dict['P_xi'],\
                reward_agentavg=DAC_dict['reward_agentavg'],nu_omega=None)

    toprint='J_max='+str(J_max)+'\n'
    hyp_txt.write(toprint)
    if is_print:
        print(toprint)
    hyp_txt.write('Time consumption: '+str((time.time()-start_time)/60)+' minutes \n\n')
    hyp_txt.close()
    return results, J_max

def plots(results,hyps,Jmax=None,color_Jmax='green',marker_Jmax='',percentile=95,fontsize=15,lgdsize=10,bottom_loc=0,left_loc=0,\
          fig_width=8,fig_height=8,J_legend_loc=4,dJ_legend_loc=1,err_legend_loc=1,plot_folder='Figures',is_plotJgap=True):
    if fontsize is not None:
        plt.rcParams.update({'font.size': fontsize})
    hyps_num=len(hyps)
    hyps=hyps.copy()
    
    sample_complexity_cut=float('inf')
    communication_complexity_cut=float('inf')
    for hyp in hyps:
        if hyp['alg']=='NAC':
            hyp['N']=int(np.round(np.sum(hyp['Nk'])))
            hyp['sample_complexity_per_iter']=hyp['Tc']*hyp['Nc']+hyp['N']
            hyp['communication_complexity_per_iter']=hyp['Tc']+hyp['Tc_prime']+hyp['Tr']+hyp['Tz']
        elif hyp['alg']=='AC':
            hyp['sample_complexity_per_iter']=hyp['Tc']*hyp['Nc']+hyp['N']
            hyp['communication_complexity_per_iter']=hyp['Tc']+hyp['Tc_prime']+hyp['Tr']
        elif hyp['alg']=='ACH':
            hyp['sample_complexity_per_iter']=hyp['Tc']*hyp['Nc']+hyp['N']
            hyp['communication_complexity_per_iter']=hyp['Tc']+hyp['Tc_prime']+hyp['Tr']
        elif hyp['alg']=='DAC-RP':     #Kaiqing's algorithm 2 with batchsize=1
            hyp['sample_complexity_per_iter']=2
            hyp['communication_complexity_per_iter']=2
        else:   #Kaiqing's algorithm 2 with batchsize>1
            hyp['sample_complexity_per_iter']=hyp['N_DAC']+hyp['N_DTD']
            hyp['communication_complexity_per_iter']=2
        sample_complexity_cut=np.min([sample_complexity_cut,hyp['sample_complexity_per_iter']*hyp['T']])
        communication_complexity_cut=np.min([communication_complexity_cut,hyp['communication_complexity_per_iter']*hyp['T']])
    
    if not os.path.isdir(plot_folder):
        os.makedirs(plot_folder)
    
    err_types=['Jw','dJ_normsq','Jw_cummean','dJ_normsq_cummean']
    for err_type in err_types:
        if err_type in ['dJ_normsq','dJ_normsq_cummean']:
            legend_loc=dJ_legend_loc
        else:
            legend_loc=J_legend_loc
        for xlabel in ['Communication complexity','Sample complexity']:
            plt.figure(figsize=(fig_width,fig_height))
            
            if xlabel=='Sample complexity':
                xmax=sample_complexity_cut
            else:
                xmax=communication_complexity_cut
            
            for hyp_index in range(hyps_num):
                hyp=hyps[hyp_index]
                result_preplot=results[hyp_index][err_type].copy()
                if xlabel=='Sample complexity':
                    x=np.arange(0,hyp['T'],hyp['sample_dx'])
                    x*=hyp['sample_complexity_per_iter']
                else:
                    x=np.arange(0,hyp['T'],hyp['communication_dx'])
                    x*=hyp['communication_complexity_per_iter']
                keep_index=range(np.sum(x<=xmax))
                x=x[keep_index]
                result_preplot=result_preplot[:,keep_index]
                if is_plotJgap:
                    if err_type not in ['dJ_normsq','dJ_normsq_cummean']:
                        result_preplot=Jmax-result_preplot
                upper_loss = np.percentile(result_preplot, percentile, axis=0)
                lower_loss = np.percentile(result_preplot, 100 - percentile, axis=0)
                avg_loss = np.mean(result_preplot, axis=0)
                
                plt.plot(x,avg_loss,color=hyp['color'],marker=hyp['marker'],label=hyp['legend'])
                plt.fill_between(x,lower_loss,upper_loss, color=hyp['color'],alpha=0.3,edgecolor="none")
            
            if err_type in ['Jw','Jw_cummean']:
                if Jmax is not None:
                    if not is_plotJgap:
                        plt.plot([0,xmax],[Jmax]*2,color=color_Jmax,linestyle=':',label=r'$J_{\max}$')

            if err_type in ['Jw','dJ_normsq']:
                plt.gcf().subplots_adjust(bottom=bottom_loc)
                plt.gcf().subplots_adjust(left=left_loc)
            else:
                plt.gcf().subplots_adjust(bottom=bottom_loc)
                plt.gcf().subplots_adjust(left=left_loc)
            plt.legend(prop={'size':lgdsize},loc=legend_loc)
            plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            
            if xlabel=='Sample complexity':
                plt.xlabel("Sample\ncomplexity")
            else:
                plt.xlabel("Communication\ncomplexity")
            
            if err_type=='dJ_normsq':
                plt.ylabel(r'$||\nabla J(\omega_t)||^2$')
            elif err_type=='Jw':
                if is_plotJgap:
                    plt.ylabel(r'$J^*-J(\omega_t)$')
                else:
                    plt.ylabel(r'$J(\omega_t)$')
            elif err_type=='dJ_normsq_cummean':
                plt.ylabel(r'$\frac{1}{t}\sum_{s=1}^t ||\nabla J(\omega_s)||^2$')
            else:   #'Jw_cummean'
                if is_plotJgap:
                    plt.ylabel(r'$J^*-\frac{1}{t}\sum_{s=1}^t J(\omega_s)$')
                else:
                    plt.ylabel(r'$\frac{1}{t}\sum_{s=1}^t J(\omega_s)$')
    
            if plot_folder is not None:   #Then begin to save figure
                if xlabel=='Sample complexity':
                    xname='_SampleComplexity'
                else:
                    xname='_CommunicationComplexity'
                plt.savefig(plot_folder+'/'+err_type+xname+'.png',dpi=200)
        

def load_results(folders):
    err_types=['Jw','dJ_normsq','Jw_cummean','dJ_normsq_cummean']
    err_types+=['absolute_Ravg_err','absolute_DTD_err','relative_Ravg_err','relative_DTD_err']
    expr_num=len(folders)
    results=[0]*expr_num
    for k in range(expr_num):
        print('Loading from '+folders[k]+' ...')
        results[k]={}
        for err_type in err_types:
            mydir=folders[k]+'/all_'+err_type+'.npy'
            if os.path.exists(mydir):
                results[k][err_type]=np.load(mydir)
                if len(results[k][err_type].shape)==1:
                    results[k][err_type]=results[k][err_type].reshape(1,-1)
    return results

