These are python codes for the ICML 2022 paper titled "Sample and Communication-Efficient Decentralized Actor-Critic Algorithms".

In DAC_main.py, you can set the variable "network" to be either 'ring' or 'full' at the beginning and run to generate the results for ring network
 or fully connected network respectively for the simulation experiment. Specifically, in the folder Results/simulation_ring for ring network, 
the result figures in Figures_AC and Figures_NAC correspond to Figures 2 and 5 in Appendix E, the other folders save the numeric results in numpy format, 
and "hyperparameters.txt" lists the hyperparameters of the alorithms. The folder Results/simulation_full for fully connected network is similar, 
except that the result figures in Figures_AC and Figures_NAC correspond to Figures 3 and 4 in Appendix E. 

Running DAC_utils_gridworld.py yields the folder Results/Cliff which contains the results of the two-agent cliff navigation problem in a grid-world, 
corresponding to Figures 7 and 8 in Appendix E. 


