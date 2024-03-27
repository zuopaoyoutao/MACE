By running "DAC_main()" on Spyder 4.1.5 using Python 3.8.5, a new folder named "FinalResults" will be created. 
This folder includes the following files: 

(1) "hyperparameters.txt" which shows the hyperparameter choices
(2) "Figures_AC" which shows the result figures for two decentralized AC algorithms, ours and Algorithm 2 of [1].
(3) "Figures_NAC" which shows the result figures for our decentralized NAC algorithms.
(4) The result data of numpy files for each hyperparameter choice are saved into the following folders
DACRP
ourAC_N100_alpha10
ourAC_N500_alpha50
ourAC_N2000_alpha200
ourNAC_N100_alpha0.2_eta0.05_K50
ourNAC_N2000_alpha4_eta1_K200
ourNAC_N500_alpha1_eta0.25_K100
DACRP1
DACRP100

Each of the above data folders contains the following numpy data files:
'all_Jw.npy', 'all_Jw_cummean.npy': Objective function value and its accumulated mean.
'all_dJ_normsq.npy', 'all_dJ_normsq_cummean.npy': squared objective gradient norm and its accumulated mean.

'all_absolute_DTD_err.npy', 'all_relative_DTD_err.npy': absolute and relative TD errors.
'all_absolute_Ravg_err.npy', 'all_relative_Ravg_err.npy': absolute and relative errors of average reward estimation 

Each data file contains a matrix where each row denotes an implementation, and each column denotes an iteration.



References:
[1] Zhang, K., Yang, Z., Liu, H., Zhang, T., & Basar, T. (2018, July). Fully decentralized multi-agent reinforcement learning with networked agents. 
In International Conference on Machine Learning (pp. 5872-5881). PMLR.

