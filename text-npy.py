#导入所需的包
import numpy as np
import pandas as pd
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

#导入npy文件路径位置
test = np.load('C:\\Users\\63023\Desktop\DAC副本 2\Results\Cliff\ourACH_N100_alpha1\\all_dJ_normsq.npy')
test1 = np.load('C:\\Users\\63023\Desktop\DAC副本 2\Results\Cliff\ourACH_N100_alpha1\\all_dJ_normsq_cummean.npy')
test2 = np.load('C:\\Users\\63023\Desktop\DAC副本 2\Results\Cliff\ourACH_N100_alpha1\\all_Jw.npy')
test3 = np.load('C:\\Users\\63023\Desktop\DAC副本 2\Results\Cliff\ourACH_N100_alpha1\\all_Jw_cummean.npy')

print(test)
print(test1)
print(test2)
print(test3)

with open('C:\\Users\\63023\Desktop\\dJ_normsq.txt', 'w') as outfile:
    for slice_2d in test:
        np.savetxt(outfile, slice_2d, fmt = '%f', delimiter = ',')

with open('C:\\Users\\63023\Desktop\\dJ_normsq_cummean.txt', 'w') as outfile:
    for slice_2d in test1:
        np.savetxt(outfile, slice_2d, fmt = '%f', delimiter = ',')

with open('C:\\Users\\63023\Desktop\\Jw.txt', 'w') as outfile:
    for slice_2d in test2:
        np.savetxt(outfile, slice_2d, fmt = '%f', delimiter = ',')

with open('C:\\Users\\63023\Desktop\\Jw_cummean.txt', 'w') as outfile:
    for slice_2d in test3:
        np.savetxt(outfile, slice_2d, fmt = '%f', delimiter = ',')