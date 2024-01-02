import numpy as np
vpvs_all = np.load('results/data/c_vpvs.npy')
vpvs_mean = np.mean(vpvs_all)
print(vpvs_mean)
np.savetxt('vpvs.txt', [vpvs_mean])
