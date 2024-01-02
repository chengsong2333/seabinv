import numpy as np
z_i, vs_i = np.loadtxt('results/mode_model.txt').T
vs_o, indices_o = np.unique(vs_i, return_index=True)
n = len(vs_o)
indices = sorted(indices_o)
vs = vs_i[indices]
z = np.zeros(n)
z[:-1] = z_i[np.array(indices[1:])-1]
z[-1] = z_i[-1]
z_vnoi = np.zeros(n)
z_vnoi[0] = z[0]/2
for i in range(n-1):
    z_vnoi[i+1] = 2*z[i]-z_vnoi[i]
np.savetxt('initmodel.txt', np.column_stack((vs, z_vnoi)))
