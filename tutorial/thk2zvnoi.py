import numpy as np
thk, vp, vs = np.loadtxt('results/best_mode_model.txt').T
vpvs = vp[0]/vs[0]
n = len(vs)
z = np.cumsum(thk)
z[-1] = 3
z_vnoi = np.zeros(n)
z_vnoi[0] = z[0]/2
for i in range(n-1):
    z_vnoi[i+1] = 2*z[i]-z_vnoi[i]
np.savetxt('initmodel.txt', np.column_stack((vs, z_vnoi)))
np.savetxt('initvpvs.txt', [vpvs])
