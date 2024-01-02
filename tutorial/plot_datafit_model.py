import numpy as np
import os.path as op
from BayHunter import SynthObs
import matplotlib.pyplot as plt
from BayHunter import Model

# idx = 1
# h = [34, 0]
# vs = [3.5, 4.4]

# idx = 2
# h = [5, 29, 0]
# vs = [3.4, 3.8, 4.5]

# idx = 3
# h = [5, 23, 8, 0]
# vs = [2.7, 3.6, 3.8, 4.4]
fname = 'best_mode_model.txt'
h, vp, vs = np.loadtxt('results/'+fname).T
cvp, cvs, cdepth = Model.get_stepmodel_from_h(h=h, vs=vs, vp=vp)
plt.figure()
plt.plot(cvs, cdepth, ls='-', lw=0.8, alpha=0.5)
plt.ylim([3,0])
plt.savefig('results/init_model.tif')

vpvs = vp[0]/vs[0]
# print(vpvs)


# surface waves
sw_x = np.linspace(0.5, 2.5, 41)
swdata = SynthObs.return_swddata(h, vs, vpvs=vpvs, x=sw_x)

_, obs_ph = np.loadtxt('./observed/CH_rdispph.dat').T
_, obs_gr = np.loadtxt('./observed/CH_rdispgr.dat').T
rwe_x, obs_hvsr = np.loadtxt('./observed/CH_rwe.dat').T

# rayleigh wave ellipticity
rwedata = SynthObs.return_rwedata(h, vs, vpvs=vpvs, x=rwe_x)

plt.figure()
plt.plot(sw_x, swdata['rdispph'][1,:], label='synthetic')
plt.plot(sw_x, obs_ph, label='observed')
plt.legend()
plt.savefig('results/ph_model.tif')

plt.figure()
plt.plot(sw_x, swdata['rdispgr'][1,:], label='synthetic')
plt.plot(sw_x, obs_gr, label='observed')
plt.legend()
plt.savefig('results/gv_model.tif')

plt.figure()
plt.plot(rwe_x, rwedata['rwe'][1,:], label='synthetic')
plt.plot(rwe_x, obs_hvsr, label='observed')
plt.legend()
plt.savefig('results/rwe_model.tif')
