import numpy as np
import os.path as op
from BayHunter import SynthObs

# idx = 1
# h = [34, 0]
# vs = [3.5, 4.4]

# idx = 2
# h = [5, 29, 0]
# vs = [3.4, 3.8, 4.5]

# idx = 3
# h = [5, 23, 8, 0]
# vs = [2.7, 3.6, 3.8, 4.4]

idx = 4
h = [0.707676947, 1.82288980-0.707676947, 0]
vs = [2.77755332, 3.23467851, 3.52474833 ]

vpvs = 1.73

path = 'observed'
datafile = op.join(path, 'TJN%d_%s.dat' % (idx, '%s'))

# surface waves
sw_x = np.linspace(3, 7, 5)
swdata = SynthObs.return_swddata(h, vs, vpvs=vpvs, x=sw_x)
SynthObs.save_data(swdata, outfile=datafile)

# rayleigh wave ellipticity
rwe_x = np.linspace(0.1, 6, 15)
rwedata = SynthObs.return_rwedata(h, vs, vpvs=vpvs, x=rwe_x)
SynthObs.save_data(rwedata, outfile=datafile)

# receiver functions
pars = {'p': 6.4}
# datafile = op.join(path, 'st%d_%s.dat' % (idx, '%s'))
rfdata = SynthObs.return_rfdata(h, vs, vpvs=vpvs, x=None)
SynthObs.save_data(rfdata, outfile=datafile)

# velocity-depth model
modfile = op.join(path, 'TJN%d_mod.dat' % idx)
SynthObs.save_model(h, vs, vpvs=vpvs, outfile=modfile)
