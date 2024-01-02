# #############################
#
# Cheng Song (chengsong@snu.ac.kr)
# Based on Jennifer Dreiling   (dreiling@gfz-potsdam.de)
#
#
# #############################

import os
# set os.environment variables to ensure that numerical computations
# do not do multiprocessing !! Essential !! Do not change !
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import os.path as op
import matplotlib

from seabinv import PlotFromStorage
from seabinv import Targets
from seabinv import utils
from seabinv import MCMC_Optimizer
from seabinv import ModelMatrix
import logging


#
# console printout formatting
#
formatter = ' %(processName)-12s: %(levelname)-8s |  %(message)s'
logging.basicConfig(format=formatter, level=logging.INFO)
logger = logging.getLogger()


#
# ------------------------------------------------------------  obs SYNTH DATA
#
# Load priors and initparams from config.ini or simply create dictionaries.
initfile = 'config.ini'
priors, initparams = utils.load_params(initfile)

# Load observed data
station = 'NJ_CC.csv'
sea_level = pd.read_csv('observed/'+station)
x_obs_raw = 1950-np.array(sea_level['Age'])
y_obs_raw = np.array(sea_level['RSL'])
sort_idx = np.argsort(x_obs_raw)
x_obs = x_obs_raw[sort_idx]
y_obs = y_obs_raw[sort_idx]

target1 = Targets.RelativeSeaLevel(x_obs, y_obs)
targets = Targets.JointTarget(targets=[target1])


#
#  ---------------------------------------------------  Quick parameter update
#
# "priors" and "initparams" from config.ini are python dictionaries. You could
# also simply define the dictionaries directly in the script, if you don't want
# to use a config.ini file. Or update the dictionaries as follows, e.g. if you
# have station specific values, etc.
# See docs/bayhunter.pdf for explanation of parameters

               # 'rfnoise_sigma': np.std(yrf_err),  # fixed to true value
               # 'swdnoise_sigma': np.std(ysw_err),  # fixed to true value
priors.update({'age': (x_obs[0], x_obs[-1]),  # optional, moho estimate (mean, std)
               'rsl': (min(y_obs),max(y_obs)),
               })
initparams.update({'nchains': 6,
                   'iter_burnin': (5000000),
                   'iter_main': (100000),
                   'propdist': ((priors['age'][1]-priors['age'][0])/30, (priors['rsl'][1]-priors['rsl'][0])/30, (priors['rsl'][1]-priors['rsl'][0])/30, 0.005),
                   })


#
#  -------------------------------------------------------  MCMC BAY INVERSION
#
# Save configfile for baywatch. refmodel must not be defined.
utils.save_baywatch_config(targets, path='.', priors=priors,
                           initparams=initparams)
#optimizer = MCMC_Optimizer(targets, initparams=initparams, priors=priors,
#                           random_seed=None, initmodel=False, parallel_tempering=True)
#optimizer.mp_inversion(nthreads=6, baywatch=False)


#
# #  ---------------------------------------------- Model resaving and plotting
path = initparams['savepath']
cfile = '%s_config.pkl' % initparams['station']
configfile = op.join(path, 'data', cfile)
obj = PlotFromStorage(configfile)
# The final distributions will be saved with save_final_distribution.
# Beforehand, outlier chains will be detected and excluded.
# Outlier chains are defined as chains with a likelihood deviation
# of dev * 100 % from the median posterior likelihood of the best chain.
obj.save_final_distribution(maxmodels=100000)
# Save a selection of important plots
obj.save_plots(nchains=initparams['nchains'], depint = 2)

#
# If you are only interested on the mean posterior velocity model, type:
file = op.join(initparams['savepath'], 'data/c_models.npy')
models = np.load(file)
singlemodels = ModelMatrix.get_singlemodels(models, dep_int = x_obs)
vs, dep = singlemodels['mean']
np.savetxt('./results/mean_model.txt', np.column_stack((dep,vs)))
vs, dep = singlemodels['median']
np.savetxt('./results/median_model.txt', np.column_stack((dep,vs)))
vs, dep = singlemodels['mode']
np.savetxt('./results/mode_model.txt', np.column_stack((dep,vs)))
stdminmax, dep = singlemodels['stdminmax']
np.savetxt('./results/std_model.txt', np.column_stack((dep, stdminmax[0,:], stdminmax[1,:])))

#
# #  ---------------------------------------------- WATCH YOUR INVERSION
# if you want to use BayWatch, simply type "baywatch ." in the terminal in the
# folder you saved your baywatch configfile or type the full path instead
# of ".". Type "baywatch --help" for further options.

# if you give your public address as option (default is local address of PC),
# you can also use BayWatch via VPN from 'outside'.
# address = '139.?.?.?'  # here your complete address !!!
