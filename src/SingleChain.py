# #############################
#
# Cheng Song   (songcheng@snu.ac.kr)
# Modified from Jennifer Dreiling
#
#
# #############################

import copy
import time
import numpy as np
import os.path as op
import random

from seabinv import Model, ModelMatrix
from seabinv import utils

import logging
logger = logging.getLogger()


PAR_MAP = {'agemod': 0, 'rslmod': 1, 'birth': 2, 'death': 2,
           'noise': 3,}
# PAR_MAP = {'vsmod': 0, 'zmod': 1, 'birth': 2, 'death': 2,
#            'noise': 3,}

class SingleChain(object):

    def __init__(self, targets, chainidx=0, initparams={}, modelpriors={},
                 sharedmodels=None, sharedmisfits=None, sharedlikes=None,
                 sharednoise=None, sharedtemperatures=None,
                 sharedlikes_current=None, random_seed=None, initmodel=None):
        self.chainidx = chainidx
        # self.sharedbeta = sharedbeta
        # self.sharedcurrentlikes = sharedcurrentlikes

        self.rstate = np.random.RandomState(random_seed)

        defaults = utils.get_path('defaults.ini')
        self.priors, self.initparams = utils.load_params(defaults)
        self.initparams.update(initparams)
        self.priors.update(modelpriors)
        self.dv = (self.priors['rsl'][1] - self.priors['rsl'][0])
        self.agemin, self.agemax = self.priors['age']
        self.rslmin, self.rslmax = self.priors['rsl']

        self.nchains = self.initparams['nchains']
        self.station = self.initparams['station']

        # t1 = np.ones(int(self.nchains/4)) # select 1/4 of chains with T=1
        # T = np.concatenate((t1,np.logspace(np.log10(1),np.log10(2),self.nchains-np.size(t1)))) # T=1-2
        # self.sharedbeta[chainidx] = 1/T[chainidx]


        # set targets and inversion specific parameters
        self.targets = targets

        # set parameters
        self.iter_phase1 = int(self.initparams['iter_burnin'])
        self.iter_phase2 = int(self.initparams['iter_main'])
        self.iterations = self.iter_phase1 + self.iter_phase2
        self.iiter = -self.iter_phase1
        self.lastmoditer = self.iiter

        self.propdist = np.array(self.initparams['propdist'])
        self.acceptance = self.initparams['acceptance']
        self.yearmin = self.initparams['yearmin']
        self.maxlayers = int(self.priors['segments'][1])

        self.initmodel = initmodel

        # chain models
        self._init_chainarrays(sharedmodels, sharedmisfits, sharedlikes,
                               sharednoise, sharedtemperatures,
                               sharedlikes_current)

        # init model and values
        self._init_model_and_currentvalues()

        # set the modelupdates
        self.modelmods = ['rslmod', 'agemod', 'birth', 'death']
        self.noisemods = [] if len(self.noiseinds) == 0 else ['noise']
        self.modifications = self.modelmods + self.noisemods

        nmods = len(self.propdist)
        self.accepted = np.zeros(nmods)
        self.proposed = np.zeros(nmods)
        self.acceptancerate = np.ones((nmods,100))

# init model and misfit / likelihood

    def _init_model_and_currentvalues(self):
        if self.initmodel:
            x_age, y_rsl = np.loadtxt('results/best_mode_model.txt').T
            imodel = np.concatenate((x_age, y_rsl))
            noiserefs = ['noise_corr', 'noise_sigma']
            init_noise = np.ones(len(self.targets.targets)*2) * np.nan
            corrfix = np.zeros(len(self.targets.targets)*2, dtype=bool)

            self.noisepriors = []
            for i, target in enumerate(self.targets.targets):
                for j, noiseref in enumerate(noiserefs):
                    idx = (2*i)+j
                    noiseprior = self.priors[target.noiseref + noiseref]

                    if type(noiseprior) in [int, float, np.float64]:
                        corrfix[idx] = True
                        init_noise[idx] = noiseprior
                    else:
                        init_noise[idx] = self.rstate.uniform(
                            low=noiseprior[0], high=noiseprior[1])

                    self.noisepriors.append(noiseprior)

            self.noiseinds = np.where(corrfix == 0)[0]
            inoise = np.loadtxt('results/best_mode_noise.txt')
            print('re-sampling')
            # self.currentmodel = imodel
        else:
            imodel = self.draw_initmodel()
            # self.currentmodel = imodel
            inoise, corrfix = self.draw_initnoiseparams()
            # self.currentnoise = inoise

        rcond = self.initparams['rcond']
        self.set_target_covariance(corrfix[::2], inoise[::2], rcond)

        x_age, y_rsl, = Model.get_age_rsl(imodel)
        self.targets.evaluate(x_age, y_rsl, noise=inoise)

        # self.currentmisfits = self.targets.proposalmisfits
        # self.currentlikelihood = self.targets.proposallikelihood

        logger.debug((x_age, y_rsl,))

        self.n = 0  # accepted models counter
        self.accept_as_currentmodel(imodel, inoise)
        self.append_currentmodel()

    def draw_initmodel(self):
        keys = self.priors.keys()
        segments = self.priors['segments'][0]+2 #
        # age = np.zeros(3)
        # rsl = np.zeros(3)
        # age[0] = agemin
        # age[2] = agemax
        # rsl[0] = rslmin
        # rsl[2] = rslmax
        rsl = self.rstate.uniform(low=self.rslmin, high=self.rslmax, size=segments)
        # rsl.sort()
        age = self.rstate.uniform(low=self.agemin, high=self.agemax, size=segments)
        age[0] = self.agemin # fix the first age
        age[-1] = self.agemax # fix the last age
        rsl[0] = self.rslmin
        rsl[-1] = self.rslmax
        # age.sort()  
        model = np.concatenate((age, rsl))
        return(model if self._validmodel(model)
               else self.draw_initmodel())

    def draw_initnoiseparams(self):
        # for each target the noiseparams are (corr and sigma)
        noiserefs = ['noise_corr', 'noise_sigma']
        init_noise = np.ones(len(self.targets.targets)*2) * np.nan
        corrfix = np.zeros(len(self.targets.targets)*2, dtype=bool)

        self.noisepriors = []
        for i, target in enumerate(self.targets.targets):
            for j, noiseref in enumerate(noiserefs):
                idx = (2*i)+j
                noiseprior = self.priors[target.noiseref + noiseref]

                if type(noiseprior) in [int, float, np.float64]:
                    corrfix[idx] = True
                    init_noise[idx] = noiseprior
                else:
                    init_noise[idx] = self.rstate.uniform(
                        low=noiseprior[0], high=noiseprior[1])

                self.noisepriors.append(noiseprior)

        self.noiseinds = np.where(corrfix == 0)[0]
        if len(self.noiseinds) == 0:
            logger.warning('All your noise parameters are fixed. On Purpose?')

        return init_noise, corrfix

    def set_target_covariance(self, corrfix, noise_corr, rcond=None):
        # SWD noise hyper-parameters: if corr is not 0, the correlation of data
        # points assumed will be exponential.
        # RF noise hyper-parameters: if corr is not 0, but fixed, the
        # correlation between data points will be assumed gaussian (realistic).
        # if the prior for RFcorr is a range, the computation switches
        # to exponential correlated noise for RF, as gaussian noise computation
        # is too time expensive because of computation of inverse and
        # determinant each time _corr is perturbed

        for i, target in enumerate(self.targets.targets):
            target_corrfix = corrfix[i]
            target_noise_corr = noise_corr[i]

            self.x_obs = target.obsdata.x

            if not target_corrfix:
                # exponential for each target
                target.get_covariance = target.valuation.get_covariance_exp
                continue

            if (target_noise_corr == 0 and np.any(np.isnan(target.obsdata.yerr))):
                # diagonal for each target, corr inrelevant for likelihood, rel error
                target.get_covariance = target.valuation.get_covariance_nocorr
                continue

            elif target_noise_corr == 0:
                # diagonal for each target, corr inrelevant for likelihood
                target.get_covariance = target.valuation.get_covariance_nocorr_scalederr
                continue

            # gauss for RF
            if target.noiseref == 'rf':
                size = target.obsdata.x.size
                target.valuation.init_covariance_gauss(
                    target_noise_corr, size, rcond=rcond)
                target.get_covariance = target.valuation.get_covariance_gauss

            # exp for noise_corr
            elif target.noiseref == 'swd':
                target.get_covariance = target.valuation.get_covariance_exp

            else:
                message = 'The noise correlation automatically defaults to the \
exponential law. Explicitly state a noise reference for your user target \
(target.noiseref) if wished differently.'
                logger.info(message)
                target.noiseref == 'swd'
                target.get_covariance = target.valuation.get_covariance_exp

    def _init_chainarrays(self, sharedmodels, sharedmisfits, sharedlikes,
                          sharednoise, sharedtemperatures,
                          sharedlikes_current):
        """from shared arrays"""
        ntargets = self.targets.ntargets
        chainidx = self.chainidx
        nchains = self.nchains

        accepted_models = int(self.iterations * np.max(self.acceptance) / 100.)
        self.nmodels = accepted_models  # 'iterations'

        msize = self.nmodels * self.maxlayers * 2
        nsize = self.nmodels * ntargets * 2
        missize = self.nmodels * (ntargets + 1)
        dtype = np.float32

        models = np.frombuffer(sharedmodels, dtype=dtype).\
            reshape((nchains, msize))
        misfits = np.frombuffer(sharedmisfits, dtype=dtype).\
            reshape((nchains, missize))
        likes = np.frombuffer(sharedlikes, dtype=dtype).\
            reshape((nchains, self.nmodels))
        noise = np.frombuffer(sharednoise, dtype=dtype).\
            reshape((nchains, nsize))
        temperatures = np.frombuffer(sharedtemperatures, dtype=dtype).\
            reshape((nchains,self.iterations))
        self.currentlike_shared = np.frombuffer(
            sharedlikes_current, dtype=dtype)

        self.chainmodels = models[chainidx].reshape(
            self.nmodels, self.maxlayers*2)
        self.chainmisfits = misfits[chainidx].reshape(
            self.nmodels, ntargets+1)
        self.chainlikes = likes[chainidx]
        self.sharedlikes = likes
        self.chainnoise = noise[chainidx].reshape(
            self.nmodels, ntargets*2)
        self.chainiter = np.ones(self.chainlikes.size) * np.nan

        self.temperatures = temperatures[chainidx]
        self.temperature = self.temperatures[0]


# update current model (change layer number and values)

    def _model_layerbirth(self, model):
        """
        Draw a random segment from x_age and assign a new RSL.

        The new RSL is based on the before RSL value at the drawn age
        position (self.propdist[2]).
        """
        x_age, y_rsl = Model.get_age_rsl(model)

        # new age
        agemin, agemax = self.priors['age']
        age_birth = self.rstate.uniform(low=agemin, high=agemax)

        ind = np.argmin((abs(x_age - age_birth)))  # closest z
        rsl_before = y_rsl[ind]
        rsl_birth = rsl_before + self.rstate.normal(0, self.propdist[2])

        age_new = np.concatenate((x_age, [age_birth]))
        rsl_new = np.concatenate((y_rsl, [rsl_birth]))

        self.drsl2 = np.square(rsl_birth - rsl_before)
        return np.concatenate((age_new, rsl_new))

    def _model_layerdeath(self, model):
        """
        Remove a random segment from model. Delete corresponding
        RSL from model.
        """
        x_age, y_rsl = Model.get_age_rsl(model)

        low = 1
        high = int(model.size / 2)
        if low==high:
            ind_death = low
        else:
            ind_death = self.rstate.randint(low, high)

        age_before = x_age[ind_death]
        rsl_before = y_rsl[ind_death]

        age_new = np.delete(x_age, ind_death)
        rsl_new = np.delete(y_rsl, ind_death)
        ind = np.argmin((abs(age_new - age_before)))
        rsl_after = rsl_new[ind]
        self.drsl2 = np.square(rsl_after - rsl_before)
        return np.concatenate((age_new, rsl_new))

    def _model_rslchange(self, model):
        """Randomly chose a layer to change RSL with Gauss distribution."""
        ind = self.rstate.randint(model.size / 2, model.size)
        rsl_mod = self.rstate.normal(0, self.propdist[1])
        model[ind] = model[ind] + rsl_mod
        return model

    def _model_age_move(self, model):
        """Randomly chose a layer to change age with Gauss distribution."""
        low = 1
        high = int(model.size / 2)
        if low==high:
            ind = low
        else:
            ind = self.rstate.randint(low, high)
        age_mod = self.rstate.normal(0, self.propdist[0])
        model[ind] = model[ind] + age_mod
        return model

    def _get_modelproposal(self, modify):
        model = copy.copy(self.currentmodel)

        if modify == 'rslmod':
            propmodel = self._model_rslchange(model)
        elif modify == 'agemod':
            propmodel = self._model_age_move(model)
        elif modify == 'birth':
            propmodel = self._model_layerbirth(model)
        elif modify == 'death':
            propmodel = self._model_layerdeath(model)

        return self._sort_modelproposal(propmodel)

    def _sort_modelproposal(self, model):
        """
        Return the sorted proposal model.

        This method is necessary, if the age from the new proposal model
        are not ordered, i.e. if one age value is added or strongly modified.
        """
        x_age, y_rsl = Model.get_age_rsl(model)
        if np.all(np.diff(x_age) > 0):   # monotone increasing
            return model
        else:
            ind = np.argsort(x_age)
            model_sort = np.concatenate((x_age[ind], y_rsl[ind]))
        return model_sort

    def _validmodel(self, model):
        """
        Check model before the forward modeling.

        - The rsl must contain all values <= 0.
        # - The layer thicknesses must be at least thickmin km.
        # - if lvz: low velocity zones are allowed with the deeper layer velocity
        #    no smaller than (1-perc) * velocity of layer above.
        # - ... and some other constraints. E.g. vs boundaries (prior) given.
        """
        x_age, y_rsl = Model.get_age_rsl(model)

        # only allow at most 1 segment between year's interval
        hist, _ = np.histogram(x_age, bins=self.x_obs)
        if np.any(hist > 1):
            return False

        # check whether nlayers lies within the prior
        layermin = self.priors['segments'][0]
        layermax = self.priors['segments'][1]
        layermodel = x_age.size
        if not (layermodel >= layermin and layermodel <= layermax):
            logger.debug("chain%d: model- nlayers not in prior"
                         % self.chainidx)
            return False

        # check model for layers with thicknesses of smaller yearmin
        if np.any(x_age[1:]-x_age[:-1] < self.yearmin):
            logger.debug("chain%d: year are not larger than yearmin"
                         % self.chainidx)
            return False

        # check whether rsl lies within the prior
        # rslmin = self.priors['rsl'][0]
        # rslmax = self.priors['rsl'][1]
        if np.any(y_rsl < self.rslmin) or np.any(y_rsl > self.rslmax):
            logger.debug("chain%d: model- rsl not in prior"
                         % self.chainidx)
            return False

        # check whether interfaces lie within prior
        # zmin = self.priors['age'][0]
        # zmax = self.priors['age'][1]
        if np.any(x_age < self.agemin) or np.any(x_age > self.agemax):
            logger.debug("chain%d: model- age not in prior"
                         % self.chainidx)
            return False

        # if self.lowvelperc is not None:
        #     # check model for low velocity zones. If larger than perc, then
        #     # compvels must be positive
        #     compvels = vs[1:] - (vs[:-1] * (1 - self.lowvelperc))
        #     if not compvels.size == compvels[compvels > 0].size:
        #         logger.debug("chain%d: low velocity zone issues"
        #                      % self.chainidx)
        #         return False

        # if self.highvelperc is not None:
        #     # check model for high velocity zones. If larger than perc, then
        #     # compvels must be positive.
        #     compvels = (vs[:-1] * (1 + self.highvelperc)) - vs[1:]
        #     if not compvels.size == compvels[compvels > 0].size:
        #         logger.debug("chain%d: high velocity zone issues"
        #                      % self.chainidx)
        #         return False

        return True

    def _get_hyperparameter_proposal(self):
        noise = copy.copy(self.currentnoise)
        ind = self.rstate.choice(self.noiseinds)

        noise_mod = self.rstate.normal(0, self.propdist[3])
        noise[ind] = noise[ind] + noise_mod
        return noise

    def _validnoise(self, noise):
        for idx in self.noiseinds:
            if noise[idx] < self.noisepriors[idx][0] or \
                    noise[idx] > self.noisepriors[idx][1]:
                return False
        return True

# accept / save current models

    def adjust_propdist(self):
        """
        Modify self.propdist to adjust acceptance rate of models to given
        percentace span: increase or decrease by five percent.
        """
        with np.errstate(invalid='ignore'):
            acceptrate = self.accepted / self.proposed * 100

        # minimum distribution width forced to be not less than 1 m/s, 1 m
        # actually only touched by vs distribution
        propdistmin = np.full(acceptrate.size, 0.001)

        for i, rate in enumerate(acceptrate):
            if np.isnan(rate):
                # only if not inverted for
                continue
            if rate < self.acceptance[0]:
                new = self.propdist[i] * 0.95
                if new < propdistmin[i]:
                    new = propdistmin[i]
                self.propdist[i] = new

            elif rate > self.acceptance[1]:
                self.propdist[i] = self.propdist[i] * 1.05
            else:
                pass

    def get_acceptance_probability(self, modify):
        """
        Acceptance probability will be computed dependent on the modification.

        Parametrization alteration (Vs or voronoi nuclei position)
            the acceptance probability is equal to likelihood ratio.

        Model dimension alteration (layer birth or death)
            the probability was computed after the formulation of Bodin et al.,
            2012: 'Transdimensional inversion of receiver functions and
            surface wave dispersion'.
        """
        if modify in ['rslmod', 'agemod', 'noise']:
            # only velocity or thickness changes are made
            # also used for noise changes
            alpha = (self.targets.proposallikelihood - self.currentlikelihood)/self.temperature

        elif modify in ['birth', ]:
            theta = self.propdist[2]  # Gaussian distribution
            # self.drsl2 = delta rsl square = np.square(v'_(k+1) - v_(i))
            A = (theta * np.sqrt(2 * np.pi)) / self.dv
            B = self.drsl2 / (2. * np.square(theta))
            C = (self.targets.proposallikelihood - self.currentlikelihood)/self.temperature

            alpha = np.log(A) + B + C

        elif modify in ['death', ]:
            theta = self.propdist[2]  # Gaussian distribution
            # self.drsl2 = delta rsl square = np.square(v'_(j) - v_(i))
            A = self.dv / (theta * np.sqrt(2 * np.pi))
            B = self.drsl2 / (2. * np.square(theta))
            C = (self.targets.proposallikelihood - self.currentlikelihood)/self.temperature

            alpha = np.log(A) - B + C
        return alpha

    def accept_as_currentmodel(self, model, noise):
        """Assign currentmodel and currentvalues to self."""
        self.currentmisfits = self.targets.proposalmisfits
        self.currentlikelihood = self.targets.proposallikelihood
        self.currentlike_shared[self.chainidx] = self.currentlikelihood
        self.currentmodel = model
        self.currentnoise = noise
        self.lastmoditer = self.iiter

    def append_currentmodel(self):
        """Append currentmodel to chainmodels and values."""
        self.chainmodels[self.n, :self.currentmodel.size] = self.currentmodel
        self.chainmisfits[self.n, :] = self.currentmisfits
        self.chainlikes[self.n] = self.currentlikelihood
        self.chainnoise[self.n, :] = self.currentnoise

        self.chainiter[self.n] = self.iiter
        self.n += 1

# run

    def iterate(self):

        # set starttime
        if self.iiter == -self.iter_phase1:
            self.tstart = time.time()
            self.tnull = time.time()

        self.temperature = self.temperatures[self.iiter+self.iter_phase1]
        proposalmodel = None # for safety

        if self.iiter < (-self.iter_phase1 + (self.iterations * 0.01)):
            # only allow age and rsl modifications the first 1 % of iterations
            modify = self.rstate.choice(['rslmod', 'agemod'] + self.noisemods)
        else:
            modify = self.rstate.choice(self.modifications)

        if modify in self.modelmods:
            proposalmodel = self._get_modelproposal(modify)
            proposalnoise = self.currentnoise
            if not self._validmodel(proposalmodel):
                proposalmodel = None

        elif modify in self.noisemods:
            proposalmodel = self.currentmodel
            proposalnoise = self._get_hyperparameter_proposal()
            if not self._validnoise(proposalnoise):
                proposalmodel = None

        if proposalmodel is None:
            # If not a valid proposal model and noise params are found,
            # leave self.iterate and try with another modification
            # should not occur often.
            logger.debug('Not able to find a proposal for %s' % modify)
            self.iiter += 1
            return

        else:
            # compute synthetic data and likelihood, misfit
            x_age, y_rsl = Model.get_age_rsl(proposalmodel)
            self.targets.evaluate(x_age, y_rsl, noise=proposalnoise)

            paridx = PAR_MAP[modify]
            self.proposed[paridx] += 1

            # Replace self.currentmodel with proposalmodel with acceptance
            # probability alpha. Accept candidate sample (proposalmodel)
            # with probability alpha, or reject it with probability (1 - alpha).
            # these are log values ! alpha is log.
            u = np.log(self.rstate.uniform(0, 1))
            alpha = self.get_acceptance_probability(modify)

            # #### _____________________________________________________________
            if u < alpha:
                # always the case if self.jointlike > self.bestlike (alpha>1)
                self.accept_as_currentmodel(proposalmodel, proposalnoise)
                # avoid that the preallocated array is overfilled if the acceptancerate is too high
                if np.sum(self.acceptancerate[paridx])>self.acceptance[1]:
                    if np.random.uniform() < self.acceptance[1]/np.sum(self.acceptancerate[paridx]):
                        self.append_currentmodel()
                else:
                    self.append_currentmodel()
                self.accepted[paridx] += 1
                self.acceptancerate[paridx][0] = 1
                self.acceptancerate[paridx] = np.roll(self.acceptancerate[paridx],1)
            else:
                self.acceptancerate[paridx][0] = 0
                self.acceptancerate[paridx] = np.roll(self.acceptancerate[paridx],1)
            
        # print inversion status information
        if self.iiter % 5000 == 0 or self.iiter == -self.iter_phase1:
            runtime = time.time() - self.tnull
            current_iterations = self.iiter + self.iter_phase1

            if current_iterations > 0:
                acceptrate = np.sum(self.acceptancerate,axis=1)
                rates = ''
                for rate in acceptrate:
                    rates += '%2d ' %rate
                acceptrate_total = float(self.n) / current_iterations * 100.

                logger.info('Chain %3d (T=%5.2f): %6d %5d + hs %8.3f\t%9d |%6.1f s  | %s (%.1f%%)' % (
                    self.chainidx,self.temperature,
                    self.lastmoditer, self.currentmodel.size/2 - 1,
                    self.currentmisfits[-1], self.currentlikelihood,
                    runtime, rates, acceptrate_total))

            self.tnull = time.time()

        # stabilize model acceptance rate
        if self.iiter % 1000 == 0:
            if np.all(self.proposed) != 0:
                self.adjust_propdist()

        self.iiter += 1

        # set endtime
        if self.iiter == self.iter_phase2:
            self.tend = time.time()
        # if self.iiter >= self.iter_phase2-1:
        #     self.sharedbeta = np.ones(self.nchains) # T=1-2
        #     print('T at chain', self.sharedbeta, self.chainidx)

    def finalize(self):
        self.tend = time.time()
        runtime = (self.tend - self.tstart)

        # update chain values (eliminate nan rows)
        self.chainmodels = self.chainmodels[:self.n, :]
        self.chainmisfits = self.chainmisfits[:self.n, :]
        self.chainlikes = self.chainlikes[:self.n]
        self.chainnoise = self.chainnoise[:self.n, :]
        self.chainiter = self.chainiter[:self.n]

        # only consider models after burnin phase
        p1ind = np.where(self.chainiter < 0)[0]
        p2ind = np.where(self.chainiter >= 0)[0]

        if p1ind.size != 0:
            wmodels, wlikes, wmisfits, wnoise = self.get_weightedvalues(
                pind=p1ind, finaliter=0)
            self.p1models = wmodels  # p1 = phase one
            self.p1misfits = wmisfits
            self.p1likes = wlikes
            self.p1noise = wnoise
            self.p1temperatures = self.temperatures[:self.iter_phase1]

        if p2ind.size != 0:
            wmodels, wlikes, wmisfits, wnoise = self.get_weightedvalues(
                pind=p2ind, finaliter=self.iiter)
            self.p2models = wmodels  # p2 = phase two
            self.p2misfits = wmisfits
            self.p2likes = wlikes
            self.p2noise = wnoise
            self.p2temperatures = self.temperatures[self.iter_phase1:]

        accmodels = float(self.p2likes.size)  # accepted models in p2 phase
        maxmodels = float(self.initparams['maxmodels'])  # for saving
        self.thinning = int(np.ceil(accmodels / maxmodels))
        self.save_finalmodels()

        logger.debug('time for inversion: %.2f s' % runtime)

    def get_weightedvalues(self, pind, finaliter):
        """
        Models will get repeated (weighted).

        Each iteration, if there was no model proposal accepted, the current
        model gets repeated once more. This weight is based on self.chainiter,
        which documents the iteration of the last accepted model."""
        pmodels = self.chainmodels[pind]  # p = phase (1 or 2)
        pmisfits = self.chainmisfits[pind]
        plikes = self.chainlikes[pind]
        pnoise = self.chainnoise[pind]
        pweights = np.diff(np.concatenate((self.chainiter[pind], [finaliter])))

        wmodels, wlikes, wmisfits, wnoise = ModelMatrix.get_weightedvalues(
            pweights, models=pmodels, likes=plikes, misfits=pmisfits,
            noiseparams=pnoise)
        return wmodels, wlikes, wmisfits, wnoise

    def save_finalmodels(self):
        """Save chainmodels as pkl file"""
        savepath = op.join(self.initparams['savepath'], 'data')
        names = ['models', 'likes', 'misfits', 'noise']

        # phase 1 -- burnin
        try:
            for i, data in enumerate([self.p1models, self.p1likes,
                                     self.p1misfits, self.p1noise]):
                outfile = op.join(savepath, 'c%.3d_p1%s' % (self.chainidx, names[i]))
                np.save(outfile, data[::self.thinning])
        except:
            logger.info('No burnin models accepted.')

        # phase 2 -- main / posterior phase
        try:
            for i, data in enumerate([self.p2models, self.p2likes,
                                     self.p2misfits, self.p2noise]):
                outfile = op.join(savepath, 'c%.3d_p2%s' % (self.chainidx, names[i]))
                np.save(outfile, data[::self.thinning])

            logger.info('> Saving %d models (main phase).' % len(data[::self.thinning]))
        except:
            logger.info('No main phase models accepted.')
