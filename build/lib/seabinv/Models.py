# #############################
#
# Cheng Song (songcheng@snu.ac.kr)
# Modified from Jennifer Dreiling
#
#
# #############################

import numpy as np
import copy


class Model(object):
    """Handle interpolating methods for a single model vector."""

    @staticmethod
    def get_age_rsl(model):
        """Return x_age, y_rsl from a input model [x_age, y_rsl]"""
        model = model[~np.isnan(model)]
        n = int(model.size / 2)  # layers

        x_age = model[:n]
        y_rsl = model[-n:]

        return x_age, y_rsl
    
    @staticmethod
    def get_age_rsl_segment(model):
        """Return vp, vs and h from a input model [vs, z_disc]"""
        x_age, y_rsl = Model.get_age_rsl(model)
        # discontinuities:
        # z_disc = (z_vnoi[:n-1] + z_vnoi[1:n]) / 2.
        x_age_segment = (x_age - np.concatenate(([0], x_age[:-1])))

        return x_age_segment, y_rsl
    
    @staticmethod
    def get_stepmodel(model):
        """Return a steplike model from input model, for plotting."""
        x_age, y_rsl = Model.get_age_rsl(model)

        # insert steps into rsl model
        x_age_step = np.concatenate([(d, d) for d in x_age])
        y_rsl_step = np.concatenate([(v, v) for v in y_rsl])

        # dep_step[-1] = np.max([150, dep_step[-1] * 2.5])  # half space

        return x_age_step, y_rsl_step

    @staticmethod
    def get_stepmodel_from_h(x_age, y_rsl):
        # """Return a steplike model from input model."""
        # # insert steps into velocity model
        # if dep is None:
        #     dep = np.cumsum(h)

        # if vp is None:
        #     if mantle is not None:
        #         vp = Model.get_vp(vs, vpvs, mantle)
        #     else:
        #         vp = vs * vpvs

        x_age = np.concatenate([(age, age) for age in x_age])
        y_rsl = np.concatenate([(rsl, rsl) for rsl in y_rsl])

        return x_age, y_rsl

    # @staticmethod
    def get_interpmodel(model, dep_int):
        """
        Return an interpolated stepmodel, for (histogram) plotting.

        Model is a vector of the parameters.
        """
        x_age_step, y_rsl_step = Model.get_stepmodel(model)
        vs_int = np.interp(dep_int, x_age_step, y_rsl_step)

        return vs_int


class ModelMatrix(object):
    """
    Handle interpolating methods for a collection of single models.

    Same as the Model class, but for a matrix. Only for Plotting
    or after inversion.
    """

    @staticmethod
    def _delete_nanmodels(models):
        """Remove nan models from model-matrix."""
        cmodels = copy.copy(models)
        mean = np.nanmean(cmodels, axis=1)
        nanidx = np.where((np.isnan(mean)))[0]

        if nanidx.size == 0:
            return cmodels
        else:
            return np.delete(cmodels, nanidx, axis=0)

    @staticmethod
    def _replace_age_rsl(models):
        """
        Return model matrix with (x_age, y_rsl) - models.

        Each model in the matrix is parametrized with (x_age, y_rsl).
        For plotting, h will be computed from z."""
        models = ModelMatrix._delete_nanmodels(models)

        for i, model in enumerate(models):
            x_age, y_rsl = Model.get_age_rsl(model)
            newmodel = np.concatenate((x_age, y_rsl))
            models[i][:newmodel.size] = newmodel
        return models

    # @staticmethod
    def get_interpmodels(models, dep_int):
        """Return model matrix with interpolated stepmodels.

        Each model in the matrix is parametrized with (vs, z)."""
        models = ModelMatrix._delete_nanmodels(models)

        ages_int = np.repeat([dep_int], len(models), axis=0)
        rsls_int = np.empty((len(models), dep_int.size))

        for i, model in enumerate(models):
            # for vs, dep 2D histogram
            rsl_int = Model.get_interpmodel(model, dep_int)
            rsls_int[i] = rsl_int

        return rsls_int, ages_int

    @staticmethod
    def get_singlemodels(models, dep_int=None, misfits=None):
        """Return specific single models from model matrix (vs, depth).
        The model is a step model for plotting.

        -- interpolated
        (1) mean
        (2) median
        (3) minmax
        (4) stdminmax

        -- binned, vs step: 0.025 km/s
                   dep step: 0.5 km or as in dep_int
        (5) mode (histogram)

        -- not interpolated
        (6) bestmisfit   - min misfit
        """
        singlemodels = dict()

        # if dep_int is None:
        #     # interpolate depth to 0.5 km bins.
        #     dep_int = np.linspace(0, 100, 201)

        vss_int, deps_int = ModelMatrix.get_interpmodels(models, dep_int)

        # (1) mean, (2) median
        mean = np.mean(vss_int, axis=0)
        median = np.median(vss_int, axis=0)

        # (3) minmax
        minmax = np.array((np.min(vss_int, axis=0), np.max(vss_int, axis=0))).T

        # (4) stdminmax
        stdmodel = np.std(vss_int, axis=0)
        stdminmodel = mean - stdmodel
        stdmaxmodel = mean + stdmodel

        stdminmax = np.array((stdminmodel, stdmaxmodel)).T

        # (5) mode from histogram
        vss_flatten = vss_int.flatten()
        vsbins = int((vss_flatten.max() - vss_flatten.min()) / 0.025)
        # in PlotFromStorage posterior_models2d
        data = np.histogram2d(vss_int.flatten(), deps_int.flatten(),
                              bins=(vsbins, dep_int))
        bins, vs_bin, dep_bin = np.array(data).T
        vs_center = (vs_bin[:-1] + vs_bin[1:]) / 2.
        dep_center = (dep_bin[:-1] + dep_bin[1:]) / 2.
        vs_mode = vs_center[np.argmax(bins.T, axis=1)]
        mode = (vs_mode, dep_center)

        # (6) bestmisfit - min misfit
        if misfits is not None:
            ind = np.argmin(misfits)
            _, vs_best, dep_best = Model.get_stepmodel(models[ind])

            singlemodels['minmisfit'] = (vs_best, dep_best)

        # add models to dictionary
        singlemodels['mean'] = (mean, dep_int)
        singlemodels['median'] = (median, dep_int)
        singlemodels['minmax'] = (minmax.T, dep_int)
        singlemodels['stdminmax'] = (stdminmax.T, dep_int)
        singlemodels['mode'] = mode

        return singlemodels

    @staticmethod
    def get_weightedvalues(weights, models=None, likes=None, misfits=None,
                           noiseparams=None):
        """
        Return weighted matrix of models, misfits and noiseparams, and weighted
        vectors of likelihoods.

        Basically just repeats values, as given by weights.
        """
        weights = np.array(weights, dtype=int)
        wlikes, wmisfits, wmodels, wnoise = (None, None, None, None)

        if likes is not None:
            wlikes = np.repeat(likes, weights)

        if misfits is not None:
            if type(misfits[0]) in [int, float, np.float64]:
                wmisfits = np.repeat(misfits, weights)
            else:
                wmisfits = np.ones((np.sum(weights), misfits[0].size)) * np.nan
                n = 0
                for i, misfit in enumerate(misfits):
                    for rep in range(weights[i]):
                        wmisfits[n] = misfit
                        n += 1

        if models is not None:
            wmodels = np.ones((np.sum(weights), models[0].size)) * np.nan

            n = 0
            for i, model in enumerate(models):
                for rep in range(weights[i]):
                    wmodels[n] = model
                    n += 1

        if noiseparams is not None:
            wnoise = np.ones((np.sum(weights), noiseparams[0].size)) * np.nan

            n = 0
            for i, noisepars in enumerate(noiseparams):
                for rep in range(weights[i]):
                    wnoise[n] = noisepars
                    n += 1

        return wmodels, wlikes, wmisfits, wnoise
