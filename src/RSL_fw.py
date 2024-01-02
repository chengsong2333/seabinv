# #############################
#
# 
# Cheng Song   (songcheng@snu.ac.kr)
#
#
# #############################

import numpy as np

class RSL(object):
    """Forward modeling of relative sea level.
    """

    def __init__(self, obsx, ref):
        self.obsx = obsx
        self.kmax = obsx.size
        self.ref = ref

        self.modelparams = {
            'mode': 1,  # mode, 1 fundamental, 2 first higher
            'flsph': 0  # flat earth model
            }

    def set_modelparams(self, **mparams):
        self.modelparams.update(mparams)

    def get_rwetags(self, ref):
        if ref == 'rsl':
            return (1, 1)
        else:
            tagerror = "Reference is not available %s" % ref
            raise ReferenceError(tagerror)

    def run_model(self, x_age, y_rsl, **params):
        """ The forward model will be run with the parameters below
        """
        x_obs = self.obsx
        y_obs = np.interp(x_obs, x_age, y_rsl)

        return x_obs, y_obs
