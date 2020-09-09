# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import logging

# from scipy.optimize import minimize
import skquant.opt as skq
from SQSnobFit import optset
import numpy as np

from qiskit.aqua.components.optimizers import Optimizer

logger = logging.getLogger(__name__)


class IMFIL(Optimizer):
    """Constrained Optimization By Linear Approximation algorithm.

    Uses scipy.optimize.minimize COBYLA
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """

    CONFIGURATION = {
        'name': 'IMFIL',
        'description': 'IMFIL Optimizer',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'cobyla_schema',
            'type': 'object',
            'properties': {
                'maxiter': {
                    'type': 'integer',
                    'default': 1000
                },
                'disp': {
                    'type': 'boolean',
                    'default': False
                },
                'rhobeg': {
                    'type': 'number',
                    'default': 1.0
                },
                'tol': {
                    'type': ['number', 'null'],
                    'default': None
                }
            },
            'additionalProperties': False
        },
        'support_level': {
            'gradient': Optimizer.SupportLevel.ignored,
            'bounds': Optimizer.SupportLevel.required,
            'initial_point': Optimizer.SupportLevel.required
        },
        'options': ['maxiter', 'disp', 'rhobeg'],
        'optimizer': ['local']
    }

    def __init__(self, maxiter=10000, disp=False, rhobeg=1.0, tol=None):
        """
        Constructor.

        For details, please refer to
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html.

        Args:
            maxiter (int): Maximum number of function evaluations.
            disp (bool): Set to True to print convergence messages.
            rhobeg (float): Reasonable initial changes to the variables.
            tol (float): Final accuracy in the optimization (not precisely guaranteed).
                         This is a lower bound on the size of the trust region.
        """
        super().__init__()
        self._maxiter = maxiter

    def get_support_level(self):
        """ return support level dictionary """
        return {
            'gradient': Optimizer.SupportLevel.ignored,
            'bounds': Optimizer.SupportLevel.supported,
            'initial_point': Optimizer.SupportLevel.supported
        }

    def optimize(self, num_vars, objective_function, gradient_function=None, variable_bounds=None, initial_point=None):
        super().optimize(num_vars, objective_function, gradient_function, variable_bounds, initial_point)
        # variable_bounds = []
        # for _ in range(len(initial_point)):
        #     variable_bounds.append([-2*np.pi,2*np.pi])
        # variable_bounds = np.array(variable_bounds, dtype=float)
        res, history = skq.minimize(func=objective_function, x0=initial_point,
                                    bounds=variable_bounds, budget=self._maxiter,
                                    method="imfil")#,options=options)

        return res.optpar, res.optval, len(history)