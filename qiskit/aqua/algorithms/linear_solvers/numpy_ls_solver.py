# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""The Numpy LinearSystem algorithm (classical)."""

from typing import List, Union, Dict, Any
import logging
import warnings
import numpy as np

from qiskit.aqua.algorithms import ClassicalAlgorithm

logger = logging.getLogger(__name__)


class NumPyLSsolver(ClassicalAlgorithm):
    r"""
    The Numpy LinearSystem algorithm (classical).

    This linear system solver computes the eigenvalues of a complex-valued square
    matrix :math:`A` of dimension :math:`n \times n` and the solution to the systems of linear
    equations defined by :math:`A\overrightarrow{x}=\overrightarrow{b}` with input vector
    :math:`\overrightarrow{b}`.

    This is a classical counterpart to the :class:`HHL` algorithm.
    """

    def __init__(self, matrix: Union[List[List[float]], np.ndarray],
                 vector: Union[List[float], np.ndarray]) -> None:
        """
        Args:
            matrix: The input matrix of linear system of equations
            vector: The input vector of linear system of equations
        """
        super().__init__()
        self._matrix = matrix
        self._vector = vector
        self._ret = {}  # type: Dict[str, Any]

    def _solve(self):
        self._ret['eigvals'] = np.linalg.eig(self._matrix)[0]
        self._ret['solution'] = list(np.linalg.solve(self._matrix, self._vector))

    def _run(self):
        """
        Run the algorithm to compute eigenvalues and solution.
        Returns:
            dict: Dictionary of results
        """
        self._solve()
        return self._ret


class ExactLSsolver(NumPyLSsolver):
    """
    The deprecated Exact LinearSystem algorithm.
    """

    def __init__(self, matrix: Union[List[List[float]], np.ndarray],
                 vector: Union[List[float], np.ndarray]) -> None:
        warnings.warn('Deprecated class {}, use {}.'.format('ExactLSsolver', 'NumPyLSsolver'),
                      DeprecationWarning)
        super().__init__(matrix, vector)
