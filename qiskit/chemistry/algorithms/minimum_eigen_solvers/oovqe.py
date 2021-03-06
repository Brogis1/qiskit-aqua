# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Orbital-Optimized Variational Quantum Eigensolver algorithm."""

from typing import Optional, List, Callable, Union, Tuple
import logging
import copy
import numpy as np
from scipy.linalg import expm

from qiskit import QuantumCircuit
from qiskit.providers import BaseBackend
from qiskit.aqua import QuantumInstance, AquaError
from qiskit.aqua.operators import (OperatorBase, ExpectationBase, LegacyBaseOperator)
from qiskit.aqua.components.optimizers import Optimizer
from qiskit.aqua.components.variational_forms import VariationalForm
from qiskit.aqua.algorithms.minimum_eigen_solvers.vqe import VQE, VQEResult
from qiskit.chemistry import QMolecule

logger = logging.getLogger(__name__)

# disable check for var_forms, optimizer setter because of pylint bug
# pylint: disable=no-member


class OOVQEResult(VQEResult):
    """Orbital-Optimized VQE Result Class."""

    @property
    def optimal_point_orbitals(self) -> List[float]:
        """Returns optimal point orbitals.

        Returns:
            optimal point orbitals.
        """
        return self.get('optimal_point_orbitals')

    @optimal_point_orbitals.setter
    def optimal_point_orbitals(self, value: List[float]) -> None:
        """Sets optimal point orbitals.

        Args:
            value: optimal point orbitals.
        """
        self.data['optimal_point_orbitals'] = value


class OOVQE(VQE):
    r"""
    The Variational Quantum Eigensolver algorithm enhanced with the Orbital Optimization.
    The core of the approach resides in the optimization of orbitals through the
    AO-to-MO coefficients matrix C. In the usual VQE, the latter remains constant throughout
    the simulation. Here, its elements are modified according to C=Ce^(-kappa) where kappa is
    an anti-hermitian matrix. This transformation preserves the spectrum but modifies the
    amplitudes of the ground state of given operator such that in the end a given ansatz
    can be closest to that ground state, producing larger overlap and lower eigenvalue than
    conventional VQE.  Kappa is parametrized and optimized inside the OOVQE in the same way as
    the gate angles. Therefore, at each step of OOVQE the coefficient matrix C is modified and
    the operator is recomputed, unlike usual VQE where operator remains constant.
    Iterative OO refers to optimization in two steps, first the wavefunction and then the
    orbitals. It allows for faster optimization as the operator is not recomputed when
    wavefunction is optimized. It is recommended to use the iterative method on real device/qasm
    simulator with noise to facilitate the convergence of the classical optimizer.
    Currently, this method has been tested with PySCF and Pyquante2 drivers.
    For more details of this method refer to: https://aip.scitation.org/doi/10.1063/1.5141835
    """
    def __init__(self,
                 operator: Optional[Union[OperatorBase, LegacyBaseOperator]] = None,
                 var_form: Optional[Union[QuantumCircuit, VariationalForm]] = None,
                 optimizer: Optional[Optimizer] = None,
                 initial_point: Optional[np.ndarray] = None,
                 expectation: Optional[ExpectationBase] = None,
                 include_custom: bool = False,
                 max_evals_grouped: int = 1,
                 aux_operators: Optional[List[Optional[Union[OperatorBase,
                                                             LegacyBaseOperator]]]] = None,
                 callback: Optional[Callable[[int, np.ndarray, float, float], None]] = None,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend]] = None,
                 orbital_rotation: Optional['OrbitalRotation'] = None,
                 core: Optional[LegacyBaseOperator] = None,
                 qmolecule: Optional[QMolecule] = None,
                 bounds: Optional[np.ndarray] = None,
                 iterative_oo: bool = False,
                 iterative_oo_iterations: int = 3,
                 ) -> None:
        """

        Args:
            operator: Qubit operator of the Hamiltonian.
            var_form: A parameterized variational form (ansatz).
            optimizer: A classical optimizer.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` then VQE will look to the variational form for a
                preferred point and if not will simply compute a random one.
            expectation: The Expectation converter for taking the average value of the
                Observable over the var_form state function. When ``None`` (the default) an
                :class:`~qiskit.aqua.operators.expectations.ExpectationFactory` is used to select
                an appropriate expectation based on the operator and backend. When using Aer
                qasm_simulator backend, with paulis, it is however much faster to leverage custom
                Aer function for the computation but, although VQE performs much faster
                with it, the outcome is ideal, with no shot noise, like using a state vector
                simulator. If you are just looking for the quickest performance when choosing Aer
                qasm_simulator and the lack of shot noise is not an issue then set `include_custom`
                parameter here to ``True`` (defaults to ``False``).
            include_custom: When `expectation` parameter here is None setting this to ``True`` will
                allow the factory to include the custom Aer pauli expectation.
            max_evals_grouped: Max number of evaluations performed simultaneously. Signals the
                given optimizer that more than one set of parameters can be supplied so that
                potentially the expectation values can be computed in parallel. Typically this is
                possible when a finite difference gradient is used by the optimizer such that
                multiple points to compute the gradient can be passed and if computed in parallel
                improve overall execution time.
            aux_operators: Optional list of auxiliary operators to be evaluated with the eigenstate
                of the minimum eigenvalue main result and their expectation values returned.
                For instance in chemistry these can be dipole operators, total particle count
                operators so we can get values for these at the ground state.
            callback: a callback that can access the intermediate data during the optimization.
                Four parameter values are passed to the callback as follows during each evaluation
                by the optimizer for its current set of parameters as it works towards the minimum.
                These are: the evaluation count, the optimizer parameters for the
                variational form, the evaluated mean and the evaluated standard deviation.
            quantum_instance: :class:`~qiskit.aqua.QuantumInstance` that the algorithm is
                 executed on.
            orbital_rotation: instance of
                :class:`~qiskit.chemistry.algorithms.minimum_eigen_solvers.OrbitalRotation` class
                that creates the matrices that rotate the orbitals needed to produce the rotated
                MO coefficients C as C = C0 * exp(-kappa).
            core: instance of the :class:`~qiskit.chemistry.core.Hamiltonian` class to
                 make new qubit operator after the orbital rotation using the data stored in
                 qmolecule.
            qmolecule: instance of the :class:`~qiskit.chemistry.QMolecule` class which has methods
                needed to recompute one-/two-electron/dipole integrals after orbital rotation
                (C = C0 * exp(-kappa)).
            bounds: bounds for variational form and orbital rotation
                 parameters given to a classical optimizer.
            iterative_oo: when ``True`` optimize first the variational form and then the
                orbitals, iteratively. Otherwise, the wavefunction ansatz and orbitals are
                optimized simultaneously.
            iterative_oo_iterations: number of iterations in the iterative procedure,
                set larger to be sure to converge to the global minimum.

        Raises:
            AquaError: if the number of orbital optimization iterations is less or equal to zero.
        """

        super().__init__(operator=operator,
                         var_form=var_form,
                         optimizer=optimizer,
                         initial_point=initial_point,
                         expectation=expectation,
                         include_custom=include_custom,
                         max_evals_grouped=max_evals_grouped,
                         aux_operators=aux_operators,
                         callback=callback,
                         quantum_instance=quantum_instance,
                         )
        self._qmolecule_rotated = None
        self._fixed_wavefunction_params = None
        if aux_operators is None:
            self._aux_operators = []
        else:
            self._aux_operators = [aux_operators] if not isinstance(aux_operators, list) else \
                aux_operators
        logger.info(self.print_settings())
        self._core = core
        self._qmolecule = qmolecule
        if orbital_rotation is None:
            self._orbital_rotation = OrbitalRotation(num_qubits=self.var_form.num_qubits,
                                                     core=self._core, qmolecule=self._qmolecule)
        self._num_parameters_oovqe = None
        if initial_point is None:
            self._set_initial_point()
        self._bounds = bounds
        if self._bounds is None:
            self._set_bounds(self._orbital_rotation.parameter_bound_value)
        self._iterative_oo = iterative_oo
        self._iterative_oo_iterations = iterative_oo_iterations
        if self._iterative_oo_iterations < 1:
            raise AquaError('Please set iterative_oo_iterations parameter to a positive number,'
                            ' got {} instead'.format(self._iterative_oo_iterations))

        # copies to overcome incompatibilities with error checks in VQAlgorithm class
        self.var_form_num_parameters = self.var_form.num_parameters
        self.var_form_bounds = copy.copy(self.var_form._bounds)

    def _run(self) -> VQEResult:
        """Run the algorithm to compute the minimum eigenvalue.

        Returns:
            The result of the VQE algorithm as ``VQEResult``.

        Raises:
            AquaError: Wrong setting of operator and backend.
        """

        if self.operator is None:  # type: ignore
            raise AquaError("The operator was never provided.")
        self._check_operator_varform()
        self._quantum_instance.circuit_summary = True
        self._eval_count = 0

        # initial orbital rotation starting point is provided
        if self._orbital_rotation.matrix_a is not None and self._orbital_rotation.matrix_b is not \
                None:
            self._qmolecule_rotated = copy.copy(self._qmolecule)
            OOVQE._rotate_orbitals_in_qmolecule(self._qmolecule_rotated, self._orbital_rotation)
            self._operator, _ = self._core.run(self._qmolecule_rotated)
            logger.info(
                '\n\nSetting the initial value for OO matrices and rotating Hamiltonian \n')
            logger.info('Optimising  Orbital Coefficient Rotation Alpha: \n%s',
                        repr(self._orbital_rotation.matrix_a))
            logger.info('Optimising  Orbital Coefficient Rotation Beta: \n%s',
                        repr(self._orbital_rotation.matrix_b))

        # save the original number of parameters as we modify their number to bypass the
        # error checks that are not tailored to OOVQE

        total_time = 0
        # iterative method
        if self._iterative_oo:
            for _ in range(self._iterative_oo_iterations):

                # optimize wavefunction ansatz
                self.var_form._num_parameters = self.var_form_num_parameters
                if isinstance(self.operator, LegacyBaseOperator):  # type: ignore
                    self.operator = self.operator.to_opflow()  # type: ignore
                self.var_form._bounds = self.var_form_bounds
                vqresult_wavefun = self.find_minimum(
                    initial_point=self.initial_point[:self.var_form_num_parameters],
                    var_form=self.var_form,
                    cost_fn=self._energy_evaluation,
                    optimizer=self.optimizer)
                self.initial_point[:self.var_form_num_parameters] = vqresult_wavefun.optimal_point

                # optimize orbitals
                self.var_form._bounds = self._bound_oo
                self.var_form._num_parameters = self._orbital_rotation.num_parameters
                self._fixed_wavefunction_params = vqresult_wavefun.optimal_point
                vqresult = self.find_minimum(
                    initial_point=self.initial_point[self.var_form_num_parameters:],
                    var_form=self.var_form,
                    cost_fn=self._energy_evaluation_oo,
                    optimizer=self.optimizer)
                self.initial_point[self.var_form_num_parameters:] = vqresult.optimal_point
                total_time += vqresult.optimizer_time
        else:
            # simultaneous method (ansatz and orbitals are optimized at the same time)
            self.var_form._bounds = self._bounds
            self.var_form._num_parameters = len(self._bounds)

            vqresult = self.find_minimum(initial_point=self.initial_point,
                                         var_form=self.var_form,
                                         cost_fn=self._energy_evaluation_oo,
                                         optimizer=self.optimizer)
            total_time += vqresult.optimizer_time

        # write original number of parameters to avoid errors due to parameter number mismatch
        self.var_form._num_parameters = self.var_form_num_parameters

        # save the results
        self._ret = {}
        self._ret['num_optimizer_evals'] = vqresult.optimizer_evals
        self._ret['min_val'] = vqresult.optimal_value
        if self._iterative_oo:
            self._ret['opt_params'] = self.initial_point[self.var_form_num_parameters:]
            self._ret['opt_params_orbitals'] = self.initial_point[:self.var_form_num_parameters]
        else:
            self._ret['opt_params'] = vqresult.optimal_point[:self.var_form_num_parameters]
            self._ret['opt_params_orbitals'] = vqresult.optimal_point[self.var_form_num_parameters:]

        self._ret['eval_time'] = total_time
        self._ret['opt_params_dict'] = vqresult.optimal_parameters

        if self._ret['num_optimizer_evals'] is not None and \
                self._eval_count >= self._ret['num_optimizer_evals']:
            self._eval_count = self._ret['num_optimizer_evals']
        self._eval_time = self._ret['eval_time']
        logger.info('Optimization complete in %s seconds.\nFound opt_params %s in %s evals',
                    self._eval_time, self._ret['opt_params'], self._eval_count)
        self._ret['eval_count'] = self._eval_count

        result = OOVQEResult()
        result.combine(vqresult)
        result.optimal_point_orbitals = self._ret["opt_params_orbitals"]
        result.eigenvalue = vqresult.optimal_value + 0j
        # record all parameters (wavefunction and orbitals) to overcome error checks
        _ret_temp_params = copy.copy(self._ret['opt_params'])
        self._ret['opt_params'] = self._ret['opt_params'][:self.var_form_num_parameters]
        result.eigenstate = self.get_optimal_vector()
        self._ret['opt_params'] = _ret_temp_params

        self._ret['energy'] = self.get_optimal_cost()
        self._ret['eigvals'] = np.asarray([self._ret['energy']])
        self._ret['eigvecs'] = np.asarray([result.eigenstate])

        if self.aux_operators:
            self._eval_aux_ops()
            # TODO remove when ._ret is deprecated
            result.aux_operator_eigenvalues = self._ret['aux_ops'][0]

        result.cost_function_evals = self._eval_count

        return result

    def _set_bounds(self,
                    bounds_var_form_val: tuple = (-2 * np.pi, 2 * np.pi),
                    bounds_oo_val: tuple = (-2 * np.pi, 2 * np.pi)) -> None:
        """ Initializes the array of bounds of wavefunction and OO parameters.
        Args:
            bounds_var_form_val: pair of bounds between which the optimizer confines the
                                 values of wavefunction parameters.
            bounds_oo_val: pair of bounds between which the optimizer confines the values of
                           OO parameters.
        Raises:
            AquaError: Instantiate OrbitalRotation class and provide it to the
                       orbital_rotation keyword argument
        """
        self._bounds = []
        bounds_var_form = [bounds_var_form_val for _ in range(self.var_form.num_parameters)]
        self._bound_oo = \
            [bounds_oo_val for _ in range(self._orbital_rotation.num_parameters)]  # type: List
        self._bounds = bounds_var_form + self._bound_oo
        self._bounds = np.array(self._bounds)

    def _set_initial_point(self, initial_pt_scalar: float = 1e-1) -> None:
        """ Initializes the initial point for the algorithm if the user does not provide his own.
        Args:
            initial_pt_scalar: value of the initial parameters for wavefunction and orbital rotation
        """
        self._num_parameters_oovqe = \
            self.var_form._num_parameters + self._orbital_rotation.num_parameters
        self._initial_point = [initial_pt_scalar for _ in range(self._num_parameters_oovqe)]

    def _energy_evaluation_oo(self, parameters: np.ndarray) -> Union[float, List[float]]:
        """
        Evaluate energy at given parameters for the variational form and parameters for
        given rotation of orbitals.

        Args:
            parameters: parameters for variational form and orbital rotations.

        Returns:
            energy of the hamiltonian of each parameter.

        Raises:
            AquaError: Instantiate OrbitalRotation class and provide it to the
                       orbital_rotation keyword argument
        """

        # slice parameter lists
        if self._iterative_oo:
            parameters_var_form = self._fixed_wavefunction_params
            parameters_orb_rot = parameters
        else:
            parameters_var_form = parameters[:self.var_form_num_parameters]
            parameters_orb_rot = parameters[self.var_form_num_parameters:]

        logger.info('Parameters of wavefunction are: \n%s', repr(parameters_var_form))
        logger.info('Parameters of orbital rotation are: \n%s', repr(parameters_orb_rot))

        # rotate the orbitals
        if self._orbital_rotation is None:
            raise AquaError('Instantiate OrbitalRotation class and provide it to the '
                            'orbital_rotation keyword argument')

        self._orbital_rotation.orbital_rotation_matrix(parameters_orb_rot)

        # preserve original qmolecule and create a new one with rotated orbitals
        self._qmolecule_rotated = copy.copy(self._qmolecule)
        OOVQE._rotate_orbitals_in_qmolecule(self._qmolecule_rotated, self._orbital_rotation)

        # construct the qubit operator
        algo_input = self._core.run(self._qmolecule_rotated)
        self.operator = algo_input[0]
        if isinstance(self.operator, LegacyBaseOperator):
            self.operator = self.operator.to_opflow()
        logger.debug('Orbital rotation parameters of matrix U at evaluation %d returned'
                     '\n %s', self._eval_count, repr(self._orbital_rotation.matrix_a))

        self.var_form._num_parameters = self.var_form_num_parameters

        # compute the energy on given state
        mean_energy = self._energy_evaluation(parameters=parameters_var_form)

        return mean_energy

    @staticmethod
    def _rotate_orbitals_in_qmolecule(qmolecule: QMolecule,
                                      orbital_rotation: 'OrbitalRotation') -> None:
        """
        Rotates the orbitals by applying a modified a anti-hermitian matrix
        (orbital_rotation.matrix_a) onto the MO coefficients matrix and recomputes all the
        quantities dependent on the MO coefficients. Be aware that qmolecule is modified
        when this executes.
        Args:
            qmolecule: instance of QMolecule class
            orbital_rotation: instance of OrbitalRotation class
        """

        # 1 and 2 electron integrals (required) from AO to MO basis
        qmolecule.mo_coeff = np.matmul(qmolecule.mo_coeff,
                                       orbital_rotation.matrix_a)
        qmolecule.mo_onee_ints = qmolecule.oneeints2mo(qmolecule.hcore,
                                                       qmolecule.mo_coeff)
        # support for unrestricted spins
        if qmolecule.mo_coeff_b is not None:
            qmolecule.mo_coeff_b = np.matmul(qmolecule.mo_coeff_b,
                                             orbital_rotation.matrix_b)
            qmolecule.mo_onee_ints_b = qmolecule.oneeints2mo(qmolecule.hcore,
                                                             qmolecule.mo_coeff)

        qmolecule.mo_eri_ints = qmolecule.twoeints2mo(qmolecule.eri,
                                                      qmolecule.mo_coeff)
        if qmolecule.mo_coeff_b is not None:
            mo_eri_b = qmolecule.twoeints2mo(qmolecule.eri,
                                             qmolecule.mo_coeff_b)
            norbs = qmolecule.mo_coeff.shape[0]
            qmolecule.mo_eri_ints_bb = mo_eri_b.reshape(norbs, norbs, norbs, norbs)
            qmolecule.mo_eri_ints_ba = qmolecule.twoeints2mo_general(
                qmolecule.eri, qmolecule.mo_coeff_b, qmolecule.mo_coeff_b, qmolecule.mo_coeff,
                qmolecule.mo_coeff)
            qmolecule.mo_eri_ints_ba = qmolecule.mo_eri_ints_ba.reshape(norbs, norbs,
                                                                        norbs, norbs)
        # dipole integrals (if available) from AO to MO
        if qmolecule.x_dip_ints is not None:
            qmolecule.x_dip_mo_ints = qmolecule.oneeints2mo(qmolecule.x_dip_ints,
                                                            qmolecule.mo_coeff)
            qmolecule.y_dip_mo_ints = qmolecule.oneeints2mo(qmolecule.y_dip_ints,
                                                            qmolecule.mo_coeff)
            qmolecule.z_dip_mo_ints = qmolecule.oneeints2mo(qmolecule.z_dip_ints,
                                                            qmolecule.mo_coeff)
        # support for unrestricted spins
        if qmolecule.mo_coeff_b is not None and qmolecule.x_dip_ints is not None:
            qmolecule.x_dip_mo_ints_b = qmolecule.oneeints2mo(qmolecule.x_dip_ints,
                                                              qmolecule.mo_coeff_b)
            qmolecule.y_dip_mo_ints_b = qmolecule.oneeints2mo(qmolecule.y_dip_ints,
                                                              qmolecule.mo_coeff_b)
            qmolecule.z_dip_mo_ints_b = qmolecule.oneeints2mo(qmolecule.z_dip_ints,
                                                              qmolecule.mo_coeff_b)


class OrbitalRotation:
    r"""
    Class that regroups methods for creation of matrices that rotate the MOs.
    It allows to create the unitary matrix U = exp(-kappa) that is parameterized with kappa's
    elements. The parameters are the off-diagonal elements of the anti-hermitian matrix kappa.
    """

    def __init__(self,
                 num_qubits: int,
                 core: Optional[LegacyBaseOperator] = None,
                 qmolecule: Optional[QMolecule] = None,
                 orbital_rotations: list = None,
                 orbital_rotations_beta: list = None,
                 parameters: list = None,
                 parameter_bounds: list = None,
                 parameter_initial_value: float = 0.1,
                 parameter_bound_value: Tuple[float, float] = (-2 * np.pi, 2 * np.pi)) -> None:

        """
        Args:
            num_qubits: number of qubits necessary to simulate a particular system.
            core: instance of the :class:`~qiskit.chemistry.core.Hamiltonian` class to
                 make new qubit operator after the orbital rotation using the data stored in
                 qmolecule.
            qmolecule: instance of the :class:`~qiskit.chemistry.QMolecule` class which has methods
                needed to recompute one-/two-electron/dipole integrals after orbital rotation
                (C = C0 * exp(-kappa)).
            orbital_rotations: list of alpha orbitals that are rotated (i.e. [[0,1], ...] the
                0-th orbital is rotated with 1-st, which corresponds to non-zero entry 01 of
                the matrix kappa).
            orbital_rotations_beta: list of beta orbitals that are rotated.
            parameters: orbital rotation parameter list of matrix elements that rotate the MOs,
                each associated to a pair of orbitals that are rotated
                (non-zero elements in matrix kappa), or elements in the orbital_rotation(_beta)
                lists.
            parameter_bounds: parameter bounds
            parameter_initial_value: initial value for all the parameters.
            parameter_bound_value: value for the bounds on all the parameters
        """

        self._num_qubits = num_qubits
        self._core = core
        self._qmolecule = qmolecule
        self._orbital_rotations = orbital_rotations
        self._orbital_rotations_beta = orbital_rotations_beta
        self._parameter_initial_value = parameter_initial_value
        self._parameter_bound_value = parameter_bound_value

        self._parameters = parameters
        if self._parameters is None:
            # todo: we overwrite the values orbital_rotations values by calling this method.
            self._create_parameter_list()
        self._num_parameters = len(self._parameters)

        self._parameter_bounds = parameter_bounds
        if self._parameter_bounds is None:
            self._create_parameter_bounds()

        # todo: if core is None then an exception will be raised on the next line
        self._freeze_core = self._core._freeze_core
        if self._core is not None:
            self._core_list = self._qmolecule.core_orbitals if self._freeze_core else None
        # todo: same here, if core is None, then there's a problem on the next line
        if self._core._two_qubit_reduction is True:
            self._dim_kappa_matrix = int((self._num_qubits + 2) / 2)
        else:
            self._dim_kappa_matrix = int(self._num_qubits / 2)

        self._check_for_errors()
        self._matrix_a = None
        self._matrix_b = None

    def _check_for_errors(self) -> None:
        """ Checks for errors such as incorrect number of parameters and indices of orbitals. """

        # number of parameters check
        if self._orbital_rotations_beta is None and self._orbital_rotations is not None:
            if len(self._orbital_rotations) != len(self._parameters):
                raise AquaError('Please specify same number of params ({}) as there are '
                                'orbital rotations ({})'.format(len(self._parameters),
                                                                len(self._orbital_rotations)))
        elif self._orbital_rotations_beta is not None and self._orbital_rotations is not None:
            if len(self._orbital_rotations) + len(self._orbital_rotations_beta) != len(
                    self._parameters):
                raise AquaError('Please specify same number of params ({}) as there are '
                                'orbital rotations ({})'.format(len(self._parameters),
                                                                len(self._orbital_rotations)))
        # indices of rotated orbitals check
        for exc in self._orbital_rotations:
            if exc[0] > (self._dim_kappa_matrix - 1):
                raise AquaError('You specified entries that go outside '
                                'the orbital rotation matrix dimensions {}, '.format(exc[0]))
            if exc[1] > (self._dim_kappa_matrix - 1):
                raise AquaError('You specified entries that go outside '
                                'the orbital rotation matrix dimensions {}'.format(exc[1]))
        if self._orbital_rotations_beta is not None:
            for exc in self._orbital_rotations_beta:
                if exc[0] > (self._dim_kappa_matrix - 1):
                    raise AquaError('You specified entries that go outside '
                                    'the orbital rotation matrix dimensions {}'.format(exc[0]))
                if exc[1] > (self._dim_kappa_matrix - 1):
                    raise AquaError('You specified entries that go outside '
                                    'the orbital rotation matrix dimensions {}'.format(exc[1]))

    def _create_parameter_list(self) -> None:
        """ Initializes the entries of matrix kappa and the initial values. """

        if self._core._two_qubit_reduction:
            half_as = int((self._num_qubits + 2) / 2)
        else:
            half_as = int(self._num_qubits / 2)

        orbital_rotations = []
        parameters = []

        for i in range(half_as):
            for j in range(half_as):
                if i < j:
                    orbital_rotations.append([i, j])
                    parameters.append(self._parameter_initial_value)

        self._orbital_rotations = orbital_rotations
        self._parameters = parameters

    def _create_parameter_bounds(self) -> None:
        """ Create bounds for parameters. """
        self._parameter_bounds = []
        for _ in range(self._num_parameters):
            self._parameter_bounds.append(self._parameter_bound_value)

    def orbital_rotation_matrix(self, parameters: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Creates 2 matrices K_alpha, K_beta that rotate the orbitals through MO coefficient
        C_alpha = C_RHF * U_alpha where U = e^(K_alpha), similarly for beta orbitals. """

        self._parameters = parameters
        k_matrix_alpha = np.zeros((self._dim_kappa_matrix, self._dim_kappa_matrix))
        k_matrix_beta = np.zeros((self._dim_kappa_matrix, self._dim_kappa_matrix))

        # allows to selectively rotate pairs of orbitals
        if self._orbital_rotations_beta is None:
            for i, exc in enumerate(self._orbital_rotations):
                k_matrix_alpha[exc[0]][exc[1]] = self._parameters[i]
                k_matrix_alpha[exc[1]][exc[0]] = -self._parameters[i]
                k_matrix_beta[exc[0]][exc[1]] = self._parameters[i]
                k_matrix_beta[exc[1]][exc[0]] = -self._parameters[i]
        else:
            for i, exc in enumerate(self._orbital_rotations):
                k_matrix_alpha[exc[0]][exc[1]] = self._parameters[i]
                k_matrix_alpha[exc[1]][exc[0]] = -self._parameters[i]

            for j, exc in enumerate(self._orbital_rotations_beta):
                k_matrix_beta[exc[0]][exc[1]] = self._parameters[j + len(self._orbital_rotations)]
                k_matrix_beta[exc[1]][exc[0]] = -self._parameters[j + len(self._orbital_rotations)]

        if self._freeze_core:
            half_as = int(self._dim_kappa_matrix + len(self._core_list))
            k_matrix_alpha_full = np.zeros((half_as, half_as))
            k_matrix_beta_full = np.zeros((half_as, half_as))
            # rotating only non-frozen part of orbitals
            dim_full_k = k_matrix_alpha_full.shape[0]  # pylint: disable=unsubscriptable-object

            if self._core_list is None:
                raise AquaError('Give _core_list, the list of molecular spatial orbitals that are '
                                'frozen (e.g. [0] for the 1s or [0,1] for respectively Li2 or N2 '
                                'for example).')
            lower = len(self._core_list)
            upper = dim_full_k
            k_matrix_alpha_full[lower:upper, lower:upper] = k_matrix_alpha
            k_matrix_beta_full[lower:upper, lower:upper] = k_matrix_beta
            self._matrix_a = expm(k_matrix_alpha_full)
            self._matrix_b = expm(k_matrix_beta_full)
        else:
            self._matrix_a = expm(k_matrix_alpha)
            self._matrix_b = expm(k_matrix_beta)

        return self._matrix_a, self._matrix_b

    @property
    def matrix_a(self) -> np.ndarray:
        """Returns matrix A."""
        return self._matrix_a

    @property
    def matrix_b(self) -> np.ndarray:
        """Returns matrix B. """
        return self._matrix_b

    @property
    def num_parameters(self) -> int:
        """Returns the number of parameters."""
        return self._num_parameters

    @property
    def parameter_bound_value(self) -> Tuple[float, float]:
        """Returns a value for the bounds on all the parameters."""
        return self._parameter_bound_value
