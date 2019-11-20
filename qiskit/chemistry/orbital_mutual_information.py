# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from __future__ import division
import matplotlib.pyplot as plt
import networkx as nx

from qiskit.chemistry.core import Hamiltonian, TransformationType, QubitMappingType
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD
from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock
from qiskit.quantum_info import Pauli
from qiskit.chemistry.bksf import bksf_mapping
from qiskit.aqua import set_qiskit_aqua_logging, Operator

import itertools
import logging
import functools
import numpy as np


set_qiskit_aqua_logging(1)
logger = logging.getLogger(__name__)


# this is to create the init_var_form

class MoleculeOperators:
    """
    Class to initialise the variational form to pass
    """

    def __init__(self, driver):
        self.driver = driver
        # self.setUp()
        # if tapering:
        #     self.tapered_qubit_op = self.test_tapered_op()
        # else:
        #     self.tapered_qubit_op = None


    def setUp(self):

        self.qmolecule = self.driver.run()
        self.core = Hamiltonian(transformation=TransformationType.FULL,  # or FULL
                                qubit_mapping=QubitMappingType.JORDAN_WIGNER,
                                two_qubit_reduction=False,
                                freeze_core=False,
                                orbital_reduction=[],
                                )
        algo_input = self.core.run(self.qmolecule)
        self.qubit_op = algo_input[0]
        self.aux_op = algo_input[1]

        self.symmetries, self.sq_paulis, self.cliffords, self.sq_list = self.qubit_op.find_Z2_symmetries()

    def test_symmetries(self):
        labels = [symm.to_label() for symm in self.symmetries]
        self.assertSequenceEqual(labels, ['ZIZIZIZI', 'ZZIIZZII'])

    def test_sq_paulis(self):
        labels = [sq.to_label() for sq in self.sq_paulis]
        self.assertSequenceEqual(labels, ['IIIIIIXI', 'IIIIIXII'])

    def test_cliffords(self):
        self.assertEqual(2, len(self.cliffords))

    def test_sq_list(self):
        self.assertSequenceEqual(self.sq_list, [1, 2])

    def test_tapered_op(self):
        # set_qiskit_chemistry_logging(logging.DEBUG)
        tapered_ops = []
        for coeff in itertools.product([1, -1], repeat=len(self.sq_list)):
            tapered_op = Operator.qubit_tapering(self.qubit_op, self.cliffords, self.sq_list, list(coeff))
            tapered_ops.append((list(coeff), tapered_op))

        smallest_eig_value = 99999999999999
        smallest_idx = -1
        for idx in range(len(tapered_ops)):
            ee = ExactEigensolver(tapered_ops[idx][1])
            eigvals = ee.run()['energies']
            curr_value = eigvals[0]
            if curr_value < smallest_eig_value:
                smallest_eig_value = curr_value
                smallest_idx = idx  # Prior knowledge of which tapered_op has ground state
            print("Lowest eigenvalue of the {}-th tapered operator (computed part) is {:.12f}".format(idx, curr_value))

        # smallest_idx = 0  # Prior knowledge of which tapered_op has ground state
        self.tapered_qubit_op = tapered_ops[smallest_idx][1]
        self.coeff_tapered_qubit_op = tapered_ops[smallest_idx][0]

        return tapered_ops[smallest_idx][1], tapered_ops[smallest_idx][0]

    def init_var_form(self, tapering = False):
        if tapering:
            # set_qiskit_chemistry_logging(logging.DEBUG)
            tapered_ops = []
            for coeff in itertools.product([1, -1], repeat=len(self.sq_list)):
                tapered_op = Operator.qubit_tapering(self.qubit_op, self.cliffords, self.sq_list, list(coeff))
                tapered_ops.append((list(coeff), tapered_op))

            smallest_eig_value = 99999999999999
            smallest_idx = -1
            for idx in range(len(tapered_ops)):
                ee = ExactEigensolver(tapered_ops[idx][1])
                eigvals = ee.run()['energies']
                curr_value = eigvals[0]
                if curr_value < smallest_eig_value:
                    smallest_eig_value = curr_value
                    smallest_idx = idx  # Prior knowledge of which tapered_op has ground state
                print("Lowest eigenvalue of the {}-th tapered operator (computed part) is {:.12f}".format(idx,
                                                                                                          curr_value))

            # smallest_idx = 0  # Prior knowledge of which tapered_op has ground state
            the_tapered_op = tapered_ops[smallest_idx][1]
            the_coeff = tapered_ops[smallest_idx][0]

            # optimizer = SLSQP(maxiter=10000, ftol=1e-7, disp=True)
            # optimizer = L_BFGS_B(maxfun=1000)
        else:
            the_tapered_op = self.qubit_op
            self.sq_list = None
            the_coeff = None
            self.symmetries = None
            self.cliffords = None

        init_state = HartreeFock(num_qubits=the_tapered_op.num_qubits,
                                 num_orbitals=self.core._molecule_info['num_orbitals'],
                                 qubit_mapping=self.core._qubit_mapping,
                                 two_qubit_reduction=self.core._two_qubit_reduction,
                                 num_particles=self.core._molecule_info['num_particles'],
                                 sq_list=self.sq_list)

        var_form = UCCSD(num_qubits=the_tapered_op.num_qubits, depth=1,
                         num_orbitals=self.core._molecule_info['num_orbitals'],
                         num_particles=self.core._molecule_info['num_particles'],
                         active_occupied=None, active_unoccupied=None,
                         initial_state=init_state,
                         qubit_mapping=self.core._qubit_mapping,
                         two_qubit_reduction=self.core._two_qubit_reduction,
                         num_time_slices=1,
                         cliffords=self.cliffords, sq_list=self.sq_list,
                         tapering_values=the_coeff, symmetries=self.symmetries,
                         )
        return var_form


# construct list of operators for the RDMs

class RDMFermionicOperator(object):
    r"""
     A set of functions to map fermionic Hamiltonians to qubit Hamiltonians.

     References:
     - E. Wigner and P. Jordan., Über das Paulische Äguivalenzverbot, \
         Z. Phys., 47:631 (1928). \
     - S. Bravyi and A. Kitaev. Fermionic quantum computation, \
         Ann. of Phys., 298(1):210–226 (2002). \
     - A. Tranter, S. Sofia, J. Seeley, M. Kaicher, J. McClean, R. Babbush, \
         P. Coveney, F. Mintert, F. Wilhelm, and P. Love. The Bravyi–Kitaev \
         transformation: Properties and applications. Int. Journal of Quantum \
         Chemistry, 115(19):1431–1441 (2015). \
     - S. Bravyi, J. M. Gambetta, A. Mezzacapo, and K. Temme, \
         arXiv e-print arXiv:1701.08213 (2017). \
     - K. Setia, J. D. Whitfield, arXiv:1712.00446 (2017)
     """

    def __init__(self, number_modes, var_form=None, parameters=None, quantum_instance=None, map_type = 'jordan_wigner', operator_mode='matrix',
                 initial_point=None, aux_operators=None, use_simulator_operator_mode = False):
        """Constructor.

        """

        self._modes = number_modes
        self._map_type = map_type
        self._a = self.mapping(map_type=self._map_type)
        # self.indices_orbitals = ind_orb_up_down
        self._operator = RDMFermionicOperator.transition_operators(orbital_indice=[0,1], a=self._a)[1]

        self._var_form = var_form
        self._parameters = parameters
        # borrowed from VQE to make a function evaluation
        if initial_point is None and var_form is not None:
            self._initial_point = var_form.preferred_init_points
        self._operator_mode = operator_mode
        self._use_simulator_operator_mode = use_simulator_operator_mode
        self._eval_count = 0
        self._quantum_instance = quantum_instance
        if aux_operators is None:
            self._aux_operators = []
        else:
            self._aux_operators = [aux_operators] if not isinstance(aux_operators, list) else aux_operators

    @property
    def modes(self):
        """Getter of modes."""
        return self._modes

    def _jordan_wigner_mode(self, n):
        """
        Jordan_Wigner mode.

        Args:
            n (int): number of modes
        """
        a = []
        for i in range(n):
            a_z = np.asarray([1] * i + [0] + [0] * (n - i - 1), dtype=np.bool)
            a_x = np.asarray([0] * i + [1] + [0] * (n - i - 1), dtype=np.bool)
            b_z = np.asarray([1] * i + [1] + [0] * (n - i - 1), dtype=np.bool)
            b_x = np.asarray([0] * i + [1] + [0] * (n - i - 1), dtype=np.bool)
            a.append((Pauli(a_z, a_x), Pauli(b_z, b_x)))
        return a

    def _parity_mode(self, n):
        """
        Parity mode.

        Args:
            n (int): number of modes
        """
        a = []
        for i in range(n):
            a_z = [0] * (i - 1) + [1] if i > 0 else []
            a_x = [0] * (i - 1) + [0] if i > 0 else []
            b_z = [0] * (i - 1) + [0] if i > 0 else []
            b_x = [0] * (i - 1) + [0] if i > 0 else []
            a_z = np.asarray(a_z + [0] + [0] * (n - i - 1), dtype=np.bool)
            a_x = np.asarray(a_x + [1] + [1] * (n - i - 1), dtype=np.bool)
            b_z = np.asarray(b_z + [1] + [0] * (n - i - 1), dtype=np.bool)
            b_x = np.asarray(b_x + [1] + [1] * (n - i - 1), dtype=np.bool)
            a.append((Pauli(a_z, a_x), Pauli(b_z, b_x)))
        return a

    def _bravyi_kitaev_mode(self, n):
        """
        Bravyi-Kitaev mode.

        Args:
            n (int): number of modes
        """

        def parity_set(j, n):
            """Computes the parity set of the j-th orbital in n modes.

            Args:
                j (int) : the orbital index
                n (int) : the total number of modes

            Returns:
                numpy.ndarray: Array of mode indexes
            """
            indexes = np.array([])
            if n % 2 != 0:
                return indexes

            if j < n / 2:
                indexes = np.append(indexes, parity_set(j, n / 2))
            else:
                indexes = np.append(indexes, np.append(
                    parity_set(j - n / 2, n / 2) + n / 2, n / 2 - 1))
            return indexes

        def update_set(j, n):
            """Computes the update set of the j-th orbital in n modes.

            Args:
                j (int) : the orbital index
                n (int) : the total number of modes

            Returns:
                numpy.ndarray: Array of mode indexes
            """
            indexes = np.array([])
            if n % 2 != 0:
                return indexes
            if j < n / 2:
                indexes = np.append(indexes, np.append(
                    n - 1, update_set(j, n / 2)))
            else:
                indexes = np.append(indexes, update_set(j - n / 2, n / 2) + n / 2)
            return indexes

        def flip_set(j, n):
            """Computes the flip set of the j-th orbital in n modes.

            Args:
                j (int) : the orbital index
                n (int) : the total number of modes

            Returns:
                numpy.ndarray: Array of mode indexes
            """
            indexes = np.array([])
            if n % 2 != 0:
                return indexes
            if j < n / 2:
                indexes = np.append(indexes, flip_set(j, n / 2))
            elif j >= n / 2 and j < n - 1:
                indexes = np.append(indexes, flip_set(j - n / 2, n / 2) + n / 2)
            else:
                indexes = np.append(np.append(indexes, flip_set(
                    j - n / 2, n / 2) + n / 2), n / 2 - 1)
            return indexes

        a = []
        # FIND BINARY SUPERSET SIZE
        bin_sup = 1
        while n > np.power(2, bin_sup):
            bin_sup += 1
        # DEFINE INDEX SETS FOR EVERY FERMIONIC MODE
        update_sets = []
        update_pauli = []

        parity_sets = []
        parity_pauli = []

        flip_sets = []

        remainder_sets = []
        remainder_pauli = []
        for j in range(n):

            update_sets.append(update_set(j, np.power(2, bin_sup)))
            update_sets[j] = update_sets[j][update_sets[j] < n]

            parity_sets.append(parity_set(j, np.power(2, bin_sup)))
            parity_sets[j] = parity_sets[j][parity_sets[j] < n]

            flip_sets.append(flip_set(j, np.power(2, bin_sup)))
            flip_sets[j] = flip_sets[j][flip_sets[j] < n]

            remainder_sets.append(np.setdiff1d(parity_sets[j], flip_sets[j]))

            update_pauli.append(Pauli(np.zeros(n, dtype=np.bool), np.zeros(n, dtype=np.bool)))
            parity_pauli.append(Pauli(np.zeros(n, dtype=np.bool), np.zeros(n, dtype=np.bool)))
            remainder_pauli.append(Pauli(np.zeros(n, dtype=np.bool), np.zeros(n, dtype=np.bool)))
            for k in range(n):
                if np.in1d(k, update_sets[j]):
                    update_pauli[j].update_x(True, k)
                if np.in1d(k, parity_sets[j]):
                    parity_pauli[j].update_z(True, k)
                if np.in1d(k, remainder_sets[j]):
                    remainder_pauli[j].update_z(True, k)

            x_j = Pauli(np.zeros(n, dtype=np.bool), np.zeros(n, dtype=np.bool))
            x_j.update_x(True, j)
            y_j = Pauli(np.zeros(n, dtype=np.bool), np.zeros(n, dtype=np.bool))
            y_j.update_z(True, j)
            y_j.update_x(True, j)
            a.append((update_pauli[j] * x_j * parity_pauli[j],
                      update_pauli[j] * y_j * remainder_pauli[j]))
        return a

    # construct the operators
    def mapping(self, map_type, threshold=0.00000001):
        """Map fermionic operator to qubit operator.

        Using multiprocess to speedup the mapping, the improvement can be
        observed when h2 is a non-sparse matrix.

        Args:
            map_type (str): case-insensitive mapping type.
                            "jordan_wigner", "parity", "bravyi_kitaev", "bksf"
            threshold (float): threshold for Pauli simplification

        Returns:
            Operator: create an Operator object in Paulis form.

        Raises:
            QiskitChemistryError: if the `map_type` can not be recognized.
        """
        """
        ####################################################################
        ############   DEFINING MAPPED FERMIONIC OPERATORS    ##############
        ####################################################################
        """

        self._map_type = map_type
        n = self._modes  # number of fermionic modes / qubits
        map_type = map_type.lower()
        if map_type == 'jordan_wigner':
            a = self._jordan_wigner_mode(n)
        elif map_type == 'parity':
            a = self._parity_mode(n)
        elif map_type == 'bravyi_kitaev':
            a = self._bravyi_kitaev_mode(n)
        elif map_type == 'bksf':
            return bksf_mapping(self)
        else:
            raise AttributeError('Please specify correct mapping jordan_wigner, parity, bravyi_kitaev, bksf')
            # raise QiskitChemistryError('Please specify the supported modes: '
            #                            'jordan_wigner, parity, bravyi_kitaev, bksf')
        return a

    def construct_circuit(self, parameter, backend=None, use_simulator_operator_mode=False):

        """Generate the circuits.

        Args:
            parameters (numpy.ndarray): parameters for variational form.
            backend (qiskit.BaseBackend): backend object.
            use_simulator_operator_mode (bool): is backend from AerProvider, if True and mode is paulis,
                           single circuit is generated.

        Returns:
            [QuantumCircuit]: the generated circuits with Hamiltonian.
        """
        input_circuit = self._var_form.construct_circuit(parameter)
        if backend is None:
            warning_msg = "Circuits used in VQE depends on the backend type, "
            from qiskit import BasicAer
            if self._operator_mode == 'matrix':
                temp_backend_name = 'statevector_simulator'
            else:
                temp_backend_name = 'qasm_simulator'
            backend = BasicAer.get_backend(temp_backend_name)
            warning_msg += "since operator_mode is '{}', '{}' backend is used.".format(
                self._operator_mode, temp_backend_name)
            logger.warning(warning_msg)
        circuit = self._operator.construct_evaluation_circuit(self._operator_mode, input_circuit, backend,
                                                              use_simulator_operator_mode=self._use_simulator_operator_mode)
        return circuit

        # This is the objective function to be passed to the optimizer that is uses for evaluation

    def _energy_evaluation(self, parameters):
        """
        Evaluate energy at given parameters for the variational form.

        Args:
            parameters (numpy.ndarray): parameters for variational form.

        Returns:
            float or list of float: energy of the hamiltonian of each parameter.
        """
        num_parameter_sets = len(parameters) // self._var_form.num_parameters
        circuits = []
        parameter_sets = np.split(parameters, num_parameter_sets)
        mean_energy = []
        std_energy = []

        for idx in range(len(parameter_sets)):
            parameter = parameter_sets[idx]
            circuit = self.construct_circuit(parameter, self._quantum_instance.backend,
                                             self._use_simulator_operator_mode)
            circuits.append(circuit)

        to_be_simulated_circuits = functools.reduce(lambda x, y: x + y, circuits)
        if self._use_simulator_operator_mode:
            extra_args = {'expectation': {
                'params': [self._operator.aer_paulis],
                'num_qubits': self._operator.num_qubits}
            }
        else:
            extra_args = {}
        result = self._quantum_instance.execute(to_be_simulated_circuits, **extra_args)

        for idx in range(len(parameter_sets)):
            # the energy
            mean, std = self._operator.evaluate_with_result(
                self._operator_mode, circuits[idx], self._quantum_instance.backend, result,
                self._use_simulator_operator_mode)
            mean_energy.append(np.real(mean))
            std_energy.append(np.real(std))
            self._eval_count += 1
            # if self._callback is not None:
            #     self._callback(self._eval_count, parameter_sets[idx], np.real(mean), np.real(std))
            logger.info('Energy evaluation {} returned {}'.format(self._eval_count, np.real(mean)))

        return mean_energy if len(mean_energy) > 1 else mean_energy[0]

    @staticmethod
    def ferm_cr_op(a_i, threshold=1e-12):

        pauli_list = []
        for alpha in range(2):
                pauli = a_i[alpha]
                coeff = 1 / 2 * np.power(-1j, alpha)
                pauli_term = [coeff, pauli]
                if np.absolute(pauli_term[0]) > threshold:
                    pauli_list.append(pauli_term)
        # print(pauli_list)
        return Operator(paulis=pauli_list)

    @staticmethod
    def ferm_annih_op(a_i, threshold=1e-20):
        pauli_list = []
        for alpha in range(2):
                pauli = a_i[alpha]
                coeff = 1 / 2 * np.power(1j, alpha)
                pauli_term = [coeff, pauli]
                if np.absolute(pauli_term[0]) > threshold:
                    pauli_list.append(pauli_term)
        # print(pauli_list)
        return Operator(paulis=pauli_list)

    @staticmethod
    def ferm_num_op(a_i, threshold=1e-20):

        creation_op= RDMFermionicOperator.ferm_cr_op(a_i, threshold)
        annihilation_op= RDMFermionicOperator.ferm_annih_op(a_i, threshold)
        number_operator = creation_op * annihilation_op

        return number_operator

    @staticmethod
    def ferm_id_op(a_i, threshold=1e-20):

        creation_op = RDMFermionicOperator.ferm_cr_op(a_i, threshold)
        annihilation_op = RDMFermionicOperator.ferm_annih_op(a_i, threshold)
        id_op = creation_op * annihilation_op + annihilation_op * creation_op

        return id_op

    @staticmethod
    def transition_operators(a, orbital_indice, threshold=1e-20):
        """
        Computes the operators necessary to construct the one/two el. density matrices
        to compute the mutual information
        :param a:
        :param orbital_indice:
        :param threshold:
        :return:
        """

        trans_op = {1: 0., 2: 0., 3: 0., 4: 0.,
                    5: 0., 6: 0., 7: 0., 8: 0.,
                    9: 0., 10: 0., 11: 0., 12: 0.,
                    13: 0., 14: 0., 15: 0., 16: 0.,
                    }

        # indices of up, down orbital
        # depends on the convention, block spin used here
        ind_up, ind_down = orbital_indice

        # initialise the builduing blocks: creation, annh, num operators
        num_op_up = RDMFermionicOperator.ferm_num_op(a[ind_up], threshold)
        num_op_down = RDMFermionicOperator.ferm_num_op(a[ind_down], threshold)

        id_op = RDMFermionicOperator.ferm_id_op(a[ind_up], threshold)

        cr_op_up = RDMFermionicOperator.ferm_cr_op(a[ind_up], threshold)
        cr_op_down = RDMFermionicOperator.ferm_cr_op(a[ind_down], threshold)

        annih_op_up = RDMFermionicOperator.ferm_annih_op(a[ind_up], threshold)
        annih_op_down = RDMFermionicOperator.ferm_annih_op(a[ind_down], threshold)

        # 1 operator
        trans_op[1] = id_op - num_op_up - num_op_down + num_op_up*num_op_down

        # 2 operator
        trans_op[2] = annih_op_down - num_op_up * annih_op_down

        # 3 operator
        trans_op[3] = annih_op_up - num_op_down * annih_op_up

        # 4 operator
        trans_op[4] = annih_op_down * annih_op_up

        # 5 operator
        trans_op[5] = cr_op_down - num_op_up * cr_op_down

        # 6 operator
        trans_op[6] = num_op_down - num_op_up * num_op_down

        # 7 operator
        trans_op[7] = cr_op_down * annih_op_up

        # 8 operator
        trans_op[8] = -num_op_down * annih_op_up

        # 9 operator
        trans_op[9] = cr_op_up - num_op_down * cr_op_up

        # 10 operator
        trans_op[10] = annih_op_down * cr_op_up

        # 11 operator
        trans_op[11] = num_op_up - num_op_up * num_op_down

        # MISTAKE CHECK
        # 12 operator
        trans_op[12] = num_op_up * annih_op_down

        # 13 operator
        trans_op[13] = cr_op_down * cr_op_up

        # 14 operator
        trans_op[14] = - num_op_down * cr_op_up

        # 15 operator
        trans_op[15] = num_op_up * cr_op_down

        # 16 operator
        trans_op[16] = num_op_up * num_op_down

        return trans_op

    def one_orbital_reduced_density_matrix(self, parameters, indices_orbitals, threshold=1e-12):
        """
        Diagonal density matrix, the elements should all be positive as they are the probabilities but it depends on
        on the state.
        :param parameters:
        :param indices_orbitals:
        :param threshold:
        :return:
        """

        trans_op = RDMFermionicOperator.transition_operators(self._a, indices_orbitals, threshold=threshold)
        rho = np.zeros((4,4))
        self._operator = trans_op[1]
        rho[0][0] = self._energy_evaluation(parameters)
        self._operator = trans_op[6]
        rho[1][1] = self._energy_evaluation(parameters)
        self._operator = trans_op[11]
        rho[2][2] = self._energy_evaluation(parameters)
        self._operator = trans_op[16]
        rho[3][3] = self._energy_evaluation(parameters)

        return rho

    def two_orbital_reduced_density_matrix(self, parameters, indices_orb_i, indices_orb_j, threshold=1e-12):
        """
        Fills the whole two el. density matrix (some products of operators render None but I set the
        manually those elements to 0)
        :param parameters:
        :param indices_orb_i:
        :param indices_orb_j:
        :param threshold:
        :return:
        """
        # transition ops for i spatial orbital
        trans_opi = RDMFermionicOperator.transition_operators(self._a, indices_orb_i, threshold=threshold)
        # transition ops for j spatial orbital
        trans_opj = RDMFermionicOperator.transition_operators(self._a, indices_orb_j, threshold=threshold)

        rho = np.zeros((16,16))

        # block 1
        self._operator = trans_opi[1]*trans_opj[1]
        rho[0][0] = self._energy_evaluation(parameters)

        # block 2
        self._operator = trans_opi[1]*trans_opj[6]
        rho[1][1] = self._energy_evaluation(parameters)
        self._operator = trans_opi[2]*trans_opj[5]
        rho[1][2] = self._energy_evaluation(parameters)
        self._operator = trans_opi[5]*trans_opj[2]
        rho[2][1] = self._energy_evaluation(parameters)
        self._operator = trans_opi[6]*trans_opj[1]
        rho[2][2] = self._energy_evaluation(parameters)

        # block 3
        self._operator = trans_opi[1]*trans_opj[11]
        rho[3][3] = self._energy_evaluation(parameters)
        self._operator = trans_opi[3]*trans_opj[9]
        rho[3][4] = self._energy_evaluation(parameters)
        self._operator = trans_opi[9]*trans_opj[3]
        rho[4][3] = self._energy_evaluation(parameters)
        self._operator = trans_opi[11]*trans_opj[1]
        rho[4][4] = self._energy_evaluation(parameters)

        # block 4
        self._operator = trans_opi[6]*trans_opj[6]
        rho[5][5] = self._energy_evaluation(parameters)

        # block 5
        self._operator = trans_opi[1]*trans_opj[16]
        rho[6][6] = self._energy_evaluation(parameters)
        self._operator = trans_opi[2]*trans_opj[15]
        rho[6][7] = self._energy_evaluation(parameters)
        self._operator = trans_opi[3]*trans_opj[14]
        rho[6][8] = self._energy_evaluation(parameters)
        self._operator = trans_opi[4]*trans_opj[13]
        rho[6][9] = self._energy_evaluation(parameters)
        self._operator = trans_opi[5]*trans_opj[12]
        rho[7][6] = self._energy_evaluation(parameters)
        self._operator = trans_opi[6]*trans_opj[11]
        rho[7][7] = self._energy_evaluation(parameters)
        self._operator = trans_opi[7]*trans_opj[10]
        rho[7][8] = self._energy_evaluation(parameters)

        # print(trans_opi[8].paulis)
        # print(trans_opj[9].paulis)
        self._operator = trans_opi[8]*trans_opj[9]
        rho[7][9] = self._energy_evaluation(parameters)
        self._operator = trans_opi[9]*trans_opj[8]
        rho[8][6] = self._energy_evaluation(parameters)
        self._operator = trans_opi[10]*trans_opj[7]
        rho[8][7] = self._energy_evaluation(parameters)
        self._operator = trans_opi[11]*trans_opj[6]
        rho[8][8] = self._energy_evaluation(parameters)
        self._operator = trans_opi[12]*trans_opj[5]
        rho[8][9] = self._energy_evaluation(parameters)
        self._operator = trans_opi[13]*trans_opj[4]
        rho[9][6] = self._energy_evaluation(parameters)
        self._operator = trans_opi[14]*trans_opj[3]
        rho[9][7] = self._energy_evaluation(parameters)
        self._operator = trans_opi[15]*trans_opj[2]
        rho[9][8] = self._energy_evaluation(parameters)
        self._operator = trans_opi[16]*trans_opj[1]
        rho[9][9] = self._energy_evaluation(parameters)

        # block 6
        self._operator = trans_opi[11] * trans_opj[11]
        rho[10][10] = self._energy_evaluation(parameters)

        # block 7
        self._operator = trans_opi[6] * trans_opj[16]
        rho[11][11] = self._energy_evaluation(parameters)
        self._operator = trans_opi[8] * trans_opj[14]
        rho[11][12] = self._energy_evaluation(parameters)
        self._operator = trans_opi[14] * trans_opj[8]
        rho[12][11] = self._energy_evaluation(parameters)
        self._operator = trans_opi[16] * trans_opj[6]
        rho[12][12] = self._energy_evaluation(parameters)

        # block 8
        self._operator = trans_opi[11] * trans_opj[16]
        rho[13][13] = self._energy_evaluation(parameters)
        self._operator = trans_opi[12] * trans_opj[15]
        rho[13][14] = self._energy_evaluation(parameters)
        self._operator = trans_opi[15] * trans_opj[12]
        rho[14][13] = self._energy_evaluation(parameters)
        self._operator = trans_opi[16] * trans_opj[11]
        rho[14][14] = self._energy_evaluation(parameters)

        # block 9
        self._operator = trans_opi[16] * trans_opj[16]
        rho[15][15] = self._energy_evaluation(parameters)

        return rho

    @staticmethod
    def compute_eigenvalues_matrix(matrix):
        """
        Gives the eigenvalues of a matrix
        :param matrix:
        :return:
        """
        from numpy import linalg as LA
        eig_vals = LA.eigvals(matrix)
        return eig_vals

    @staticmethod
    def compute_information(matrix):
        """
        Computes the entropy using usual ln formula
        :param matrix:
        :return:
        """
        eig_vals = RDMFermionicOperator.compute_eigenvalues_matrix(matrix)
        entropy = 0.
        for eig in eig_vals:
            if eig != 0:
                if eig < 0:
                    eig = abs(eig)
                    logger.info('WARNING /!\ : Absolute value of eigenvalues was set ! This should not happen.')
                entropy += -eig * np.log(eig)

        return entropy

    @staticmethod
    def compute_mutual_information(rho1i,rho1j, rho2):
        """
        Computes mutual information for chosen spatial orbitals
        :param rho1i:
        :param rho1j:
        :param rho2:
        :return:
        """
        entropy_i = RDMFermionicOperator.compute_information(rho1i)
        entropy_j = RDMFermionicOperator.compute_information(rho1j)
        entropy_ij = RDMFermionicOperator.compute_information(rho2)
        mutual_info = 1 / 2 * (entropy_ij - entropy_i - entropy_j)
        return mutual_info

    def mutual_information_matrix_element(self, orbitals_i, orbitals_j, parameters):
        """
        Computes a matrix element
        :param orbitals_i:
        :param orbitals_j:
        :param parameters:
        :return:
        """

        rho1i = self.one_orbital_reduced_density_matrix(indices_orbitals=orbitals_i, parameters=parameters)
        rho1j = self.one_orbital_reduced_density_matrix(indices_orbitals=orbitals_j, parameters=parameters)
        rho2 = self.two_orbital_reduced_density_matrix(indices_orb_i=orbitals_i, indices_orb_j=orbitals_j,
                                                      parameters=parameters)
        mut_info = self.compute_mutual_information(rho1i=rho1i, rho1j=rho1j, rho2=rho2)
        return mut_info

    def mutual_information_matrix(self, num_orbitals, parameters):
        """
        Computes whole Iij, mutual information matrix only the upper off diag elements (non-zero ones)
        :param num_orbitals:
        :param parameters:
        :return:
        """

        # in the block spin notation
        half_act_space = int(num_orbitals / 2)
        list_orbitals = []

        matrix = np.zeros((half_act_space, half_act_space))

        # make list of all orbitals
        for i in range(half_act_space):
            list_orbitals.append([i, i + half_act_space])
        for i in range(half_act_space):
            for j in range(half_act_space):
                if i < j:
                    matrix[i][j] = self.mutual_information_matrix_element(orbitals_i= list_orbitals[i],
                                                                          orbitals_j=list_orbitals[j],
                                                                          parameters=parameters)
        return matrix

    def total_information(self, num_orbitals, parameters):
        """
        Computes the sum of entopies of all the orbitals
        :param num_orbitals:
        :param parameters:
        :return:
        """

        # in the block spin notation
        half_act_space = int(num_orbitals / 2)
        list_orbitals = []

        # make list of all orbitals
        for i in range(half_act_space):
            list_orbitals.append([i, i + half_act_space])

        total_info = 0.
        orbs_info = []
        for orbs in list_orbitals:
                rho1i = self.one_orbital_reduced_density_matrix(indices_orbitals=orbs, parameters=parameters)
                info = self.compute_information(rho1i)
                orbs_info.append(info)
                total_info += info

        return total_info, orbs_info


def plot_graph_mutual_info(list_edges_weights, max_width=20, path=None):
    """
    Plots the orbital diagram without consideration for single orb entropy, all node sizes same
    :param list_edges_weights:
    :param max_width:
    :return:
    """
    G = nx.Graph()
    size = len(list_edges_weights)

    for i in range(size):
        G.add_edge(list_edges_weights[i][0], list_edges_weights[i][1], weight=list_edges_weights[i][2])

    weights = []
    for i in range(size):
        weights.append(list_edges_weights[i][2])

    max_weight = max(weights)

    # widths of edges for all the points
    widths = []
    for i in range(size):
        widths.append(weights[i] / max_weight * max_width)

    edges = []
    for i in range(size):
        edges.append([(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] == weights[i]])

    pos = nx.circular_layout(G)  # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)

    for i in range(size):
        nx.draw_networkx_edges(G, pos, edgelist=edges[i],
                               width=widths[i])

    # labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')

    plt.axis('off')
    if path is not None:
        plt.savefig(path)
    plt.show()
    plt.close()

    return plt


def plot_graph_mutual_and_sinle_orb_info(list_edges_weights, list_orb_entr,  max_width=19, max_node_size=2000, shift_width = 1,
                                         shift_node_size = 200, coloring= True, threshold= None, path=None):
    """
    Plots the orbital diagram with node size proportional to the single orbital entropy

    :param list_edges_weights:
    :param max_width:
    :return:
    """
    """"""
    G = nx.Graph()
    size = len(list_edges_weights)

    for i in range(size):
        G.add_edge(list_edges_weights[i][0], list_edges_weights[i][1], weight=list_edges_weights[i][2])

    # nodes
    # widths of edges for all the points
    entropies = []
    size_entr = len(list_orb_entr)
    for i in range(size_entr):
        entropies.append(list_orb_entr[i])
    max_size = max(entropies)
    node_sizes = []
    for i in range(size_entr):
        node_sizes.append(entropies[i] / max_size * max_node_size + shift_node_size)

    # edges
    # widths of edges for all the points
    weights = []
    for i in range(size):
        weights.append(list_edges_weights[i][2])
    max_weight = max(weights)

    if threshold == None:
        # average of weights is the threshold
        sum = 0
        n= 0
        for i in weights:
            sum += i
            n+=1
        threshold = sum / n

    widths = []
    for i in range(size):
        widths.append(weights[i] / max_weight * max_width + shift_width)
    edges = []
    edges_colors = []
    for i in range(size):
        edges.append([(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] == weights[i]])
        if weights[i] > threshold:
            edges_colors.append('black')
        else:
            edges_colors.append('lime')

    pos = nx.circular_layout(G)  # positions for all nodes
    # pos = nx.spring_layout(G)  # positions for all nodes

    # nodes
    nodes=list(G)
    print(nodes)

    for i in range(size):
        nx.draw_networkx_edges(G, pos, edgelist=edges[i],
                               width=widths[i], edge_color=edges_colors[i])

    for i in range(len(nodes)):
        print(i)
        nx.draw_networkx_nodes(G, pos, nodelist=[nodes[i]],  node_size=node_sizes[i], node_color='red')

    # labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')

    plt.axis('off')
    if path is not None:
        plt.savefig(path)
    plt.show()
    plt.close()


    return plt

def plot_orb_entropy(list_orb_entr, path=None):
    """
    Makes a bar plot of the orbital entropies
    :param list_orb_entr:
    :return:
    """
    # plot orbital energy
    bars = []
    for i in range(len(list_orb_entr)):
        bars.append(i)
    bars = tuple(bars)
    # bars = ('A', 'B', 'C', 'D')
    y_pos = np.arange(len(bars))

    # Create bars
    plt.bar(y_pos, list_orb_entr)

    # Create names on the x-axis
    plt.xticks(y_pos, bars)

    # Show graphic
    if path is not None:
        plt.savefig(path)
    plt.show()
    plt.close()
    # print(total_entr, list_orb_entr)
    return plt


# initialise operators
# fer_op =RDMFermionicOperator(number_modes=2)
# a = fer_op.mapping(map_type='jordan_wigner')
# a0dag = fer_op.ferm_cr_op(a[1], threshold=1e-32)
# a0 = fer_op.ferm_annih_op(a[1], threshold=1e-32)
# numb_op0 = fer_op.ferm_num_op(a[1], threshold=1e-32)
# op = a0dag*a0
# id_op = fer_op.ferm_id_op(a[1], threshold=1e-32)
# trans_op = fer_op.transition_operators(a, [0,1])
# orbital_indice = [0,1]
# # rho1orb = fer_op.one_orbital_reduced_density_matrix(threshold=1e-12)
## trans_op_vals = list(trans_op.values())
#
#
# # print(a0.get_flat_pauli_list())
# # print(a0dag.get_flat_pauli_list())
# # print(numb_op0.get_flat_pauli_list())
# # print(op.get_flat_pauli_list())
# print(trans_op_vals[0].to_paulis())
# # print(id_op.get_flat_pauli_list())
# # print(trans_op.values())
# assert op == numb_op0



# mat_a0 = a0.to_matrix()

# print(a[1][0])
# print(a[1][1])
# print(np.power(-1j, 1) * Operator(Pauli(label=[a[1][1]])))

###################


# driver = PySCFDriverOpenShell(atom="Be 0. 0. 0.; H 0. 0. 1.",
#                      unit=UnitsType.ANGSTROM,
#                      charge=0,
#                      spin=1,
#                      basis='sto3g',
#                     calc_type='uhf')
#
# driver = PySCFDriver(atom="Li 0. 0. 0.; H 0. 0. 1.",
#                               unit=UnitsType.ANGSTROM,
#                               charge=1,
#                               spin=1,
#                               basis='sto3g',
#                               hf_method=HFMethodType.UHF)
#
# driver = PySCFDriver(atom= "C; H 1 1; H 1 1 2 125.0",
#                      unit=UnitsType.ANGSTROM,
#                      charge=0,
#                      spin=2,
#                      basis='sto3g',
#                     hf_method=HFMethodType.UHF)
#
# driver = PySCFDriverOpenShell(atom="H 0. 0. 0.; H 0. 0. 1.;H 0. 0. 2.;H 0. 0. 3.",
#                      unit=UnitsType.ANGSTROM,
#                      charge=0,
#                      spin=0,
#                      basis='sto3g',
#                     calc_type='uhf')

#
# driver = PySCFDriverOpenShell(atom="Be 0. 0. 0.; H 0. 0. 1.",
#                      unit=UnitsType.ANGSTROM,
#                      charge=2,
#                      spin=1,
#                      basis='sto3g',
#                     calc_type='uhf')
#
# driver = PySCFDriver(atom="H 0. 0. 0.; H 0. 0. 1.",
#                      unit=UnitsType.ANGSTROM,
#                      charge=0,
#                      spin=0,
#                      basis='sto3g')
#
# driver = PySCFDriver(atom="H 1.738000 .0 .0; H 0.15148 1.73139 .0; H -1.738 .0 .0; H -0.15148 -1.73139 .0",
#                      unit=UnitsType.ANGSTROM,
#                      charge=0,
#                      spin=0,
#                      basis='sto3g')

# import numpy as np
# from qiskit import BasicAer
# from qiskit.aqua import QuantumInstance, Operator
# from qiskit.aqua.algorithms import ExactEigensolver
# from numpy import linalg as LA
# import matplotlib.pyplot as plt


# for the initialisation of the variational form

# MolHam = MoleculeOperators(driver)
# MolHam.setUp()

# needed things

# backend = BasicAer.get_backend('statevector_simulator')
# quantum_instance = QuantumInstance(backend=backend)
# var_form = MolHam.init_var_form()
# num_parameters = var_form.num_parameters
# parameters = []
# for i in range(num_parameters):
#     parameters.append(2.)
# parameters = np.array(parameters)
# orbitals_j = [0,4]
# orbitals_i = [1,5]


# test the functions

# RMD = RDMFermionicOperator(number_modes=8, operator_mode='matrix', var_form=var_form,
#                            map_type='jordan_wigner', parameters=parameters, quantum_instance=quantum_instance)
#
# rho1 = RMD.one_orbital_reduced_density_matrix(indices_orbitals=orbitals_i, parameters=parameters)
# rho1j = RMD.one_orbital_reduced_density_matrix(indices_orbitals=orbitals_j, parameters=parameters)
# LA.eigvals(rho1)
# entr = RMD.compute_information(rho1)
# rho2 = RMD.two_orbital_reduced_density_matrix(indices_orb_i= orbitals_i, indices_orb_j= orbitals_j, parameters=parameters)
# mut_info = RMD.compute_mutual_information(rho1i=rho1, rho1j=rho1j, rho2=rho2)
#
# print(rho1)
# # print(MolHam.qubit_op)
# print(entr)
# print(mut_info)

# Working Test code

# RMD = RDMFermionicOperator(number_modes=4, operator_mode='matrix', var_form=var_form,
#                            map_type='jordan_wigner', parameters=parameters, quantum_instance=quantum_instance)
# mut_info_matrix = RMD.mutual_information_matrix(num_orbitals=4, parameters=parameters)
# print(mut_info_matrix)
#
# plt.matshow(mut_info_matrix);
# plt.colorbar()
# plt.show()