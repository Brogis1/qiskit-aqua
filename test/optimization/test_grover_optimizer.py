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

"""Test Grover Optimizer."""

import unittest
from test.optimization import QiskitOptimizationTestCase
from docplex.mp.model import Model
import numpy as np
from qiskit import Aer
from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit.aqua.algorithms import NumPyMinimumEigensolver
from qiskit.optimization.algorithms import GroverOptimizer, MinimumEigenOptimizer
from qiskit.optimization.problems import QuadraticProgram


class TestGroverOptimizer(QiskitOptimizationTestCase):
    """GroverOptimizer tests."""

    def setUp(self):
        super().setUp()
        aqua_globals.random_seed = 1
        self.q_instance = QuantumInstance(Aer.get_backend('statevector_simulator'),
                                          seed_simulator=921, seed_transpiler=200)

    def validate_results(self, problem, results):
        """Validate the results object returned by GroverOptimizer."""
        # Get expected value.
        solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
        comp_result = solver.solve(problem)

        # Validate results.
        np.testing.assert_array_almost_equal(comp_result.x, results.x)
        self.assertEqual(comp_result.fval, results.fval)
        self.assertAlmostEqual(results.fval, results.intermediate_fval)

    def test_qubo_gas_int_zero(self):
        """Test for when the answer is zero."""

        # Input.
        model = Model()
        x_0 = model.binary_var(name='x0')
        x_1 = model.binary_var(name='x1')
        model.minimize(0*x_0+0*x_1)
        op = QuadraticProgram()
        op.from_docplex(model)

        # Will not find a negative, should return 0.
        gmf = GroverOptimizer(1, num_iterations=1, quantum_instance=self.q_instance)
        results = gmf.solve(op)
        np.testing.assert_array_almost_equal(results.x, [0, 0])
        self.assertEqual(results.fval, 0.0)
        self.assertAlmostEqual(results.fval, results.intermediate_fval)

    def test_qubo_gas_int_simple(self):
        """Test for simple case, with 2 linear coeffs and no quadratic coeffs or constants."""

        # Input.
        model = Model()
        x_0 = model.binary_var(name='x0')
        x_1 = model.binary_var(name='x1')
        model.minimize(-x_0+2*x_1)
        op = QuadraticProgram()
        op.from_docplex(model)

        # Get the optimum key and value.
        n_iter = 8
        gmf = GroverOptimizer(4, num_iterations=n_iter, quantum_instance=self.q_instance)
        results = gmf.solve(op)
        self.validate_results(op, results)

        self.assertIsNotNone(results.operation_counts)
        self.assertEqual(results.n_input_qubits, 2)
        self.assertEqual(results.n_output_qubits, 4)

    def test_qubo_gas_int_simple_maximize(self):
        """Test for simple case, but with maximization."""

        # Input.
        model = Model()
        x_0 = model.binary_var(name='x0')
        x_1 = model.binary_var(name='x1')
        model.maximize(-x_0+2*x_1)
        op = QuadraticProgram()
        op.from_docplex(model)

        # Get the optimum key and value.
        n_iter = 8
        gmf = GroverOptimizer(4, num_iterations=n_iter, quantum_instance=self.q_instance)
        results = gmf.solve(op)
        self.validate_results(op, results)

    def test_qubo_gas_int_paper_example(self):
        """Test the example from https://arxiv.org/abs/1912.04088."""

        # Input.
        model = Model()
        x_0 = model.binary_var(name='x0')
        x_1 = model.binary_var(name='x1')
        x_2 = model.binary_var(name='x2')
        model.minimize(-x_0+2*x_1-3*x_2-2*x_0*x_2-1*x_1*x_2)
        op = QuadraticProgram()
        op.from_docplex(model)

        # Get the optimum key and value.
        n_iter = 10
        gmf = GroverOptimizer(6, num_iterations=n_iter, quantum_instance=self.q_instance)
        results = gmf.solve(op)
        self.validate_results(op, results)


if __name__ == '__main__':
    unittest.main()
