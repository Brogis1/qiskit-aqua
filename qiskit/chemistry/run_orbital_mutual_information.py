# mutual information import
from qiskit.chemistry.orbital_mutual_information import *

# aqua imports
from qiskit import Aer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms.adaptive import VQE
from qiskit.aqua.components.optimizers import SLSQP
from qiskit.chemistry.core import Hamiltonian, TransformationType, QubitMappingType
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD
from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock
import numpy as np

##############################################################################
# Electronic structure calculation (ground state with VQE)
##############################################################################

r = 0.735

molecule = "H 0.000000 0.000000 0.000000;H 0.000000 0.000000 " + str(r)
# molecule = "H 1.738000 .0 .0; H .15148 1.73139 .0; H -1.738 .0 .0;x H -0.15148 -1.73139 .0"

driver = PySCFDriver(atom=molecule,
                     unit=UnitsType.ANGSTROM,
                     charge=0,
                     spin=0,
                     basis='sto3g')

qmolecule = driver.run()

core = Hamiltonian(transformation=TransformationType.FULL,
                   qubit_mapping=QubitMappingType.JORDAN_WIGNER,  # JORDAN
                   two_qubit_reduction=False,
                   freeze_core=False,
                   orbital_reduction=[])
qubit_op, _ = core.run(qmolecule)

the_tapered_op = qubit_op

# optimizers
optimizer = SLSQP(maxiter=1000, ftol=1e-5)

init_state = HartreeFock(num_qubits=the_tapered_op.num_qubits,
                         num_orbitals=core._molecule_info['num_orbitals'],
                         qubit_mapping=core._qubit_mapping,
                         two_qubit_reduction=core._two_qubit_reduction,
                         num_particles=core._molecule_info['num_particles'], )

# UCCSD Ansatz
var_form = UCCSD(num_qubits=the_tapered_op.num_qubits, depth=1,
                 num_orbitals=core._molecule_info['num_orbitals'],
                 num_particles=core._molecule_info['num_particles'],
                 active_occupied=None, active_unoccupied=None,
                 initial_state=init_state,
                 qubit_mapping=core._qubit_mapping,
                 two_qubit_reduction=core._two_qubit_reduction,
                 num_time_slices=1,
                 shallow_circuit_concat=False)

# set up VQE
algo = VQE(the_tapered_op, var_form, optimizer)

# Choose the backend (use Aer instead of BasicAer)
backend = Aer.get_backend('statevector_simulator')
quantum_instance = QuantumInstance(backend=backend)

# run the algorithm
algo_result = algo.run(quantum_instance)

# get the results
_, result = core.process_algorithm_result(algo_result)

energy = result['energy']
print(result)
opt_params = result['algorithm_retvals']['opt_params']
parameters = np.array(opt_params)


###########################################################
# Orbital entanglement calculation
###########################################################

# string where you save the plots of mutual information
path_save_single_orb = None
path_save_mutual = None

num_orbitals = core._molecule_info['num_orbitals']
RMD = RDMFermionicOperator(number_modes=num_orbitals, operator_mode='matrix', var_form=var_form,
                           map_type='jordan_wigner', parameters=parameters, quantum_instance=quantum_instance)

mut_info_matrix = RMD.mutual_information_matrix(num_orbitals=num_orbitals, parameters=parameters)

# prepare the list of orbtials and weights
list_orbs_weights = []
for i in range(mut_info_matrix.shape[0]):
    for j in range(mut_info_matrix.shape[0]):
        if i<j:
            list_orbs_weights.append([i,j,abs(mut_info_matrix[i][j])])

# all the mutual informations for all orbital combinations
print('\n Iij: ', list_orbs_weights)

# without green coloring
plot_graph_mutual_info(list_orbs_weights,max_width=25)


RMD = RDMFermionicOperator(number_modes=num_orbitals, operator_mode='matrix', var_form=var_form,
                           map_type='jordan_wigner', parameters=parameters, quantum_instance=quantum_instance)

total_entr, list_orb_entr = RMD.total_information(num_orbitals=num_orbitals, parameters=parameters)


# plot the full graph with the nodes prop to the one-orbital entropy
plt_mut_single_orb=plot_graph_mutual_and_sinle_orb_info(list_orbs_weights, list_orb_entr, max_width=25, path=path_save_mutual)

# plot of the single orbital entropy
plt_orb_entr = plot_orb_entropy(list_orb_entr, path=path_save_single_orb)



