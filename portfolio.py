#from qiskit_aer.primitives import SamplerV2
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import pandas as pd
import random
import numpy as np

from qiskit.quantum_info import state_fidelity
from qiskit.quantum_info import random_statevector

import scipy
import qiskit
from qiskit import Aer, execute, transpile

import sys
sys.path.append("./src/groundstate_prep")
sys.path.append("./src/rqcopt")
from ground_state_prep_qiskit import qetu_rqc_oneLayer



def run(input_data):
    d = input_data
    
    # We are taking the top 4 assets with highest 
    # expected returns and will optimize over them
    # This is of course a significantly low number
    # but for simulability of small scale quantum 
    # computers we limit ourselves to small data set and low qubit counts.
    filtering_number = 4 # User input

    L = d['num_assets']
    asset_names = list(d["assets"].keys())
    p = [list(d['assets'][name]['history'].values()) for name in asset_names] # Stock prices over time
    P = [p[i][-1] if len(p[i])!=0 else 0 for i in range(L)] # Current Stock Prices


    # Expected Return:
    r = [[(p[i][t]-p[i][t-1])/p[i][t-1] if t>0 else 0 for t in range(len(p[i]))] for i in range(len(p))]
    mu = []
    for i in range(L):
        exp = np.sum(np.array([r[i][t] for t in range(len(r[i]))])) / len(r[i])
        if not np.isnan(exp):
            mu.append(exp)
        else:
            mu.append(-1)


    # We take only the highest 10 expected returns and will optimize over them.
    # This can be user defined parameter. As the quantum computer scales up,
    # we'll be able to increase this number.
    indices = np.argsort(mu)
    mu_filtered = []
    r_filtered = []
    p_filtered = []
    for index in indices[-filtering_number:]:
        mu_filtered.append(mu[index])
        r_filtered.append(r[index])
        p_filtered.append(p[index])

    mu = mu_filtered
    r = r_filtered
    L = filtering_number
    p = p_filtered
    P = [p[i][-1] for i in range(L)] # Current Stock Prices, after initial filtering.

    # We have a fixed budget for the number of qubits that this algortihm is
    # 'allowed' to use. The more qubits you can accommodate on your hardware,
    # the higher of a 'filtering_number' you can choose and optimize over
    # a larger set of assets. For our simulation reasons, we hard-code a very low
    # number of qubits as budget - as we'll use state_vector simulations.
    N = 8 # User input, algortihm will use N +- 1 many qubits.

    N = N - 1
    # Qubit budget is translated into an absolute budget B by using stock prices after filtering.
    # User can also hard-code the budget here, allows for more capability.
    B = 2**((np.sum(np.array([np.log(price)/np.log(2) for price in P])) + N)/filtering_number)
    n_max = [np.floor(B/P[i]) for i in range(L)]
    D = [np.ceil(np.log(n_max[i])/np.log(2)) for i in range(L)]
    N = int(np.sum(np.array(D)))

    # Co-Variance Matrix:
    Sigma = [
        [   np.sum(np.array([
            (r[i][t]-mu[i])*(r[j][t]-mu[j]) for t in range( min( len(r[i]),  len(r[j])) )
        ]))/(len(r[i]) - 1)
            for j in range(L)
        ] for i in range(L)
    ]

    # To better optimize the process, we include the budget 
    # constraint and the fact that only
    # an integer multiple of each stock can be bought. 
    # This is to avoid unspent budget/overspending.

    P_p = [P[i]/B for i in range(L)]
    mu_p = [mu[i]*P_p[i] for i in range(L)]
    Sigma_p = [[P_p[i]*P_p[j]*Sigma[i][j] for j in range(L)] for i in range(L)]

    # Convert to Binary transformation.
    C = [[] for i in range(L)]
    for i in range(L):
        for j in range(i):
            C[i] += [0]*int(D[j])
        C[i] += [2**d for d in range(int(D[i]))]
        for j in range(i+1, L):
            C[i] += [0]*int(D[j])
    C = np.array(C)

    mu_pp = C.T @ mu_p
    Sigma_pp = C.T @ Sigma_p @ C
    P_pp = C.T @ P_p

    
    q = 1 # User-Defined Risk aversion factor.
    lamb = 10 # Budget penalty param.

    # QUBO is mapped to an Ising Hamiltonian.
    # Our idea leverages the quantum computer's inherent advantage of simulating
    # physical systems by mapping optimization problem to a physical Hamiltonian.
    # In this analogy, co-variance matrix elements are spin interaction factors
    # and expected returns are external magnetic field.
    Sigma_pp = np.array(Sigma_pp)
    J = q/4 * Sigma_pp
    h = [-0.5*mu_pp[i] + 0.5*q*(Sigma_pp@np.array([1 for j in range(N)]))[i] for i in range(N)]
    pi = [P_pp[i]/2 for i in range(N)]
    beta = 1 - np.sum(np.array([P_pp[i]/2 for i in range(N)]))


    # We start the quantum protocol now. We'll apply successive 
    # eigenstate filtering through QETU sequence. We start with a 
    # random initial state.
    vec = random_statevector(2**N)

    spectrum_upper_bound = 1e2 # Can be put arbitrarily high but then the number of
                               # successive filtering stages (M) should be increased
                               # in exchange.
    M = 3 # The more filtering stages you apply, the more  
          # you are certain of achieving a lower energy subspace.

    ket_0 = np.array([1, 0]) # ancilla qubit is init. as ket{0}
    qcs_qetu = []
    for i in range(M):

        # This linear transformation compresses the spectrum into [0, pi].
        spectrum_upper_bound = spectrum_upper_bound/2
        spectrum_lower_bound = -1
        max_spectrum_length = spectrum_upper_bound - spectrum_lower_bound
        c1 = (np.pi) / (max_spectrum_length)
        c2 = - c1 * spectrum_lower_bound
    
        # Time evolution block:
        t = c1
        qc_U = qiskit.QuantumCircuit(N)
        qc_U_dag = qiskit.QuantumCircuit(N)
        nsteps = 1
        for n in range(nsteps):
            qc_U.append(trotterized_time_evolution(J, h, pi, lamb, beta, t/nsteps, N).to_gate(), [i for i in range(N)])
            qc_U_dag.append(trotterized_time_evolution(J, h, pi, lamb, beta, -t/nsteps, N).to_gate(), [i for i in range(N)])
        
        qc_cU = qiskit.QuantumCircuit(N+1)
        qc_cU.append(qc_U.to_gate().control(), [N] + [i for i in range(N)])
        qc_cU_dag = qiskit.QuantumCircuit(N+1)
        qc_cU_dag.append(qc_U_dag.to_gate().control(), [N] + [i for i in range(N)])
        qc_cfUs = [qc_cU, qc_cU_dag]
        
        # QETU Circuit:
        mu, d, c, phis_max_iter, = (0.99, 30, 0.95, 10)
        qc_qetu, _ = qetu_rqc_oneLayer(N, 0, 0, 1, mu, d=d, c2=c2,
                                        qc_cU_custom=(qc_cfUs[0], qc_cfUs[1])
                                        )
        qcs_qetu.append(qc_qetu)

    backend = Aer.get_backend("statevector_simulator") # Just for the sake of simulations we use
                                                       # we use state_vector. This severely limits the
                                                       # qubit budget in our simulations. 
                                                       # If we had more time we could have implemented 
                                                       # a more efficient version such that we could 
                                                       # simulate with more qubits then circa 10 :)
    qc_RQC = qiskit.QuantumCircuit(N+1, N+1)
    qc_RQC.initialize(np.kron(ket_0, vec))
    for qc_qetu in qcs_qetu:
        # We apply QETU circuits one by one. In between each QETU sequence, we need to measure
        # the ancilla qubit and make sure its measured as 0. If it's measured as 1 the sequence is
        # canceled and we start over. Measuring the ancilla this way as 0 is probabilistic. 
        # This code implementation reports the success probability of measuring the ancilla as 0
        # at each filtering stage.
        qc_RQC.append(qc_qetu.to_gate(), [i for i in range(N+1)])
        bR = execute(transpile(qc_RQC), backend).result().get_statevector().data
        aR = np.kron(np.array([[1,0],[0,0]]), np.identity(2**N)) @ bR
        print("Projecting the ancilla qubit onto 0 with success prob: ", np.linalg.norm(aR)**2)
        aR = aR / np.linalg.norm(aR)
        qc_RQC.reset([i for i in range(N+1)])
        qc_RQC.initialize(aR)

    # The outcome we get as measurement result, converges more and more into a bunch of 
    # bitstrings in the low energy manifold of our Hamiltonian. Even though we won't necessarily
    # get ideal lowest energy state, we are guaranteed to be in the low energy sub-space thanks
    # to the non-variational nature of our purely quantum algorithm :) 
    # This is the advantage of non variational ground state preparation!

    qc_to_measure = qiskit.QuantumCircuit(N+1, N+1)
    qc_to_measure.initialize(aR)
    qc_to_measure.measure([i for i in range(N)], [i for i in range(N)])
    backend = Aer.get_backend("aer_simulator")
    shots = 1e3
    counts = execute(transpile(qc_to_measure), backend, shots=shots).result().get_counts()

    opt_bstr = ''.join('1' if x == '0' else '0' for x in sorted(counts, key=counts.get, reverse=True)[2][1:])
    b_Ising = []
    for ch in opt_bstr:
        b_Ising.append(0 if ch == '1' else 1)
    b_Ising = np.array(b_Ising)
    q_opt = C@b_Ising
    print("Optimal Portfolio: ", q_opt)

    # Turn the integer valued optimal stocks to weights.
    total_money_to_invest = 0
    result = {}
    for i, index in enumerate(indices[-filtering_number:]):
        total_money_to_invest+=q_opt[i]*P[i]
        result[asset_names[index]] = q_opt[i]*P[i]
    for i, index in enumerate(indices[-filtering_number:]):
        result[asset_names[index]] = result[asset_names[index]]/total_money_to_invest

    return {'selected_assets_weights':result, 'num_selected_assets': len(result)}





def trotterized_time_evolution(J, h, pi, lamb, beta, t, L):
    """
        Trotterized Time evolution encoding of the Ising Hamiltonian.
    """
    qc = qiskit.QuantumCircuit(L)
    Z = np.array([[1.,  0.], [0., -1.]])

    gates = []
    for i in range(L):
        for j in range(i+1, L):
            hloc = (J[i][j] + J[j][i] + 2*lamb*pi[i]*pi[j]) * np.kron(Z, Z)
            qc2 = qiskit.QuantumCircuit(2)
            qc2.unitary(scipy.linalg.expm(-1j*t*0.5*hloc), [0, 1], label='str')
            gates.append((qc2, [L-i-1, L-j-1]))
            qc.append(gates[-1][0].to_gate(), gates[-1][1])
    gates.reverse()
    for gate in gates:
        qc.append(gate[0].to_gate(), gate[1])

    for i in range(L):
        qc.rz(2*t*(h[i] - 2*beta*lamb*pi[i]), L-i-1)

    qc.p(
        -t*(lamb*beta**2+np.sum(np.array([J[i][i] + lamb*pi[i]*pi[i] for i in range(L)]))), L-1
    )
    qc.x(L-1)
    qc.p(
        -t*(lamb*beta**2+np.sum(np.array([J[i][i] + lamb*pi[i]*pi[i] for i in range(L)]))), L-1
    )
    qc.x(L-1)
    return qc
