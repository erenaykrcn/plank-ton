# Demonstrate dataset creation for QCELS Fitting with depolarizing quantum noise.
import qiskit
from qiskit import Aer, execute, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.providers.aer.noise import NoiseModel, errors
from qiskit.converters import circuit_to_dag

import numpy as np
import scipy
import h5py
import rqcopt as oc

import sys
sys.path.append("../rqcopt")
from optimize import ising1d_dynamics_opt



def estimate_phases(L, J, g, prepared_state, eigenvalues_sort, tau, N, 
    shots, depolarizing_error, c2=0, rqc=True, 
    coeffs=None, rqc_layers=5, reuse_RQC=0, nsteps=1, 
    hamil=None, control=False, beta=None,
    tfim_2D_trotter=None, heisenberg_trotter=None,
    trotterized_time_evolution=None, pi=None, lamb=None, h=None,
    return_counts=False, mid_cbits=0, noise_model=None, get_cx=False, qasm=False, longest_path=False,
    Lx=None, Ly=None, bayesian=None, get_sv=False, J1J2_2D_trotter=None, delta_tau=None
    ):
    backend = qiskit.Aer.get_backend("aer_simulator")

    if noise_model is None:
        x1_error = errors.depolarizing_error(depolarizing_error*0.1, 1)
        x2_error = errors.depolarizing_error(depolarizing_error, 2)
        x3_error = errors.depolarizing_error(depolarizing_error, 3)
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(x1_error, ['u1', 'u2', 'u3'])
        noise_model.add_all_qubit_quantum_error(x2_error, ['cu', 'cx','cy', 'cz'])
        noise_model.add_all_qubit_quantum_error(x3_error, ['ccu', 'ccx','ccy', 'ccz'])
        print(noise_model)

    phase_estimates_with_noise = []
    phase_exacts = []
    counts_list = []
    for n in range(1, N):
        state = prepared_state.copy()
        t = n*tau if delta_tau is None else tau + (n-1)*delta_tau
        qc_cU = qiskit.QuantumCircuit(L+1)
        qc_cU_ins = qiskit.QuantumCircuit(L+1)
        print("t: ", t)
        if rqc:
            V_list = []
            path = f"../../../aff/src/rqcopt/results/ising1d_L{L if reuse_RQC==0 else reuse_RQC}_t{t}_dynamics_opt_layers{rqc_layers}.hdf5"
            try:
                with h5py.File(path, "r") as f:
                    V_list = list(f["Vlist"])
            except FileNotFoundError:
                strang = oc.SplittingMethod.suzuki(2, 1)
                _, coeffs_start_n5 = oc.merge_layers(2*strang.indices, 2*strang.coeffs)
                # divide by 2 since we are taking two steps
                coeffs_start_n5 = [0.5*c for c in coeffs_start_n5]
                print("coeffs_start_n5:", coeffs_start_n5)
                V_list = ising1d_dynamics_opt(5, t, False, coeffs_start_n5, path, niter=16)
            perms = [None if i % 2 == 0 else np.roll(range(L), -1) for i in range(len(V_list))]
            qcs = []
            for V in V_list:
                qc = qiskit.QuantumCircuit(2)
                qc.unitary(V, [0, 1])
                qcs.append(qc)
            for layer, qc_gate in enumerate(qcs):
                cgate = qc_gate.to_gate()
                for j in range(L//2):
                    if perms[layer] is not None:
                        qc_cU.append(cgate, [L-(perms[layer][2*j]+1), L-(perms[layer][2*j+1]+1)])
                    else:
                        qc_cU.append(cgate, [L-(2*j+1), L-(2*j+2)])

            qc_cU_ins.x(L)
            for j in range(L-1, -1, -1):
                if j % 2 == 1:
                    qc_cU_ins.cy(L, j)
                else:
                    qc_cU_ins.cz(L, j)
            qc_cU_ins.x(L)
            
            for l in range(nsteps):
                qc_cU_ins.append(qc_cU.to_gate(), [m for m in range(L+1)])

  
            qc_cU_ins.x(L)
            for j in range(L-1, -1, -1):
                if j % 2 == 1:
                    qc_cU_ins.cy(L, j)
                else:
                    qc_cU_ins.cz(L, j)
            qc_cU_ins.x(L)
            if c2 !=  0:
                # This makes sign wrong sometimes! Fix IT!
                # Make qc_cU_ins shifted!!
                qc_cU_ins.cp(-c2, L, 0)
                qc_cU_ins.x(0)
                qc_cU_ins.cp(-c2, L, 0)
                qc_cU_ins.x(0)
            qc_cU = qc_cU_ins
        elif trotterized_time_evolution is not None:
            t = 2*nsteps * t
            print("t: ", t)
            qc_U = trotterized_time_evolution(J, h, pi, lamb, beta, t, L)
            if c2 !=  0:
                qc_U.p(-c2, 0)
                qc_U.x(0)
                qc_U.p(-c2, 0)
                qc_U.x(0)

            if hamil is not None and c2==0:
                backend2 = Aer.get_backend("unitary_simulator")
                qc_unit = execute(transpile(qc_U), backend2).result().get_unitary(qc_U, L).data
                U = scipy.linalg.expm(-1j * t * hamil)
                print("U Error: ", np.linalg.norm(qc_unit-U, ord=2))
            qc_cU = qiskit.QuantumCircuit(L+1)
            qc_cU.append(qc_U.to_gate().control(), [L] + [i for i in range(L)])
        elif tfim_2D_trotter is not None:
            qc_cU = tfim_2D_trotter(t, 0, L, J, g)
        elif heisenberg_trotter is not None:
            qc_cU = heisenberg_trotter(L, J[0], J[1], J[2], t, 0, coeffs, False)
        elif J1J2_2D_trotter is not None:
            qc_cU = J1J2_2D_trotter(t, 0, L, J[0], J[1])
        else:
            nsteps = 1
            if t > 1:
                nsteps = int(np.ceil(t))
            hloc = construct_ising_local_term(J, g)
            print("nsteps: ", nsteps)
            controlled_trotterized_time_evolution(
                qc_cU, coeffs, hloc, t/(nsteps) if control else t/(2*nsteps), 
                L, nsteps, hamil, control)
            
        if bayesian is not None:
            qc_bayesian = qc_BPE(L, state, qc_cU, mid_cbits=mid_cbits, param=bayesian)
        else:
            qpe_real, qpe_imag = qc_QPE(L, state, qc_cU, mid_cbits=mid_cbits)

        count_ops = 0
        if get_cx:
            dag = circuit_to_dag(transpile(qpe_real, basis_gates=noise_model.basis_gates+['unitary', 'initialize']))
            count_ops = dag.count_ops_longest_path() if longest_path else dag.count_ops()

        print("getting counts")
        if bayesian is None:
            counts_real = execute(transpile(qpe_real), backend, noise_model=noise_model, shots=shots).result().get_counts()
            counts_imag = execute(transpile(qpe_imag), backend, noise_model=noise_model, shots=shots).result().get_counts()
        else:
            counts = execute(transpile(qc_bayesian), backend, noise_model=noise_model, shots=shots).result().get_counts()
            if get_sv:
                sv = execute(transpile(qc_bayesian), Aer.get_backend('statevector_simulator')).result().get_statevector().data
                return sv, counts
            elif qasm:
                return counts, qc_bayesian.qasm()
            else:
                return counts

        if return_counts:
            if get_cx:
                if qasm:
                    counts_list.append((counts_real, counts_imag, count_ops, (qpe_real.qasm(), qpe_imag.qasm())))
                else:
                    counts_list.append((counts_real, counts_imag, count_ops))
            else:
                counts_list.append((counts_real, counts_imag))
        else:
            try:
                phase_est_real = counts_real["0"]/shots - counts_real["1"]/shots
            except KeyError:
                continue
            
            try:
                phase_est_imag = counts_imag["0"]/shots - counts_imag["1"]/shots
            except KeyError:
                continue
            phase_est = phase_est_real + 1j*phase_est_imag
            phase_estimates_with_noise.append((t, phase_est))
            exact_phase = np.exp(-1j*t*eigenvalues_sort[0])
            phase_exacts.append((t, exact_phase))
    if return_counts:
        return counts_list
    else:
        return phase_estimates_with_noise, phase_exacts



def qc_QPE(L, qpe_real, qc_cU, mid_cbits=0):
    #qpe_real = qiskit.QuantumCircuit(L+1, 1)
    #qpe_real.initialize(initial_state)
    qpe_real.h(L)
    qpe_imag = qpe_real.copy()
    qpe_imag.append(qc_cU.to_gate(), [i for i in range(L+1)])
    qpe_real.append(qc_cU.to_gate(), [i for i in range(L+1)])
    qpe_imag.p(-0.5*np.pi, L)
    qpe_imag.h(L)
    qpe_real.h(L)
    qpe_real.measure(L, mid_cbits)
    qpe_imag.measure(L, mid_cbits)
    return qpe_real, qpe_imag


def qc_BPE(L, qc, qc_cU, mid_cbits=0, param=0):
    qc.h(L)
    qc.append(qc_cU.to_gate(), [i for i in range(L+1)])
    qc.rz(param, L)
    qc.h(L)
    qc.measure(L, mid_cbits)
    return qc


def qc_QPE_noisy_sim(L, qc_qetu, qc_cU, qetu_repeat=3):
    qpe_real = qiskit.QuantumCircuit(L+1, 1)

    backend = Aer.get_backend("statevector_simulator")
    for i in range(qetu_repeat):
        qpe_real.append(qc_qetu.to_gate(), [i for i in range(L+1)])
        bR = execute(transpile(qpe_real), backend).result().get_statevector().data
        aR = np.kron(np.array([[1,0],[0,0]]), np.identity(2**L)) @ bR
        aR = aR / np.linalg.norm(aR)
        statePrep_Gate = StatePreparation(aR, label='meas0')
        qpe_real.reset([i for i in range(L+1)])
        qpe_real.append(statePrep_Gate, [i for i in range(L+1)])

    qpe_real.h(L)
    qpe_real.append(qc_cU.to_gate(), [i for i in range(L+1)])
    qpe_imag = qpe_real.copy()
    qpe_imag.p(-0.5*np.pi, L)
    qpe_imag.h(L)
    qpe_real.h(L)
    qpe_real.measure(L, 0)
    qpe_imag.measure(L, 0)
    return qpe_real, qpe_imag


def controlled_trotterized_time_evolution(qc_state, coeffs, hloc, dt, L, nsteps=1, hamil=None, control=True):
    if not control:
        qc = qiskit.QuantumCircuit(L)
        qc_state.x(L)
        for j in range(L-1, -1, -1):
            if j % 2 == 1:
                qc_state.cy(L, j)
            else:
                qc_state.cz(L, j)
        qc_state.x(L)


        for n in range(nsteps):
            Vlist = [scipy.linalg.expm(-1j*c*dt*hloc) for c in coeffs]
            Vlist_gates = []
            for V in Vlist:
                qc2 = qiskit.QuantumCircuit(2)
                qc2.unitary(V, [0, 1], label='str')
                Vlist_gates.append(qc2)
            perms = [None if i % 2 == 0 else np.roll(range(L), -1) for i in range(len(coeffs))]
            for layer, qc_gate in enumerate(Vlist_gates):
                for j in range(L//2):
                    if perms[layer] is not None:
                        qc.append(qc_gate.to_gate(), [L-(perms[layer][2*j]+1), L-(perms[layer][2*j+1]+1)])
                    else:
                        qc.append(qc_gate.to_gate(), [L-(2*j+1), L-(2*j+2)])
        
        qc_state.append(qc.to_gate(), [i for i in range(L)])

        if hamil is not None:
            backend = Aer.get_backend("unitary_simulator")
            qc_unit = execute(transpile(qc), backend).result().get_unitary(qc, L).data
            U = scipy.linalg.expm(-1j * dt * nsteps * hamil)
            print("U Error: ", np.linalg.norm(qc_unit-U, ord=2))    


        qc_state.x(L)
        for j in range(L-1, -1, -1):
            if j % 2 == 1:
                qc_state.cy(L, j)
            else:
                qc_state.cz(L, j)
        qc_state.x(L)
    else:
        qc = qiskit.QuantumCircuit(L+1)
        for n in range(nsteps):
            Vlist = [scipy.linalg.expm(-1j*c*dt*hloc) for c in coeffs]
            Vlist_gates = []
            for V in Vlist:
                qc2 = qiskit.QuantumCircuit(2)
                qc2.unitary(V, [0, 1], label='str')
                Vlist_gates.append(qc2)
            perms = [None if i % 2 == 0 else np.roll(range(L), -1) for i in range(len(coeffs))]
            for layer, qc_gate in enumerate(Vlist_gates):
                for j in range(L//2):
                    if perms[layer] is not None:
                        qc.append(qc_gate.to_gate().control(), [L, L-(perms[layer][2*j]+1), L-(perms[layer][2*j+1]+1)])
                    else:
                        qc.append(qc_gate.to_gate().control(), [L, L-(2*j+1), L-(2*j+2)])
        qc_state.append(qc.to_gate(), [i for i in range(L+1)])


def construct_ising_local_term(J, g):
    X = np.array([[0.,  1.], [1.,  0.]])
    Z = np.array([[1.,  0.], [0., -1.]])
    I = np.identity(2)
    return J*np.kron(Z, Z) + g*0.5*(np.kron(X, I) + np.kron(I, X))

