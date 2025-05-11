import qiskit
import numpy as np
from scipy.linalg import expm
import cvxpy as cp
from numpy import polynomial as P
from pyqsp.angle_sequence import QuantumSignalProcessingPhases
import rqcopt as oc
import scipy
from qiskit.transpiler.passes import GatesInBasis
from qiskit.converters import circuit_to_dag
from qiskit import QuantumCircuit

# Below are the functions to randomize a diagonal hamiltonian
# and QETU circuit using directly controlled time evolution block.
# Implemented with numpy.

unnorm_Hadamard = np.array([[1, 1],[1, -1]])
ket_0 = np.array([1, 0])



def construct_ising_local_term(J, g):
    """
    Construct local interaction term of Ising Hamiltonian on a one-dimensional
    lattice for interaction parameter `J` and external field parameter `g`.
    """
    # Pauli-X and Z matrices
    X = np.array([[0.,  1.], [1.,  0.]])
    Z = np.array([[1.,  0.], [0., -1.]])
    I = np.identity(2)
    return J*np.kron(Z, Z) + g*0.5*(np.kron(X, I) + np.kron(I, X))


# Quantum Signal Processing step to optimize the phi values.
def get_phis(x, d, h, c=0.99, reverse=False):
    assert d % 2 == 0
    a_list = np.linspace(-1, 1, 101)
    
    poly_even_step, even_coeffs = approx_polynomial(500, d, x-h, x+h, c, 0.01, reverse=reverse)
    poly_even_step  = poly_even_step.convert(kind=P.Polynomial)
    
    ket_0 = np.array([1,0])
    
    # TODO: Degrees of freedom is actually d/2! With that we can get much smoother functions!
    phi_primes    = QuantumSignalProcessingPhases(poly_even_step, signal_operator="Wx")
    phi_primes[0] = (phi_primes[-1] + phi_primes[0])/2
    phi_primes[-1] = phi_primes[0]
    
    """qsp_polynom = [poly_even_step(a) for a in a_list]
    plt.plot(a_list, qsp_polynom, label="Polynom")
    qsp_polynom = [np.vdot(ket_0, U(phi_primes, a)@ket_0).real for a in a_list]
    plt.plot(a_list, qsp_polynom, "--", label=r"$Re{\langle0|U_{\phi}|0\rangle}$")
    qsp_polynom = [np.vdot(ket_0, U(phi_primes, a)@ket_0).imag for a in a_list]
    plt.plot(a_list, qsp_polynom, "--", label=r"$Im{\langle0|U_{\phi}|0\rangle}$")
    plt.legend()"""
    
    phis = [phi_prime + np.pi/4 if i==0 or i==len(phi_primes)-1 else phi_prime + np.pi/2 for i, phi_prime in enumerate(phi_primes)]
    return (phis, phi_primes, poly_even_step)


# Functions for approximating the even step function with polynomials.
def Cheby_polyf(x: [float], k: int):
    x = np.array(x)
    y = np.arccos(x)
    ret = np.cos((2*k)*y)
    return ret


def approx_polynomial(M, d, sigma_minus, sigma_plus, c, eps, reverse=False,  x_list_arg=None):
    assert d%2 == 0
    x = []
    for j in range(M):
        xj = -np.cos(j*np.pi/(M-1))
        if np.abs(xj)<= sigma_minus or np.abs(xj) >= sigma_plus:
            x.append(xj)
    
    def cost_func(x_c, coeff, c):
        A = []
        for k in range(int(coeff.shape[0])):
            A.append(Cheby_polyf(x_c, k))
        A = np.array(A)
        A = A.transpose()
        
        b = [0 for _ in range(len(x_c))]
        for j, xj in enumerate(x_c):
            if reverse:
                if np.abs(xj) >= sigma_minus:
                    b[j] = 0
                else:
                    b[j] = c
            else:
                if np.abs(xj) <= sigma_minus:
                    b[j] = 0
                else:
                    b[j] = c
        b = np.array(b)
        return (A @ coeff) - b
    
    x_list = np.linspace(-1, 1, 101)
    if x_list_arg:
        x_list = x_list_arg
    
    coeff = cp.Variable(int(d/2))
    constraints = [np.sum([ck*Cheby_polyf([x_i], k)[0] for k,ck in enumerate(coeff)]) <= c-eps for x_i in x_list]
    constraints += [np.sum([ck*Cheby_polyf([x_i], k)[0] for k,ck in enumerate(coeff)]) >= eps for x_i in x_list]

    objective = cp.Minimize(cp.sum_squares(cost_func(x, coeff, c)))
    
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    coeffs = coeff.value

    func = [(  np.sum([ck*Cheby_polyf([x], k)[0] for k,ck in enumerate(coeffs)])  ) for x in x_list]
    for f in func:
        if f < 0 or f > 1: 
            raise Exception("Found polynomial exceeds the [0,1] Interval!")
    
    return P.chebyshev.Chebyshev([coeffs[int(i/2)] if i%2==0 else 0 for i in range(2*len(coeffs)-1)]), coeffs


def qc_U(two_qubit_gates, L, perms):
    U = qiskit.QuantumCircuit(L)
    for layer, qc_gate in enumerate(two_qubit_gates):
        assert L%2 == 0
        for j in range(L//2):
            if perms[layer] is not None:
                U.append(qc_gate.to_gate(), [perms[layer][2*j], perms[layer][2*j+1]])
            else:
                U.append(qc_gate.to_gate(), [2*j, 2*j+1])
    return U


def qc_U_Strang(L, J, g, t, nsteps):
    U = qiskit.QuantumCircuit(L)
    
    dt = t/nsteps
    hloc = construct_ising_local_term(J, g)
    coeffs = oc.SplittingMethod.suzuki(2, 1).coeffs
    #strang = oc.SplittingMethod.suzuki(2, 1)
    #_, coeffs = oc.merge_layers(2*strang.indices, 2*strang.coeffs)
    #coeffs = [0.5*c for c in coeffs]

    Vlist = [scipy.linalg.expm(-1j*c*dt*hloc) for c in coeffs]
    Vlist_gates = []
    for V in Vlist:
        #decomp = TwoQubitBasisDecomposer(gate=CXGate())
        #qc = decomp(V)
        qc = qiskit.QuantumCircuit(2)
        qc.unitary(V, [0, 1], label='str')
        Vlist_gates.append(qc)
    perms = [None if i % 2 == 0 else np.roll(range(L), -1) for i in range(len(coeffs))]

    for layer, qc_gate in enumerate(Vlist_gates):
        assert L%2 == 0
        for j in range(L//2):
            if perms[layer] is not None:
                U.append(qc_gate.to_gate(), [perms[layer][2*j], perms[layer][2*j+1]])
            else:
                U.append(qc_gate.to_gate(), [2*j, 2*j+1])
    return U


unnorm_Hadamard = np.array([[1, 1],[1, -1]])


def U_tau(H, c1=1):
    # U will be delivered in the further steps by the Riemannian optimization. 
    # For now we take it to be a black-box. The default tau is also 1/2 as the
    # control free implementation rescales the evolution time.
    expH = expm(-1j * 0.5 * c1 * H )
    return expH


def cfree_shifted_U(U_sh, dagger, c2=0):
    L = int(np.log(len(U_sh)) / np.log(2))
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])
    I = np.identity(2)
    
    X_I = X
    for j in range(L):
        X_I = np.kron(X_I, I)
        
    K = np.identity(1)
    for j in range(L):
        if j % 2 == 0:
            K = np.kron(K, Y)
        else:
            K = np.kron(K, Z)
    
    # K is controlled by the ancilla qubit, applied when qubit is 0.
    cK  = (1+0j) * np.identity(2*len(U_sh))
    offset = len(U_sh)
    for i in range(offset):
        for j in range(offset):
            cK[i][j] = K[i][j]
    
    cU = (1+0j) * np.identity(2*len(U_sh))
    U_ext = np.kron(I, U_sh)
    cU = cK @ U_ext @ cK
    
    # Shift the global phase, controlled by the ancilla.
    # TODO: Learn if this is physically realizable.
    cC2 = (1+0j) * np.identity(2*len(U_sh))
    C2 = expm(-1j*c2*np.identity(offset))
    for i in range(offset):
        for j in range(offset):
            cC2[offset+i][offset+j] = C2[i][j]
    cU = cC2 @ cU
    
    if dagger:
        cU = X_I @ cU @ X_I
    return cU


def QETU_cf(U, phis, c2=0):
    Q = S(phis[len(phis)-1], len(U))
    for i in range(1, len(phis)):
        Q = Q @ cfree_shifted_U(U, i % 2 == 1, c2) @ S(phis[len(phis)-1-i], len(U))
    return Q


def qc_cfU_R(qc_U, dagger, c2):
    L = qc_U.num_qubits
    qc_cfU = qiskit.QuantumCircuit(L+1)
    
    if dagger:
        qc_cfU.x(L)
    
    qc_cfU.x(L)
    for j in range(L-1, -1, -1):
        if j % 2 == 1:
            qc_cfU.cy(L, j)
        else:
            qc_cfU.cz(L, j)
    qc_cfU.x(L)
    
    qc_cfU.append(qc_U.to_gate(), [i for i in range(L)])
    
    qc_cfU.x(L)
    for j in range(L-1, -1, -1):
        if j % 2 == 1:
            qc_cfU.cy(L, j)
        else:
            qc_cfU.cz(L, j)
    qc_cfU.x(L)

    qc_cfU.cp(-c2, L, 0)
    qc_cfU.x(0)
    qc_cfU.cp(-c2, L, 0)
    qc_cfU.x(0)
    
    if dagger:
        qc_cfU.x(L)
    
    return qc_cfU


def qc_QETU_cf_R(qc_U, phis, c2=0):
    """
        Control Free Implementation of the QETU Circuit
        for the TFIM Hamiltonian. Encoded reversely as 
        qiskit uses Little Endian.
    """
    L = qc_U.num_qubits
    qc = qiskit.QuantumCircuit(L+1)
    qc.rx(-2*phis[0], L)
    for i in range(1, len(phis)):
        qc.append(qc_cfU_R(qc_U, i%2==0, c2).to_gate(), [i for i in range(L+1)])
        qc.rx(-2*phis[i], L)
    return qc


def qc_QETU_R(qc_cU, phis, c2=0, multi_ancilla=1):
    """
        Control Free Implementation of the QETU Circuit
        Encoded reversely as qiskit uses Little Endian.
    """
    L = qc_cU[0].num_qubits - multi_ancilla
    qc = qiskit.QuantumCircuit(L+multi_ancilla)

    for anc_ind in range(multi_ancilla):
        qc.rx(-2*phis[0], L+anc_ind)
    for i in range(1, len(phis)):
        if i%2==1:
            qc.append(qc_cU[0].to_gate(), [j for j in range(L+multi_ancilla)])
        else:
            qc.append(qc_cU[1].to_gate(), [j for j in range(L+multi_ancilla)])

        #qc.cp(-c2, L, 0)
        #qc.x(0)
        #qc.cp(-c2, L, 0)
        #qc.x(0)
        for anc_ind in range(multi_ancilla):
            qc.rx(-2*phis[i], L+anc_ind)

    #print(qc.decompose().draw())
    return qc














    

    
