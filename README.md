# Quantum Eigenstate Filtering for Portfolio Optimization

We propose and implement a non variational, purely quantum algorithm to perform a 
"Integer Constraint", Portfolio Optimization problem - inspired by the Hamiltonian simulation
quantum algortihms. \\

We make a mapping between {0, 1} binary variables of integers and {-1, 1} spin variables of
being up and down. The inherently discrete nature of quantum states allow us to optimize 
the number of stocks an investor should buy, over a distibution of binary digits for each 
asset.\\

Our implementation showcases the purely quantum protocol over a small number of assets after
an initial filtering of many assets. This is due to the small number of qubits we can 
simulate with statevector simulations. If one wants to optimize over a larger subset of assets,
one should equivalently provide a larger qubit budget for the algorithm.\\

In our implementation the budget for the total number of qubits, the number of assets to optimize over after an initial (trivial) filtering, risk aversion parameter of the investor, an optional total budget constraint and the penalty parameter for not exceeding this budget
can be configured freely by the user. In the code, we hard-coded these parameters.\\

For the initial filtering, we compute the expected returns of each asset, sort them out and take the highest l number of assets to be considered in the quantum optimization protocol to reduce the
data set. This number l can be increased by the user if the user can also accommodate more qubits 
accordingly.

Afterwards, we compute the co-variance matrix. We map these parameters (mainly the co-variance matrix and the expected returns vector) to the parameters of a "fully connected Ising model". The ground state of this Ising model will correspond to the optimal, integer valued number of stocks one should invest in,
given the additional parameters. 

To prepare the ground state of this Ising model, we use "eigenstate filtering" through the so-called QETU sequence. See the paper [here](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.3.040305). This approach is "non-variational" and is guaranteed to deliver a result from the low energy subspace of the Hamiltonian, given that one can repeat the filtering multiple times on the given hardware/backend.

The main advantage of our approach is that, contrary to variational methods we get very precise error bounds on the outcome and the outcome is guaranteed to deliver one of the lowest energy solutions of the Hamiltonian, hence one of the optimal solutions to the portfolio problem.

Our quantum algorithm also scales optimally (see the theoretical optimum explained in [this paper](https://arxiv.org/abs/2002.12508)) with respect to the inverse spectral gap - which is in case a linear scaling.





