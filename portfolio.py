from qiskit_aer.primitives import SamplerV2
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import pandas as pd
import random
def dummy_circuit_with_Aer():
    i=5
    circ = QuantumCircuit(i)
    circ.h(0)
    
    for j in range(1,i):
        circ.cx(0, j)
    
    circ.measure_all()

    
    # Construct a simulator with Aer
    # You can choose one of this simulation methods:
    # automatic (by default), statevector, density_matrix, stabilizer, extended_stabilizer, matrix_product_state, unitary and superop
    # https://qiskit.github.io/qiskit-aer/tutorials/1_aersimulator.html
    
    simulator = AerSimulator(method='statevector')
    circ = transpile(circ, simulator)
    job = simulator.run([circ], shots=128)
    
    # Perform a simulation
    result = job.result()
    counts_ideal = result.get_counts(circ)
    #print(i,'Counts(ideal):', counts_ideal) 

def dummy_circuit_with_Sampler():
    i=5
    circ = QuantumCircuit(i)
    circ.h(0)
    
    for j in range(1,i):
        circ.cx(0, j)
    
    circ.measure_all()
    
    # Construct an ideal simulator with SamplerV2
    sampler = SamplerV2()
    job = sampler.run([circ], shots=128)
    
    # Perform an ideal simulation
    result_ideal = job.result()
    counts_ideal = result_ideal[0].data.meas.get_counts()
    #print(i,'Counts(ideal):', counts_ideal) 
def run(input_data):

    # HowTo build circuits with Qiskit
    dummy_circuit_with_Aer()
    dummy_circuit_with_Sampler()

    # How to read input data:
    num_assets=input_data['num_assets']
    assets=input_data['assets']
    evaluation_date=input_data['evaluation_date']
    df=pd.DataFrame()
    series=[]
    for asset in assets:
        series.append(pd.Series(assets[asset]['history'],name=asset))
    df=pd.concat(series,axis=1)


    # How to return output data:
    aux={}
    for asset in assets:
        if random.random()>0.95:
            aux[asset]=random.randint(1,10000000)
    result={}
    for asset in aux:
        result[asset]=aux[asset]/sum(aux.values())
    return {'selected_assets_weights':result, 'num_selected_assets': len(result)}
