from qiskit_aer.primitives import SamplerV2
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import pandas as pd
import random



def run(input_data):

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
