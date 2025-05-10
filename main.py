import portfolio
import copy
import pandas as pd
def run(input_data,solver_params,extra_arguments):
    from_=input_data['from']
    to_=input_data['to']
    assets=input_data['assets']
    dfs=[]
    for asset in assets:
        dfs.append(pd.Series(assets[asset]['history'],name=asset))
    df=pd.concat(dfs,axis=1).bfill().pct_change()
    ini=1
    results={}
    results[max([x for x in df.index.tolist() if x<from_])]=ini
    for t in [ x for x in df.index.tolist() if x>=from_ and x<=to_]:
        d={}
        d['num_assets']=input_data['num_assets']
        assets_=copy.deepcopy(assets)
        for asset in assets_:
            assets_[asset]['history']={x:assets_[asset]['history'][x] for x in assets_[asset]['history'] if x<t}
        d['assets']=assets_
        d['evaluation_date']=t
        r=portfolio.run(d)['selected_assets_weights']
        gain=0
        for asset in r:
            gain=gain+(df[asset][t]*r[asset])
        ini=ini*(1+gain)
        results[t]=ini
    res={}
    res['results']=results
    res['final_result']=ini
    return res
    
