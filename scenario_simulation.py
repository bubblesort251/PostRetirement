#This library houses the function that generates asset returns for n scenarios

import numpy as np
import pandas as pd


#Generates scenarios given
def gen_scenarios(n, age_0, r_n, Q_n, P, r_c = 0, Q_c = 0):
    #INPUTS:
        #n: number of scenarios
        #age_0: age initialization
        #r_n: Expected return during normal period
        #Q_n: Covariance Matrix of Normal period
        #P: transition matrix [[nn,nc], [cn, cc]]
        #r_c: Expected return during crash period
        #Q_c: Transition matrix during crash period

        #All matrix inputs are numpy arrays
    #Outputs:
        #scenarios:

    #Begin Function
    periods = (119 - age_0)*12
    scenarios = []
    p_n = P[0] #array of [nn, nc]
    p_c = P[1] # array of [cn, cc]
    for i in range(n):
        s=[]
        regime = 0 # 0 for normal and 1 for crash
        for t in range(periods):
            # if t is normal regime
            if regime == 0:
                # t+1 is normal regime
                if np.random.uniform() <= p_n[0]:
                    s.append(np.random.multivariate_normal(r_n, Q_n).tolist())
                    regime = 0
                # t+1 is crash regime
                else:
                    s.append(np.random.multivariate_normal(r_c, Q_c).tolist())
                    regime = 1
            # if t is crash regime
            else:
            # t+1 is normal regime
                if np.random.uniform() <= p_c[0]:
                    s.append(np.random.multivariate_normal(r_n, Q_n).tolist())
                    regime = 0
                # t+1 is crash regime
                else:
                    s.append(np.random.multivariate_normal(r_c, Q_c).tolist())
                    regime = 1
        scenarios.append(np.asarray(s))
    return scenarios
