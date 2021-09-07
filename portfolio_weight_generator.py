import numpy as np
import pandas as pd
import longevity_lib as lon
import scipy
import math
import time
from numpy.linalg import inv,pinv
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def fix_mortality_table(mortality_table):
    mt = mortality_table.copy()
    mt.loc[mt.shape[0]-1,'Death Probability Male'] = 1.0
    mt.loc[mt.shape[0]-1, 'Death Probability Female'] = 1.0

    return mt


#calculates dt
class WeightCalculator:
    def __init__(self, method, age_0, last_age, gender='M', optional_dict=None):
        self.age_0 = age_0
        self.gender = 'M'
        self.method = method
        self.last_age = last_age
        self.optional_arguments = dict()
        for key, value in optional_dict.items():
            self.optional_arguments[key] = value
        self.gen_weights()


    def gen_weights(self):
        if(self.method == 'fixed mix'):
            me = self.optional_arguments['equity proportion']
            mt = self.optional_arguments['tontine proportion']
            mb = 1-me - mt
            mult_e = me/.8
            target = np.zeros((self.last_age - self.age_0 + 1, 10, self.optional_arguments['n sim']))
            for i in range(self.last_age+1-self.age_0):
                target[i,:,0] = [mult_e*.4, mult_e*.1, mult_e*.2, .5*mb, 0, .5*mb, mult_e*.1, 0, 0, mt]
            for i in range(1, self.optional_arguments['n sim']):
                target[:,:,i] = target[:,:,0]

            self.target = target

        if(self.method == 'glide path'):
            cond_exposure_tontine = self.optional_arguments['tontine proportion of bond exposure']

            target = np.zeros((self.last_age - self.age_0 + 1, 10, self.optional_arguments['n sim']))

            #Sweep over ages
            for i in range(self.last_age+1-self.age_0):
                age = i + self.age_0
                prop_equity = max((100.-age)/100, 0)
                prop_bonds = (1 - prop_equity)*(1-cond_exposure_tontine)
                prop_tontine = (1-prop_equity)*cond_exposure_tontine

                target[i,:,0] = [prop_equity*.4/.8, prop_equity*.1/.8, prop_equity*.2/.8, .5*prop_bonds, 0, .5*prop_bonds, prop_equity*.1/.8, 0, 0, prop_tontine]

            for i in range(1, self.optional_arguments['n sim']):
                target[:,:,i] = target[:,:,0]

            self.target = target


        if(self.method == 'risk parity no tontine'):
            covar = self.optional_arguments['covariance matrix']
            inds = self.optional_arguments['indices of assets']
            weights = calc_risk_parity_weights(covar)
            
            target = np.zeros((self.last_age - self.age_0 + 1, 10, self.optional_arguments['n sim']))
            
            temp_weights = calc_risk_parity_weights(self.optional_arguments['covariance matrix'].iloc[inds, inds]).values.transpose()
            
            weights = np.zeros((1,10))
            weights[0,inds] = temp_weights
            for i in range(self.last_age + 1 - self.age_0):
                target[i,:,0] = weights[0,:]
            for i in range(1, self.optional_arguments['n sim']):
                target[:,:,i] = target[:,:,0]
            self.target = target


        if(self.method == 'markowitz no tontine'):
            covar = self.optional_arguments['covariance matrix']
            expected_returns = self.optional_arguments['expected returns']
            inds = self.optional_arguments['indices of assets']
            cols = list(covar.columns)
            list_of_cols = [cols[i] for i in inds]

            expected_returns = expected_returns.loc[list_of_cols, list(expected_returns.columns)]
            covar = covar.loc[list_of_cols, list_of_cols]

            target = np.zeros((self.last_age - self.age_0 + 1, 10, self.optional_arguments['n sim']))

            portfolio = calc_efficient_frontier(expected_returns, covar, interestRate=-.90/100./12., plotFrontier=False, returnEfficientFrontier=False, equalityConstraint=True)

            weights = np.zeros((1,10))
            weights[0,inds] = portfolio.values[:,0]
            for i in range(self.last_age + 1 - self.age_0):
                target[i,:,0] = weights[0,:]
            for i in range(1, self.optional_arguments['n sim']):
                target[:,:,i] = target[:,:,0]
            self.target = target
            


        if(self.method == 'risk parity with tontine'):
            self.create_log_fact()
            covar = self.optional_arguments['covariance matrix']
            inds = self.optional_arguments['indices of assets']
            nsim = self.optional_arguments['longevity table'].shape[0]
            out = np.zeros((self.last_age - self.age_0 + 1, 10, nsim))

            #Loop over the simulations
            for sim in range(nsim):
                if(sim % 100 == 0):
                    print('On Sim ' + str(sim))
                for age in range(self.age_0, self.last_age + 1):
                    #if you have already died, it doesn't matter
                    if(self.optional_arguments['longevity table'].loc[sim, age] == -1):
                        temp_weights = calc_risk_parity_weights(self.optional_arguments['covariance matrix'].iloc[inds, inds]).values.transpose()
                        out[age-self.age_0:, inds, sim] = temp_weights
                        break

                    else:
                        #1 Extract Death Probaiblity
                        p_death = self.extract_p(age)
                        #2 Extract Volatility One Dies Based on P and number of people
                        vol, er = self.calculate_vol(sim, age, 1.-p_death)
                        #add to covariance matrix
                        n = len(inds)+1
                        cols = list(self.optional_arguments['covariance matrix'].columns[inds])
                        covar = pd.DataFrame(np.zeros((n,n)), columns=cols+['Tontine'], index=cols+['Tontine'])
                        covar.loc[cols, cols] = self.optional_arguments['covariance matrix']
                        covar.loc['Tontine', 'Tontine'] = vol
                        #call risk parity
                        temp_weights = calc_risk_parity_weights(covar).values.transpose()
                        out[age-self.age_0:, inds + [9], sim] = temp_weights

            self.target = out


        if(self.method == 'markowitz with tontine'):
            self.create_log_fact()
            covar = self.optional_arguments['covariance matrix']
            expected_returns = self.optional_arguments['expected returns']
            inds = self.optional_arguments['indices of assets']
            nsim = self.optional_arguments['longevity table'].shape[0]
            cols = list(covar.columns)
            list_of_cols = [cols[i] for i in inds]

            expected_returns = expected_returns.loc[list_of_cols, list(expected_returns.columns)]
            covar = covar.loc[list_of_cols, list_of_cols]

            out = np.zeros((self.last_age - self.age_0 + 1, 10, nsim))

            #Loop over the simulations

            t = time.time()

            for sim in range(nsim):
                if(sim % 100 == 0):
                    print('On Sim ' + str(sim))
                for age in range(self.age_0, self.last_age + 1):
                    #if you have already died, it doesn't matter
                    if(self.optional_arguments['longevity table'].loc[sim, age] == -1):
                        temp_weights = calc_efficient_frontier(expected_returns, covar, interestRate=-.90/100./12., equalityConstraint=True, plotFrontier=False)
                        out[age-self.age_0:, inds, sim] = temp_weights.values[:,0]
                        break

                    else:
                        #1 Extract Death Probaiblity
                        p_death = self.extract_p(age)
                        #2 Extract Volatility One Dies Based on P and number of people
                        vol, er = self.calculate_vol(sim, age, 1.-p_death)
                        #add to covariance matrix
                        n = len(list_of_cols)+1
                        #Adjust covariance matrix
                        covar_temp = pd.DataFrame(np.zeros((n,n)), columns=list_of_cols+['Tontine'], index=list_of_cols+['Tontine'])
                        covar_temp.loc[list_of_cols, list_of_cols] = covar.loc[list_of_cols, list_of_cols]
                        covar_temp.loc['Tontine', 'Tontine'] = vol
                        #Adjust Expected Returns for this
                        return_df = pd.DataFrame(np.zeros((n,1)), columns=['Returns'], index=list_of_cols+['Tontine'])
                        return_df.loc[list_of_cols, 'Returns'] = expected_returns.loc[list_of_cols, 'Returns']
                        return_df.loc['Tontine', 'Returns'] = er


                        #call risk parity
                        #if(age == 81):
                        #    print(return_df)
                        #    print(covar_temp)
                        #    w,v=np.linalg.eig(covar_temp)
                        #    print(w)
                        temp_weights = calc_efficient_frontier(return_df, covar_temp, interestRate=-.90/100./12., equalityConstraint=True, plotFrontier=False)
                        out[age-self.age_0:, inds + [9], sim] = temp_weights.values[:,0]
            self.target = out
            # do stuff
            elapsed = time.time() - t
            print(elapsed)

    def extract_p(self, age):
        if(self.gender == 'M'):
            return self.optional_arguments['mortality table'].loc[age, 'Death Probability Male']
        else:
            return self.optional_arguments['mortality table'].loc[age, 'Death Probability Female']


    def create_log_fact(self):
        n = int(self.optional_arguments['tontine table'].iloc[0,0])

        d = dict()
        out = 0.00
        for i in range(1,n+1):
            out = out + np.log(i)
            d[i] = out
        d[0] = 0
        self.log_fact_dict = d

    def calculate_vol(self, sim, age, p_alive):
        e_r_2 = 0.
        e_r = 0.

        n = int(self.optional_arguments['tontine table'].loc[sim, age])
        for k in range(n):
            e_r = e_r + self.iter_r_n_k(k, n, p_alive)*((n+1)/(k+1)-1)
            e_r_2 = e_r_2 + self.iter_r_n_k(k, n, p_alive)*(((n+1)/(k+1)-1)**2)

        vol = np.sqrt(e_r_2 - e_r**2)/np.sqrt(12)
        return vol, e_r/12.



    def iter_r_n_k(self, k, n, p):
        #P is probability of living
        out = 0.00
        out = self.log_fact_dict[n] - self.log_fact_dict[int(k)] - self.log_fact_dict[int(n-k)]
        out = out + k*np.log(p) + (n-k)*np.log(1-p)
        out = np.exp(out)
        return out

def store_weights_matrix(weights, path):
    out = pd.DataFrame(np.zeros((weights.shape[0]*weights.shape[2], weights.shape[1])))
    for i in range(weights.shape[2]):
        out.loc[i*weights.shape[0]:(i+1)*weights.shape[0]-1,:] = weights[:,:,i]
    out.to_csv(path, index=False)

def load_weights_matrix(path, n_years):
    data = pd.read_csv(path)
    out = np.zeros((n_years, data.shape[1], int(data.shape[0]/n_years)))
    for i in range(out.shape[2]):
        out[:,:,i] = data.loc[i*n_years:(i+1)*n_years-1, :]
    return out


#The code below calculates the Risk parity Weights
# risk budgeting optimization
def calculate_portfolio_var(w,V):
    # function that calculates portfolio risk
    w = np.matrix(w)
    return (w*V*w.T)[0,0]

def calculate_risk_contribution(w,V):
    # function that calculates asset contribution to total risk
    w = np.matrix(w)
    sigma = np.sqrt(calculate_portfolio_var(w,V))
    # Marginal Risk Contribution
    MRC = V*w.T
    # Risk Contribution
    RC = np.multiply(MRC,w.T)/sigma
    return RC

def risk_budget_objective(x,pars):
    # calculate portfolio risk
    V = pars[0]# covariance table
    x_t = pars[1] # risk target in percent of portfolio risk
    sig_p =  np.sqrt(calculate_portfolio_var(x,V)) # portfolio sigma
    risk_target = np.asmatrix(np.multiply(sig_p,x_t))
    asset_RC = calculate_risk_contribution(x,V)
    J = sum(np.square(asset_RC-risk_target.T))[0,0] # sum of squared error
    return J

def total_weight_constraint(x):
    return np.sum(x)-1.0

def long_only_constraint(x):
    return x

def calc_risk_parity_weights(covarMatrix, targetScale = .5):
    '''def_calc_risk_parity_weights, calculates risk parity weights
    INPUTS:
        coarMatrix: pd DataFrame, covariance matrix
    OUTPUTS:
        dfWeights: pd Dataframe, weight vector
    '''
    #Scale Covariance Matrix to Avoid Rounding Errors
    normVal = np.linalg.norm(covarMatrix)
    covarMatrixScaled = targetScale/normVal*covarMatrix

    n = covarMatrixScaled.shape[0]
    x_t = [1/n for x in range(n)] # your risk budget percent of total portfolio risk (equal risk)
    w0 = x_t
    V = np.matrix(covarMatrixScaled)
    cons = ({'type': 'eq', 'fun': total_weight_constraint},
    {'type': 'ineq', 'fun': long_only_constraint})
    res= minimize(risk_budget_objective, w0, args=[V,x_t], method='SLSQP',constraints=cons, options={'disp': False})
    w_rb = np.asmatrix(res.x)
    dfWeights = pd.DataFrame(w_rb, columns=list(covarMatrixScaled.columns))
    dfWeights = dfWeights.transpose()
    dfWeights.columns = ['Weights']
    return dfWeights




def calc_efficient_frontier(dfRet, dfCov, interestRate=0, nPoints=1000, plotFrontier=True, returnEfficientFrontier=False, equalityConstraint=True):
    '''calc_efficient_frontier calculates the efficient frontier, and calculates the 1 period efficient portfolio
    INPUTS:
        dfRet: pandas df, index are asset names, column is 'Returns', 1 period return of the asset
        dfCov: pandas df, covariance matrix, should contain the same assets as dfRet
        interestRate: float, 1 period interest rate
        nPoints: number of points to calculate on the efficient frontier
        plotFrontier: Boolean, if true, returns the efficient frontier
        returnEfficientFrontier: Boolean, if true, returns the efficient frontier code
        returnEfficientPortfolio: Boolean, if true, returns the efficient portfolio weights
        equalityConstraint: Boolean, if true, forces the sum of the weights to be 1
    OUTPUTS:
        efficientPortfolio: pandas df, sharpe ratio optimal weights
        frontier: pandas df, expected return vs volatility
    '''
    #Step 1: Set up calculation
    scale=100
    dfRet = dfRet*scale
    dfCov = dfCov*(scale**2)
    if(set(list(dfRet.index)) != set(list(dfCov.columns))):
        print('Returns DataFrame and Covariance DataFrame do not contain the same assets')
    assets = list(dfCov.columns)
    rvec = np.linspace(0,dfRet.max(),101)
    e = np.ones((len(assets),1))
    vals = np.zeros((rvec.shape[0],2))
    #Step 2: Sweep over minimum returns, calculate the efficient portfolio
    for i in range(0, rvec.shape[0]):
        r = rvec[i]
        w = cp.Variable(len(assets))
        #objective = cp.Minimize(cp.quad_form(w,np.matrix(dfCov.values)))
        objective = cp.Minimize(cp.quad_form(w,dfCov.values))
        constraints = []
        constraints.append(0 <= w)
        if(equalityConstraint):
            constraints.append(w.T@e == 1)
        else:
            constraints.append(w.T@e <= 1)
        #constraints.append(w.T*np.matrix(dfRet.values) >= r)
        constraints.append(w.T@dfRet.values >= r)
        prob = cp.Problem(objective, constraints)
        result = prob.solve(solver=cp.ECOS)
        #Compute the expected return
        #vals[i][0] = np.matrix(w.value).dot(dfRet.values)[0,0]
        vals[i][0] = np.reshape(w.value, (dfRet.shape[0], 1)).transpose().dot(dfRet.values)[0,0]
        vals[i][1] = math.sqrt(result)

    frontier = pd.DataFrame(vals, columns=['Expected Return', 'Volatility'])
    frontier = frontier/scale

    #Calculate Return of Efficient Portfolio
    frontier['Unscaled Sharpe'] = (frontier['Expected Return'] - interestRate)/frontier['Volatility']
    expectedReturnTarget = frontier.loc[frontier['Unscaled Sharpe'].idxmax(), 'Expected Return']

    r = expectedReturnTarget*scale
    w = cp.Variable(len(assets))
    objective = cp.Minimize(cp.quad_form(w,dfCov.values))
    constraints = []
    constraints.append(0 <= w)
    if(equalityConstraint):
        constraints.append(w.T@e == 1)
    else:
        constraints.append(w.T@e <= 1)
    #constraints.append(w.T*np.matrix(dfRet.values) >= r)
    constraints.append(w.T@dfRet.values >= r)
    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver=cp.ECOS)

    efficientPortfolio = pd.DataFrame(w.value, columns=['Efficient Portfolio Weights'], index=list(dfRet.index))
    efficientPortfolio = efficientPortfolio/efficientPortfolio.sum()

    if(plotFrontier):
        plt.plot(frontier['Volatility'], frontier['Expected Return'])
        plt.plot(expectedReturnTarget, frontier.loc[frontier['Unscaled Sharpe'].idxmax(), 'Volatility'])
        plt.xlabel('Volatility')
        plt.ylabel('Expected Return')
        plt.title('Efficient Frontier')
        plt.ylim([min(frontier['Expected Return'].min(), 0),1.1*frontier['Expected Return'].max()])
        plt.show()

    
    if(returnEfficientFrontier):
        return efficientPortfolio, frontier
    else:
        return efficientPortfolio


