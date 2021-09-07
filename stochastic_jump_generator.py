#This calculates the stochastic jump data generating process
import numpy as np
import pandas as pd
import math
import random


#this function calcualtes the dws
def calc_dwts(n_periods, n_scen, dt, num_vars=2):
    return np.sqrt(dt)*np.random.normal(size=(n_periods, num_vars, n_scen))

# this function calculates the stochastic jumps, given an input arrival: lambda_p*vt
def compute_arrivals(arrival, dt, cutoff=10):
    out = np.zeros((10,))
    p = np.random.rand()
    z = 0
    for n in range(cutoff-1):
        z = z + ((arrival*dt)**n)*np.exp(-arrival*dt)/math.factorial(n)
        if(p <= z):
            return n
    return cutoff-1

#generates the joint realizations of stock prices, volatility and jumps
def generate_stocks_vol_and_jumps(r, l_1, l_p, l_q, n, k, v_bar, sigma, rho, dt, n_periods, n_scen):
    #INPUTS:
        #r: base risk free rate
        #l_1: risk compensation for volatility
        #n: parameter, sizes shocks
        #l_p: risk compensation parameter
        #l_q: different risk compensation parameter
        #k: rate of mean reversion of vol
        #v_bar: average volatility
        #sigma: volatility of volatility
        #rho: correlation coefficient
    #OUTPUTS:
        #st_mat: n_periods+1 x 3 x n_scen numpy array of St, vt, n_jumps for each period and each n_jumps
        #rst_mat: n_periods x 2 x n_scen matrix of r_st and dvt values

    st_mat = np.zeros((n_periods+1, 3, n_scen))
    rst_mat = np.zeros((n_periods, 2, n_scen))

    normal_kicks = calc_dwts(n_periods, n_scen, dt)

    for scen in range(n_scen):
        if(scen % 25 == 0):
            print('Starting Run ' + str(scen))
        #initialize S_t
        st_mat[0,0,scen] = 1
        #initialize v_t
        st_mat[0,1,scen] = v_bar
        #compute first arrival
        st_mat[0,2,scen] = compute_arrivals(l_p*st_mat[0,1,scen], dt)
        #now loop over time steps
        for i in range(1,n_periods+1):
            #Extract values to make code easier to read
            s_t = st_mat[i-1,0,scen]
            v_t = st_mat[i-1,1,scen]
            n_jump = st_mat[i-1,2,scen]

            dwt_s = normal_kicks[i-1, 0, scen]
            dwt_v = normal_kicks[i-1, 1, scen]

            #Calculate ds_t
            ds_t = s_t*((r + l_1*v_t + n*(l_p - l_q)*v_t)*dt + np.sqrt(v_t)*dwt_s + n*n_jump*dt)
            #calculate dv_t
            dv_t = k*(v_bar - v_t)*dt + sigma*np.sqrt(v_t)*(rho*dwt_s + np.sqrt(1-rho**2)*dwt_v)

            #Now, add new S_t and v_t to matrices
            st_mat[i,0,scen] = s_t + ds_t
            st_mat[i,1,scen] = max(v_t + dv_t,0)

            #Now add the deltas to the correct matrix
            rst_mat[i-1,0,scen] = ds_t/s_t
            rst_mat[i-1,1,scen] = dv_t

            #Now calculate new jump for this time step
            st_mat[i,2,scen] = compute_arrivals(l_p*v_t, dt)

    return st_mat, rst_mat


#generates the joint realizations of stock prices, volatility and jumps
def generate_stocks_vol_and_jumps_w_bonds(r, l_1, l_p, l_q, n, k, v_bar, sigma, rho, dt, n_periods, n_scen, mu_bonds, sigma_bonds, corr_bonds):
    #INPUTS:
        #r: base risk free rate
        #l_1: risk compensation for volatility
        #n: parameter, sizes shocks
        #l_p: risk compensation parameter
        #l_q: different risk compensation parameter
        #k: rate of mean reversion of vol
        #v_bar: average volatility
        #sigma: volatility of volatility
        #rho: correlation coefficient
    #OUTPUTS:
        #st_mat: n_periods+1 x 3 x n_scen numpy array of St, vt, n_jumps for each period and each n_jumps
        #rst_mat: n_periods x 2 x n_scen matrix of r_st and dvt values

    st_mat = np.zeros((n_periods+1, 3, n_scen))
    rst_mat = np.zeros((n_periods, 3, n_scen))

    normal_kicks = calc_dwts(n_periods, n_scen, dt, num_vars=3)

    for scen in range(n_scen):
        if(scen % 25 == 0):
            print('Starting Run ' + str(scen))
        #initialize S_t
        st_mat[0,0,scen] = 1
        #initialize v_t
        st_mat[0,1,scen] = v_bar
        #compute first arrival
        st_mat[0,2,scen] = compute_arrivals(l_p*st_mat[0,1,scen], dt)
        #now loop over time steps
        for i in range(1,n_periods+1):
            #Extract values to make code easier to read
            s_t = st_mat[i-1,0,scen]
            v_t = st_mat[i-1,1,scen]
            n_jump = st_mat[i-1,2,scen]

            dwt_s = normal_kicks[i-1, 0, scen]
            dwt_v = normal_kicks[i-1, 1, scen]
            dwt_b = normal_kicks[i-1, 2, scen]

            #Calculate ds_t
            ds_t = s_t*((r + l_1*v_t + n*(l_p - l_q)*v_t)*dt + np.sqrt(v_t)*dwt_s + n*n_jump*dt)
            #calculate dv_t
            dv_t = k*(v_bar - v_t)*dt + sigma*np.sqrt(v_t)*(rho*dwt_s + np.sqrt(1-rho**2)*dwt_v)

            #Now, add new S_t and v_t to matrices
            st_mat[i,0,scen] = s_t + ds_t
            st_mat[i,1,scen] = max(v_t + dv_t,0)

            #Now add the deltas to the correct matrix
            rst_mat[i-1,0,scen] = ds_t/s_t
            rst_mat[i-1,1,scen] = dv_t

            #Now add bonds return to correct matrix
            rst_mat[i-1,2,scen] = mu_bonds*dt + sigma_bonds*(corr_bonds*dwt_s + np.sqrt(1-corr_bonds**2)*dwt_b)

            #Now calculate new jump for this time step
            st_mat[i,2,scen] = compute_arrivals(l_p*v_t, dt)

    return st_mat, rst_mat


# Geometric Brownian Motion generator

def gen_gbm_model(n_periods, n_scen, mean, covariance):
    out = []
    np.random.seed(777)
    for i in range(n_scen):
        if(i % 50 == 0):
            print('Gen Run i = ' + str(i))
        out.append(np.random.multivariate_normal(mean, covariance, size=n_periods))

    return out
