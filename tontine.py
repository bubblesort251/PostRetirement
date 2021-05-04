import numpy as np
import pandas as pd
import longevity_lib as lon
import scipy
import math

# models a traditional tontine where a group of N people pool together POOL moneys and invest it in a risk−free asset. Calculates the mortality of the members of the tontine using MORTALITY_RATES
#(tuple of male mortality and female mortality ) , PMALE ( probability of being born male in US), and age of everyone in tontine. Returns the expected wealth and yearly payouts of an individual
def tontine(pool, n, longevity, mortality_rates, pmale, age_0, ir):
    payout = []
    contribution = pool/n
    max_t = int(max(longevity)-age_0)
    # for each scenario , generated table of longevity for the individual
    for i in range(len(longevity)):
        wealth = pool
        payout_s = [None]*max_t # array of payouts for each scenario/longevity
        
        # generate each tontine member’s profile
        rands = np.random.uniform(size=n-1)
        sex = 1*(rands > pmale) # 0 male and 1 for female
        members = np.array(list(map(lambda i: lon.mortality(age_0, mortality_rates[sex[i]]), range(len(sex)))))
        
        # model the tontine pool for each year
        total_dead = 0
        for t in range(age_0, int(longevity[i])):
            income = wealth*(ir[i,t-age_0]) # grow by 1−yr treasury bond yield curve rate 
            wealth = wealth+income
            if total_dead < n-1:
                dead = np.count_nonzero(members == t) # find number of members dead 
                total_dead += dead
                drawdown = max(0, income + contribution*dead)
                wealth -= drawdown
            elif total_dead == n-1:
                drawdown = wealth
            else:
                break
            payout_s[t-age_0] = drawdown/(n-total_dead)
        # append payout for each tontine scenario
        payout.append(payout_s)
    payout = np.array(payout)
    return payout


def tontine2(pool, n, longevity, mortality_rates, pmale, age_0, ir):
    payout = []
    payout_proportion = []
    contribution = pool/n
    max_t = int(max(longevity)-age_0)
    # for each scenario , generated table of longevity for the individual
    for i in range(len(longevity)):
        wealth = pool
        payout_s = [None]*max_t # array of payouts for each scenario/longevity
        payout_proportion_s = [None]*max_t
        
        # generate each tontine member’s profile
        rands = np.random.uniform(size=n-1)
        sex = 1*(rands > pmale) # 0 male and 1 for female
        members = np.array(list(map(lambda i: lon.mortality(age_0, mortality_rates[sex[i]]), range(len(sex)))))
        
        # model the tontine pool for each year
        total_dead = 0
        for t in range(age_0, int(longevity[i])):
            income = wealth*(ir[i,t-age_0]) # grow by 1−yr treasury bond yield curve rate 
            wealth = wealth+income
            if total_dead < n-1:
                dead = np.count_nonzero(members == t) # find number of members dead 
                total_dead += dead
                drawdown = max(0, income + contribution*dead)
                wealth -= drawdown
            elif total_dead == n-1:
                drawdown = wealth
            else:
                break
            payout_s[t-age_0] = drawdown/(n-total_dead)
            payout_proportion_s[t-age_0] = drawdown/(drawdown + wealth)
        # append payout for each tontine scenario
        payout.append(payout_s)
        payout_proportion.append(payout_proportion_s)
    payout = np.array(payout)
    payout_proportion = np.array(payout_proportion)
    return payout, payout_proportion


#Now create the CRRA utility optimal Tontine Structure
#Calculates the survivial probability of someone age x to live another t years
def t_p_x(t, x, mortality_table, gender='M'):
    #Inputs:
        #t: number of years to live
        #x: current age
        #mortality_table: pandas df, index is exact_age, and has columns "Death Probability (Gender)" for each gender

    #Outputs:
        #Probability alive at age t+x

    alive = 1.0
    for i in range(t):
        if(gender == 'M'):
            alive = alive*(1-mortality_table.loc[x+i, 'Death Probability Male'])
        else:
            alive = alive*(1-mortality_table.loc[x+i, 'Death Probability Femaale'])
    return alive


#Calculate theta_{n,gamma}(p), from theorem 2 on Optimal_retirement_income_tontines, this is an intermediate function
def theta_n_gamma_p(n, gamma, p):
    #Inputs:
        #n: number of people in tontine
        #gamma: longevity risk aversion
        #p: output of t_p_x, probability one has lived this long

    #Outputs:
        #out: float value, theta_n_gamma_p

    out = 0
    #Loop from k = 0 to k = n-1
    for k in range(n):
        out = out + scipy.special.comb(n-1, k)*((p**k))*((1-p)**(n-1-k))*((n/(k+1))**(1-gamma))
    return out

#Calculate Beta_{n,gamma}(p) an intermediate term from theorem 2 on Optimal_retirement_income_tontines
def beta_n_gamma_p(n, gamma, p):
    return p*theta_n_gamma_p(n, gamma, p)


#Fixes the mortality table so that you don't have a posibility of living past the table
def fix_mortality_table(mortality_table, gender='M'):
    mt = mortality_table.copy()

    if(gender=='M'):
        mt.loc[mt.shape[0]-1,'Death Probability Male'] = 1.0
    else:
        mt.loc[mt.shape[0]-1, 'Death Probability Female'] = 1.0

    return mt

#Calculates D^(OT)_{n, gamma}(1)
#This is formula 9 on Optimal_retirement_income_tontines
def DOT_n_gamma_p_1(rs, n, gamma, original_age, mortality_table, gender='M'):
    #INPUTS:
        #rs: array with yearly interest rates
        #n: original number of subscribers in the tontine pool
        #gamma: longevity risk aversion parameter
        #original_age: age of retirement / beginning of tontine
        # mortality_table: pandas df, index should be age

    #Outputs:   
        #out: floating point number



    #Step 2: Calculate Function values at differnet ts
    f_array = []
    for t in range(mortality_table.shape[0]-original_age+1):
        if(len(f_array)==0):
            f_array.append(1.0)
        else:
            #Initialize Array Value with Discount Rate
            f_array.append(f_array[-1]/(1+rs[t]))

    for t in range(len(f_array)):
        #Calculate survival probability to this age
        p = t_p_x(t, original_age, mortality_table, gender=gender)
        #Multiply Discount Rate by Beta(t_p_x)
        f_array[t] = f_array[t]*(beta_n_gamma_p(n, gamma, p)**(1/gamma))


    #Step 3: Perform Integration
    integral = scipy.integrate.simps(f_array, dx=1)

    return 1.0/integral

#Calculates optimal withdrawl rate from CRRA utility function
def calc_d_t(rs, n, gamma, original_age, mortality_table, gender='M'):
    #INPUTS:
        #Inputs are exactly the same as DOT_n_gamma_p_1
    #OUTPUTS:
        #dts: numpy array, with d(t) values
    #Step ):
    mt = fix_mortality_table(mortality_table, gender='gender')

    #Step 1: Calculate D^(OT)_{n, gamma}(1)
    const = DOT_n_gamma_p_1(rs, n, gamma, original_age, mt, gender='M')

    #Step 2: Calculate Varying Term
    Betas = np.array([beta_n_gamma_p(n, gamma, t_p_x(t, original_age, mt, gender=gender))**(1/gamma) for t in range(len(rs))])

    #Step 3: Multiply it
    out = Betas*const

    return out





