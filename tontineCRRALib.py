import numpy as np
import pandas as pd
import longevity_lib as lon
import scipy
import math
import time


def fix_mortality_table(mortality_table):
    mt = mortality_table.copy()
    mt.loc[mt.shape[0]-1,'Death Probability Male'] = 1.0
    mt.loc[mt.shape[0]-1, 'Death Probability Female'] = 1.0

    return mt

#longevity_to_survivors: takes a longevity vector, and calculates the number of survivors at a given time
#number of survivors DOES NOT INCLUDE the original person
def longevity_to_survivors(num_tontines, mortality_table, pmale, age_0, n_in_tontine):
    #INPUTS:
        #num_tontines: integer, number of tontines to calculate
        #mortality_table: pandas df, index is exact age, needs two columns Death Probability Male and Death Probabilty Female
        #pmale: probability person in tontine is male
        #age_0: start age of retirement / beginning of the death count
        #n_in_tontine: number of people in the tontine
    #Outputs:
        #alive_at_time: pandas df, num_tontines x max-age - age_0, number of alive people at a given age
    m = mortality_table.shape[0]
    out = pd.DataFrame(-1*np.ones((num_tontines, m - age_0)), columns=[i for i in range(age_0, m)])
    mt = fix_mortality_table(mortality_table)
    m_mortality = mt['Death Probability Male'].to_numpy()
    f_mortality = mt['Death Probability Female'].to_numpy()
    mortality_rates = [m_mortality, f_mortality]

    for i in range(num_tontines):
        # generate each tontine member’s profile
        rands = np.random.uniform(size=n_in_tontine)
        sex = 1*(rands > pmale) # 0 male and 1 for female
        members = np.array(list(map(lambda i: lon.mortality(age_0, mortality_rates[sex[i]]), range(len(sex)))))

        for j in range(out.shape[1]):
            a = len(members[members >= age_0 + j])
            if(a > 0):
                out.iloc[i,j] = len(members[members >= age_0 + j])
            else:
                break


    return out

#calc_d_t(rs, n, gamma, original_age, mortality_table, gender='M')

#This calculates the d(t) for every point in the table
def calculate_dt_over_time(longevity_table, original_age, r, gamma, mortality_table, gender='M'):
    #INPUTS:
        #longevity_table: pandas df, elements are number of people alive (aside from base person) at a given time, if entry is -1, then base person is dead
            #columns are the ages
        #r: float, interest rate used for discounting
        #gamma: relative risk aversion parameter


    #fix mortality table
    start = time.time()
    mt = fix_mortality_table(mortality_table)

    out = pd.DataFrame(-1*np.ones(longevity_table.shape), columns=longevity_table.columns)

    #for i in range(longevity_table.shape[0]):
    for i in range(11):
        if(i%10 == 0):
            print('has taken ' + str(time.time()-start) + ' seconds to finish ' + str(i) + ' runs')
        for j in range(out.shape[1]):
            if(longevity_table.iloc[i,j] != -1):
                #You need to calculate dt here
                #calculate r
                rs = [0] + [r for k in range(mortality_table.shape[0]-j-original_age)]
                dts = calc_d_t(rs, int(longevity_table.iloc[i,j]), gamma, original_age + j, mt, gender=gender)
                out.iloc[i,j] = dts[0]
            else:
                break
    return out

#calc_crra_tontine calculates the payout to an individual across time
def calc_crra_tontine_basic(pool, longevity_table, dt_table, ir_mat):
    #INPUTS:
        #longevity_table: output of longevity_to_survivors, pandas df, should have number of alive people at a time
        #dt_table: output of calculate_dt_over_time, pandas df, columns are age, rows are different tontines, value if d(t) for that point
        #pool: amount of dollars beginning in the tontine
        #ir_mat: interest rate matrix

    #OUTPUTS:
        #per_person_payout: output table of payouts

    per_person_payout = pd.DataFrame(-1*np.ones(longevity_table.shape), columns=longevity_table.columns)

    #Loop over tontines, and calculate payout numbers
    for i in range(longevity_table.shape[0]):
        wealth = pool
        for j in range(longevity_table.shape[1]):
            #calulate drawdown by treatment
            #if(longevity_table.iloc[i,j] == 1):
            #    drawdown = max(0, wealth)
            #    break
            if(longevity_table.iloc[i,j] > 0):
                drawdown = max(0, wealth*dt_table.iloc[i,j])
            else:
                break

            per_person_payout.iloc[i,j] = drawdown/longevity_table.iloc[i,j]

            wealth = wealth - drawdown

            #if you haven't hit the terminal condition, now increase wealth by amount
            wealth = wealth*(1 + ir_mat[i,j])

    return per_person_payout


#calc_crra_tontine_basic_real calculates the payout to an individual across time, in real terms defined by discount_mat
def calc_crra_tontine_basic_real(pool, longevity_table, dt_table, ir_mat, discount_mat):
    #INPUTS:
        #longevity_table: output of longevity_to_survivors, pandas df, should have number of alive people at a time
        #dt_table: output of calculate_dt_over_time, pandas df, columns are age, rows are different tontines, value if d(t) for that point
        #pool: amount of dollars beginning in the tontine
        #ir_mat: interest rate matrix
        #discount_mat: discount matrix

    #OUTPUTS:
        #per_person_payout: output table of payouts, real

    per_person_payout = pd.DataFrame(-1*np.ones(longevity_table.shape), columns=longevity_table.columns)
    discount_mult_mat = np.append(np.array(np.ones((1000,1))), np.cumprod(discount_mat+1, axis=1), axis=1)

    #Loop over tontines, and calculate payout numbers
    for i in range(longevity_table.shape[0]):
        wealth = pool
        for j in range(longevity_table.shape[1]):
            #calulate drawdown by treatment
            #if(longevity_table.iloc[i,j] == 1):
            #    drawdown = max(0, wealth)
            #    break
            if(longevity_table.iloc[i,j] > 0):
                drawdown = max(0, wealth*dt_table.iloc[i,j])
            else:
                break

            per_person_payout.iloc[i,j] = drawdown/longevity_table.iloc[i,j]/discount_mult_mat[i,j]

            wealth = wealth - drawdown

            #if you haven't hit the terminal condition, now increase wealth by amount
            wealth = wealth*(1 + ir_mat[i,j])

    return per_person_payout


#calc_crra_tontine calculates the payout to an individual across time
def calc_crra_tontine_basic_max_dist(pool, longevity_table, dt_table, ir_mat, r, swr):
    #INPUTS:
        #longevity_table: output of longevity_to_survivors, pandas df, should have number of alive people at a time
        #dt_table: output of calculate_dt_over_time, pandas df, columns are age, rows are different tontines, value if d(t) for that point
        #pool: amount of dollars beginning in the tontine
        #ir_mat: interest rate matrix
        #r: assumed risk free rate capping max distribution
        #swr: the maximum withdrawl rate

    #OUTPUTS:
        #per_person_payout: output table of payouts

    per_person_payout = pd.DataFrame(-1*np.ones(longevity_table.shape), columns=longevity_table.columns)
    n_people = longevity_table.iloc[0,0]

    #Loop over tontines, and calculate payout numbers
    for i in range(longevity_table.shape[0]):
        wealth = pool
        for j in range(longevity_table.shape[1]):
            #calulate drawdown by treatment
            #if(longevity_table.iloc[i,j] == 1):
            #    drawdown = max(0, wealth)
            #    break
            if(longevity_table.iloc[i,j] > 0):
                drawdown = longevity_table.iloc[i,j]*min(wealth*dt_table.iloc[i,j]/longevity_table.iloc[i,j], pool/n_people*swr*((1+r)**j))
            else:
                break

            per_person_payout.iloc[i,j] = drawdown/longevity_table.iloc[i,j]

            wealth = wealth - drawdown

            #if you haven't hit the terminal condition, now increase wealth by amount
            wealth = wealth*(1 + ir_mat[i,j])

    return per_person_payout


#calc_crra_tontine calculates the payout to an individual across time
def calc_crra_tontine_basic_max_dist_real(pool, longevity_table, dt_table, ir_mat, r, swr, discount_mat):
    #INPUTS:
        #longevity_table: output of longevity_to_survivors, pandas df, should have number of alive people at a time
        #dt_table: output of calculate_dt_over_time, pandas df, columns are age, rows are different tontines, value if d(t) for that point
        #pool: amount of dollars beginning in the tontine
        #ir_mat: interest rate matrix
        #r: assumed risk free rate capping max distribution
        #swr: the maximum withdrawl rate
        #discount_mat: discount rate matrix

    #OUTPUTS:
        #per_person_payout: output table of payouts

    per_person_payout = pd.DataFrame(-1*np.ones(longevity_table.shape), columns=longevity_table.columns)
    n_people = longevity_table.iloc[0,0]
    discount_mult_mat = np.append(np.array(np.ones((1000,1))), np.cumprod(discount_mat+1, axis=1), axis=1)

    #Loop over tontines, and calculate payout numbers
    for i in range(longevity_table.shape[0]):
        wealth = pool
        for j in range(longevity_table.shape[1]):
            #calulate drawdown by treatment
            #if(longevity_table.iloc[i,j] == 1):
            #    drawdown = max(0, wealth)
            #    break
            if(longevity_table.iloc[i,j] > 0):
                drawdown = longevity_table.iloc[i,j]*min(wealth*dt_table.iloc[i,j]/longevity_table.iloc[i,j], pool/n_people*swr*((1+r)**j))
            else:
                break

            per_person_payout.iloc[i,j] = drawdown/longevity_table.iloc[i,j]/discount_mult_mat[i,j]

            wealth = wealth - drawdown

            #if you haven't hit the terminal condition, now increase wealth by amount
            wealth = wealth*(1 + ir_mat[i,j])

    return per_person_payout


#calc_crra_tontine calculates the payout to an individual across time
def calc_crra_tontine_fm(pool, longevity_table, dt_table, scenarios, targets):
    #INPUTS:
        #longevity_table: output of longevity_to_survivors, pandas df, should have number of alive people at a time
        #dt_table: output of calculate_dt_over_time, pandas df, columns are age, rows are different tontines, value if d(t) for that point
        #pool: amount of dollars beginning in the tontine
        #scenarios: array, elements are 648 x 9 array of returns
        #targets: target whatever

    #OUTPUTS:
        #per_person_payout: output table of payouts

    per_person_payout = pd.DataFrame(-1*np.ones(longevity_table.shape), columns=longevity_table.columns)

    #Loop over tontines, and calculate payout numbers
    for i in range(longevity_table.shape[0]):
        if(i%100 == 0):
            print('Begin run ' + str(i) + ' of ' + str(per_person_payout.shape[0]))
        wealth = pool
        for j in range(longevity_table.shape[1]):
            #calulate drawdown by treatment
            #if(longevity_table.iloc[i,j] == 1):
            #    drawdown = max(0, wealth)
            #    break
            if(longevity_table.iloc[i,j] > 0):
                drawdown = max(0, wealth*dt_table.iloc[i,j])
            else:
                break

            per_person_payout.iloc[i,j] = drawdown/longevity_table.iloc[i,j]

            wealth = wealth - drawdown

            #if you haven't hit the terminal condition, now increase wealth by amount
            wealth_vals = wealth*targets[j]
            returns = np.cumprod(np.array(scenarios[i][j*12:(j+1)*12]) + 1, axis=0)
            cum_returns = returns[11,:]
            #now add it back together
            wealth = np.sum(wealth_vals[0:7]*cum_returns[0:7]) + wealth_vals[7]*(cum_returns[7]-1) + wealth_vals[8]*cum_returns[8]


    return per_person_payout


#calc_crra_tontine calculates the payout to an individual across time
def calc_crra_tontine_dynamic(pool, longevity_table, dt_table, scenarios, targets, r):
    #INPUTS:
        #longevity_table: output of longevity_to_survivors, pandas df, should have number of alive people at a time
        #dt_table: output of calculate_dt_over_time, pandas df, columns are age, rows are different tontines, value if d(t) for that point
        #pool: amount of dollars beginning in the tontine
        #scenarios: array, elements are 648 x 9 array of returns
        #targets: target whatever

    #OUTPUTS:
        #per_person_payout: output table of payouts

    per_person_payout = pd.DataFrame(-1*np.ones(longevity_table.shape), columns=longevity_table.columns)

    #Loop over tontines, and calculate payout numbers
    for i in range(longevity_table.shape[0]):
        if(i%100 == 0):
            print('Begin run ' + str(i) + ' of ' + str(per_person_payout.shape[0]))
        wealth = pool
        wealth_theo = pool
        for j in range(longevity_table.shape[1]):
            #calulate drawdown by treatment
            #if(longevity_table.iloc[i,j] == 1):
            #    drawdown = max(0, wealth)
            #    break
            if(longevity_table.iloc[i,j] > 0):
                drawdown = max(0, min(wealth_theo*dt_table.iloc[i,j], wealth))
            else:
                break

            per_person_payout.iloc[i,j] = drawdown/longevity_table.iloc[i,j]

            wealth = wealth - drawdown
            wealth_theo = wealth_theo - drawdown

            #if you haven't hit the terminal condition, now increase wealth by amount
            wealth_vals = wealth*targets[j]
            returns = np.cumprod(np.array(scenarios[i][j*12:(j+1)*12]) + 1, axis=0)
            cum_returns = returns[11,:]
            #now add it back together
            wealth = np.sum(wealth_vals[0:7]*cum_returns[0:7]) + wealth_vals[7]*(cum_returns[7]-1) + wealth_vals[8]*cum_returns[8]
            wealth_theo = wealth_theo*(1+r)


    return per_person_payout


def calc_crra_tontine_max_dist(pool, longevity_table, dt_table, scenarios, targets, r, swr):
    #INPUTS:
        #longevity_table: output of longevity_to_survivors, pandas df, should have number of alive people at a time
        #dt_table: output of calculate_dt_over_time, pandas df, columns are age, rows are different tontines, value if d(t) for that point
        #pool: amount of dollars beginning in the tontine
        #scenarios: array, elements are 648 x 9 array of returns
        #targets: target whatever

    #OUTPUTS:
        #per_person_payout: output table of payouts

    per_person_payout = pd.DataFrame(-1*np.ones(longevity_table.shape), columns=longevity_table.columns)
    n_people = longevity_table.iloc[0,0]
    #Loop over tontines, and calculate payout numbers
    for i in range(longevity_table.shape[0]):
        if(i%100 == 0):
            print('Begin run ' + str(i) + ' of ' + str(per_person_payout.shape[0]))
        wealth = pool
        for j in range(longevity_table.shape[1]):
            #calulate drawdown by treatment
            #if(longevity_table.iloc[i,j] == 1):
            #    drawdown = max(0, wealth)
            #    break
            if(longevity_table.iloc[i,j] > 0):
                drawdown = longevity_table.iloc[i,j]*min(wealth*dt_table.iloc[i,j]/longevity_table.iloc[i,j], pool/n_people*swr*((1+r)**j))
                
            else:
                break

            per_person_payout.iloc[i,j] = drawdown/longevity_table.iloc[i,j]

            wealth = wealth - drawdown

            #if you haven't hit the terminal condition, now increase wealth by amount
            wealth_vals = wealth*targets[j]
            returns = np.cumprod(np.array(scenarios[i][j*12:(j+1)*12]) + 1, axis=0)
            cum_returns = returns[11,:]
            #now add it back together
            wealth = np.sum(wealth_vals[0:7]*cum_returns[0:7]) + wealth_vals[7]*(cum_returns[7]-1) + wealth_vals[8]*cum_returns[8]


    return per_person_payout



def calc_crra_tontine_max_dist_real(pool, longevity_table, dt_table, scenarios, targets, r, swr, discount_mat):
    #INPUTS:
        #longevity_table: output of longevity_to_survivors, pandas df, should have number of alive people at a time
        #dt_table: output of calculate_dt_over_time, pandas df, columns are age, rows are different tontines, value if d(t) for that point
        #pool: amount of dollars beginning in the tontine
        #scenarios: array, elements are 648 x 9 array of returns
        #targets: target whatever
        #discount_mat: inflation matrix

    #OUTPUTS:
        #per_person_payout: output table of payouts

    per_person_payout = pd.DataFrame(-1*np.ones(longevity_table.shape), columns=longevity_table.columns)
    n_people = longevity_table.iloc[0,0]
    discount_mult_mat = np.append(np.array(np.ones((1000,1))), np.cumprod(discount_mat+1, axis=1), axis=1)

    #Loop over tontines, and calculate payout numbers
    for i in range(longevity_table.shape[0]):
        if(i%100 == 0):
            print('Begin run ' + str(i) + ' of ' + str(per_person_payout.shape[0]))
        wealth = pool
        for j in range(longevity_table.shape[1]):
            #calulate drawdown by treatment
            #if(longevity_table.iloc[i,j] == 1):
            #    drawdown = max(0, wealth)
            #    break
            if(longevity_table.iloc[i,j] > 0):
                drawdown = longevity_table.iloc[i,j]*min(wealth*dt_table.iloc[i,j]/longevity_table.iloc[i,j], pool/n_people*swr*((1+r)**j))
                
            else:
                break

            per_person_payout.iloc[i,j] = drawdown/longevity_table.iloc[i,j]/discount_mult_mat[i,j]

            wealth = wealth - drawdown

            #if you haven't hit the terminal condition, now increase wealth by amount
            wealth_vals = wealth*targets[j]
            returns = np.cumprod(np.array(scenarios[i][j*12:(j+1)*12]) + 1, axis=0)
            cum_returns = returns[11,:]
            #now add it back together
            wealth = np.sum(wealth_vals[0:7]*cum_returns[0:7]) + wealth_vals[7]*(cum_returns[7]-1) + wealth_vals[8]*cum_returns[8]


    return per_person_payout



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

    out = 0.00
    for k in range(int(n)):
        out = out + scipy.special.comb(n-1, k)*((p**k))*((1-p)**(n-1-k))*((n/(k+1))**(1-gamma))

    return out

#Calculate Beta_{n,gamma}(p) an intermediate term from theorem 2 on Optimal_retirement_income_tontines
def beta_n_gamma_p(n, gamma, p):
    return p*theta_n_gamma_p(n, gamma, p)


#Fixes the mortality table so that you don't have a posibility of living past the table
def fix_mortality_table(mortality_table):
    mt = mortality_table.copy()
    mt.loc[mt.shape[0]-1,'Death Probability Male'] = 1.0
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


    #Step 1: Calculate D^(OT)_{n, gamma}(1)

    #print('const')
    const = DOT_n_gamma_p_1(rs, n, gamma, original_age, mortality_table, gender='M')
    #print(const)

    #Step 2: Calculate Varying Term
    #print('Betas')
    Betas = np.array([beta_n_gamma_p(n, gamma, t_p_x(t, original_age, mortality_table, gender=gender))**(1/gamma) for t in range(len(rs))])
    #print(Betas)

    #Step 3: Multiply it
    out = Betas*const

    #print('out')
    #print(out)

    return out




#Calc quantile calculates the quantile ranges of payouts per person by age
def calc_quantile(longevity_table, payout_per_person, age_0, quantile):
    median = []
    top = []
    bottom = []
    
    for i in range(payout_per_person.shape[1]):
        temp_payout = pd.DataFrame(payout_per_person.loc[:,age_0+i])
        temp_num = pd.DataFrame(longevity_table.loc[:,age_0+i])
        temp_payout_2 = temp_payout.loc[temp_payout[age_0 + i] != -1]
        temp_num_2 = temp_num.loc[temp_payout[age_0 + i] != -1]
        temp = []
        for j in range(temp_num_2.shape[0]):
            temp = temp + int(temp_num_2.iloc[j,0])*[temp_payout_2.iloc[j,0]]
        if len(temp) != 0:
            median.append(np.median(temp))
            top.append(np.quantile(temp, 1-quantile))
            bottom.append(np.quantile(temp, quantile))
    return median, bottom, top

#

#calc_metrics: calculates pre-determined set of metrics for payout function
def calc_metrics_cvar_tontine(longevity_table, payout_matrix, name, ages=[70,79,90], quantile=.05):
    out = pd.DataFrame(np.zeros((int(2*len(ages))+1, 1)), columns=[name])
    out.index = ['Payout First Year of Retirement'] + ['Median Payout at Age ' + str(age) for age in ages] + ['CVar of Payment at Age ' + str(age) for age in ages]
    age_0 = payout_matrix.columns[0]

    out.loc['Payout First Year of Retirement', name] = payout_matrix.iloc[0,0]
    #Calc Meian Payout at each age
    for age in ages:
        temp_payout = pd.DataFrame(payout_matrix.loc[:,age])
        temp_num = pd.DataFrame(longevity_table.loc[:,age])
        temp_payout_2 = temp_payout.loc[temp_payout[age] != -1]
        temp_num_2 = temp_num.loc[temp_payout[age] != -1]
        temp = []
        for j in range(temp_num_2.shape[0]):
            temp = temp + int(temp_num_2.iloc[j,0])*[temp_payout_2.iloc[j,0]]
        if len(temp) != 0:
            out.loc['Median Payout at Age ' + str(age)] = np.median(temp)
            temp = np.array(temp)
            cutoff = np.quantile(temp, quantile)
            out.loc['CVar of Payment at Age ' + str(age)] = np.mean(temp[temp < cutoff])

    return out









