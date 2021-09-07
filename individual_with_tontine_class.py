#This library runs the self directed strategies
import numpy as np
import pandas as pd
import longevity_lib as lon
import scipy
import math
import time
import matplotlib.pyplot as plt

#creates options dict for strategy
def create_options_dict(target_portfolio, name_of_strat):
    d = dict()
    d['Target Portfolio'] = target_portfolio
    d['Name of Run'] = name_of_strat
    d['Strategy Specification'] = dict()
    return d

#options_dict should also have a 'Strategy Type' element, list with three things, name of pre-annuity strategy, and annuity flag 'annuity' or 'no annuity', 3rd element is option strategy

#defines Seld Directed Strategy
class individualWithTontine:
    def __init__(self, longevity_table, tontine_table, scenarios, inflation_matrix, w_0, age_0, female_ind, gamma=30):
        self.longevity_table = longevity_table
        self.tontine_table = tontine_table
        self.scenarios = scenarios
        self.inflation_matrix = inflation_matrix
        self.w_0 = w_0
        self.age_0 = age_0
        self.female_ind = female_ind
        self.gamma = gamma
        self.strat_performance = dict()

    #Helper Functions for gen_payment_output
    def initialize_wealth(self, options_dict):
        if(options_dict['Strategy Type'][1] =='no annuity'):
            initial_wealth = self.w_0
            amount_placed_in_annuity = -1
            return initial_wealth, amount_placed_in_annuity

        if(options_dict['Strategy Type'][1] == 'annuity'):
            initial_wealth = self.w_0*(1-options_dict['Strategy Specification']['amount_placed_in_annuity'])
            amount_placed_in_annuity = self.w_0*options_dict['Strategy Specification']['amount_placed_in_annuity']
            return initial_wealth, amount_placed_in_annuity

    def extract_age_of_death(self):
        age_of_death = []
        for i in range(self.longevity_table.shape[0]):
            for age in range(self.age_0, self.longevity_table.columns[-1]+1):
                if(self.longevity_table.loc[i, age] == -1):
                    age_of_death.append(age-1)
                    break
        age_of_death = np.array(age_of_death)
        return age_of_death

    def calc_tontine_return(self, wealth_invested, cum_tbill_returns, scen, age, options_dict):
        #If you die during this period, you get nothing from the tontine
        if(self.longevity_table.loc[scen, age+1] == -1):
            return 0
        #Otherwise, you get the normal return
        else:
            # Degenerate Case when no other people in the tontine, give t-bill return
            if(self.tontine_table.loc[scen, age] == -1):
                return wealth_invested*cum_tbill_returns
            #else, the standard return
            else:
                survivors = max(self.tontine_table.loc[scen, age+1], 0)
                current = max(self.tontine_table.loc[scen, age], 0)
                return wealth_invested*cum_tbill_returns*(current+1)/(survivors+1)


    #update wealth given investment that year
    def update_wealth_with_investment(self, scen, age, wealth, options_dict):
        if(options_dict['Strategy Type'][2] == None):
            year = age - self.age_0
            wealth_vals = wealth*options_dict['Target Portfolio'][year, :, scen]
            returns = np.cumprod(np.array(self.scenarios[scen][year*12:(year+1)*12]) + 1, axis=0)
            cum_returns = returns[11,:]
            #now add it back together
            wealth = np.sum(wealth_vals[0:7]*cum_returns[0:7]) + wealth_vals[7]*(cum_returns[7]-1) + wealth_vals[8]*cum_returns[8]
            wealth = wealth + self.calc_tontine_return(wealth_vals[9], cum_returns[8], scen, age, options_dict)

        if(options_dict['Strategy Type'][2] == 'collar'):
            #if you haven't hit the terminal condition, now increase wealth by amount
            premium_diff = options_dict['Strategy Specification']['premium_diff']
            ratios_bottom = options_dict['Strategy Specification']['ratios_bottom']
            ratios_top = options_dict['Strategy Specification']['ratios_top']
            target_portfolio = options_dict['Target Portfolio'] 

            year = age - self.age_0
            wealth_vals = wealth*options_dict['Target Portfolio'][year, :, scen]
            returns = np.cumprod(np.array(self.scenarios[scen][year*12:(year+1)*12]) + 1, axis=0)
            cum_returns = returns[11,:]

            #now add it back together
            wealth = 0
            for i in range(3):
                wealth = wealth + wealth_vals[i]*max(ratios_bottom[i] + premium_diff[i], min(cum_returns[i] + premium_diff[i], ratios_top[i] + premium_diff[i]))

            wealth = wealth + np.sum(wealth_vals[3:7]*cum_returns[3:7]) + wealth_vals[7]*(cum_returns[7]-1) + wealth_vals[8]*cum_returns[8]
            wealth = wealth + self.calc_tontine_return(wealth_vals[9], cum_returns[8], scen, age, options_dict)

        if(options_dict['Strategy Type'][2] == 'write_calls'):
            premium_diff = options_dict['Strategy Specification']['premium_diff']
            ratios_bottom = options_dict['Strategy Specification']['ratios_bottom']
            ratios_top = options_dict['Strategy Specification']['ratios_top']
            target_portfolio = options_dict['Target Portfolio'] 

            year = age - self.age_0
            wealth_vals = wealth*options_dict['Target Portfolio'][year, :, scen]
            returns = np.cumprod(np.array(self.scenarios[scen][year*12:(year+1)*12]) + 1, axis=0)
            cum_returns = returns[11,:]


            wealth = 0
            for i in range(3):
                wealth = wealth + wealth_vals[i]*min(cum_returns[i] + premiums[i], ratios_top[i] + premiums[i])

            wealth = wealth + np.sum(wealth_vals[3:7]*cum_returns[3:7]) + wealth_vals[7]*(cum_returns[7]-1) + wealth_vals[8]*cum_returns[8]
            wealth = wealth + self.calc_tontine_return(wealth_vals[9], cum_returns[8], scen, age, options_dict)


        if(options_dict['Strategy Type'][2] == 'buy_puts'):
            premium_diff = options_dict['Strategy Specification']['premium_diff']
            ratios_bottom = options_dict['Strategy Specification']['ratios_bottom']
            ratios_top = options_dict['Strategy Specification']['ratios_top']
            target_portfolio = options_dict['Target Portfolio'] 

            year = age - self.age_0
            wealth_vals = wealth*options_dict['Target Portfolio'][year, :, scen]
            returns = np.cumprod(np.array(self.scenarios[scen][year*12:(year+1)*12]) + 1, axis=0)
            cum_returns = returns[11,:]

            wealth = 0
            for i in range(3):
                wealth = wealth + wealth_vals[i]*max(cum_returns[i] + premium_diff[i], ratios_bottom[i] + premium_diff[i])

            wealth = wealth + np.sum(wealth_vals[3:7]*cum_returns[3:7]) + wealth_vals[7]*(cum_returns[7]-1) + wealth_vals[8]*cum_returns[8]
            wealth = wealth + self.calc_tontine_return(wealth_vals[9], cum_returns[8], scen, age, options_dict)

        return wealth

    def calc_wealth_at_death(self, scen, age, wealth, wealth_at_death, options_dict):
        if(age == self.age_0):
            wealth_at_death[scen, 0] = wealth
        else:
            wealth_at_death[scen, 0] = wealth/np.cumprod(self.inflation_matrix[scen, 0:age-self.age_0]+1)[-1]

        return wealth_at_death

    def time_step_payout_over_time(self, scen, wealth, age, payout_matrix, wealth_at_death, wealth_at_start_of_annuity, is_ruined, options_dict):
        #Time step the amount payed over time
        #if it is an annuity type, check if you've hit the annuity
        if(options_dict['Strategy Type'][1] == 'annuity'):
            #If you're at the start of the annuity, record the wealth
            if(age == options_dict['Strategy Specification']['start_of_annuity']):
                if(age > self.age_0):
                   wealth_at_start_of_annuity[scen, 0] = wealth/np.cumprod(self.inflation_matrix[scen, 0:age-self.age_0]+1)[-1]
                else:
                    wealth_at_start_of_annuity[scen, 0] = wealth

            #Check if you've made the annuity, look at the payout
            if(age >= options_dict['Strategy Specification']['start_of_annuity']):
                #If current age is after annuity starts
                amount_placed_in_annuity = options_dict['Strategy Specification']['amount_placed_in_annuity']
                annuity_payout = options_dict['Strategy Specification']['annuity_payout']
                if(age > self.age_0):
                    payout_matrix.loc[scen, age] = self.w_0*amount_placed_in_annuity*annuity_payout*np.cumprod(self.inflation_matrix[scen, 0:age-self.age_0]+1)[-1]
                else:
                    payout_matrix.loc[scen, age] = self.w_0*amount_placed_in_annuity*annuity_payout

                #zero at end says ruin_flag = 0
                return payout_matrix, wealth, wealth_at_start_of_annuity, is_ruined
            else: #you're before the annuity, determine the payout structure based on the strategy
                payout_matrix, wealth, is_ruined = self.pre_annuity_payout_calc(scen, wealth, age, payout_matrix, wealth_at_death, is_ruined, options_dict)
                return payout_matrix, wealth, wealth_at_start_of_annuity, is_ruined

        else: #you aren't running an annuity
            payout_matrix, wealth, is_ruined = self.pre_annuity_payout_calc(scen, wealth, age, payout_matrix, wealth_at_death, is_ruined, options_dict)
            return payout_matrix, wealth, wealth_at_start_of_annuity, is_ruined


    def pre_annuity_payout_calc(self, scen, wealth, age, payout_matrix, wealth_at_death, is_ruined, options_dict):
        #This funciton takes all versions of the pre-annuity payout structure
        #Check if you're ruined
        if(wealth == 0):    
            is_ruined[scen, 0] = 1

        #SWR Strategy
        if(options_dict['Strategy Type'][0] == 'twr'):
            if(age == self.age_0):
                desired_drawdown = self.w_0*options_dict['Strategy Specification']['twr']
            else:
                desired_drawdown = self.w_0*options_dict['Strategy Specification']['twr']*np.cumprod(self.inflation_matrix[scen, 0:age-self.age_0]+1)[-1]

            payout_matrix.loc[scen, age] = min(desired_drawdown, wealth)
            wealth = wealth - payout_matrix.loc[scen, age]

            return payout_matrix, wealth, is_ruined
        #Harmonic Strategy
        if(options_dict['Strategy Type'][0] == 'harmonic'):
            payout_matrix.loc[scen, age] = wealth/(options_dict['Strategy Specification']['start_of_annuity'] - age)
            wealth = wealth - payout_matrix.loc[scen, age]

            return payout_matrix, wealth, is_ruined

        #TWR + Harmonic
        if(options_dict['Strategy Type'][0] == 'twr_harmonic'):
            if(age == self.age_0):
                desired_drawdown = self.w_0*options_dict['Strategy Specification']['twr']
            else:
                desired_drawdown = self.w_0*options_dict['Strategy Specification']['twr']*np.cumprod(self.inflation_matrix[scen, 0:age-self.age_0]+1)[-1]

            payout_matrix.loc[scen, age] = min(desired_drawdown, wealth/(options_dict['Strategy Specification']['start_of_annuity'] - age))
            wealth = wealth - payout_matrix.loc[scen, age]

            return payout_matrix, wealth, is_ruined

                #SWR + Harmonic
        if(options_dict['Strategy Type'][0] == 'twr_harmonic_without_annuity'):
            if(age == self.age_0):
                desired_drawdown = self.w_0*options_dict['Strategy Specification']['twr']
            else:
                desired_drawdown = self.w_0*options_dict['Strategy Specification']['twr']*np.cumprod(self.inflation_matrix[scen, 0:age-self.age_0]+1)[-1]

            life_expectancy = self.female_ind*options_dict['Strategy Specification']['Life Expectancy Table'].loc[age, 'Life Expectancy Female']
            life_expectancy = life_expectancy + (1-self.female_ind)*options_dict['Strategy Specification']['Life Expectancy Table'].loc[age, 'Life Expectancy Male']

            payout_matrix.loc[scen, age] = min(desired_drawdown, wealth/(life_expectancy))
            wealth = wealth - payout_matrix.loc[scen, age]

            return payout_matrix, wealth, is_ruined



    #generates payment output, returns payment information, wealth_at_death and wealth_at_start_of_annuity
    def gen_payment_output(self, options_dict):
        #Begin Funciton
        m = self.longevity_table.columns[-1]
        n_scen = self.longevity_table.shape[0]
        out = pd.DataFrame(-1*np.ones((n_scen, m - self.age_0+1)), columns=[i for i in range(self.age_0, m+1)])

        #Find age of death
        age_of_death = self.extract_age_of_death()

        cols = [self.age_0 + i for i in range(m-self.age_0)]
        payout_matrix = pd.DataFrame(-np.ones((n_scen, m-self.age_0)), columns=cols)

        wealth_at_death = -np.ones((n_scen, 1))
        wealth_at_start_of_annuity = -np.ones((n_scen, 1))
        is_ruined = np.zeros((n_scen, 1))

        #Now loop over all the longevities and figure out the payout to the person
        for scen in range(n_scen):
            #we are in a specific scen

            #Step 1: put some amount into the annuity, if applicable, if not, just ignore
            initial_wealth, amount_placed_in_annuity = self.initialize_wealth(options_dict)
            wealth = initial_wealth

            for age in range(self.age_0, age_of_death[scen]+1):
                #increment values
                payout_matrix, wealth, wealth_at_start_of_annuity, is_ruined = self.time_step_payout_over_time(scen, wealth, age, payout_matrix, wealth_at_death, wealth_at_start_of_annuity, is_ruined, options_dict)

                #if you're ruined, then update the remaining thingy
                if(is_ruined[scen, 0] == 1):
                   payout_matrix.loc[scen, age:age_of_death[scen]+1] = 0
                   break

                #If you haven't hit ruin, update wealth
                wealth = self.update_wealth_with_investment(scen, age, wealth, options_dict)

            #Update is_ruined calculator
            if(is_ruined[scen,0] == -1):
                is_ruined[scen,0] = 0

            wealth_at_death = self.calc_wealth_at_death(scen, age, wealth, wealth_at_death, options_dict)


        #Now Deflate the payments back to current dollars
        cum_inflation = np.cumprod(self.inflation_matrix + 1, axis=1)

        #Loop over payout_matrix and do the ofsetting inflation thing
        for i in range(1, payout_matrix.shape[1]):
            payout_matrix.iloc[:,i] = payout_matrix.iloc[:,i]/cum_inflation[:,i-1]

        #Now set the payouts per person after death back to negative 1
        for scen in range(n_scen):
            death = age_of_death[scen]
            payout_matrix.loc[scen, death+1:] = -1

        #also do an inflation adjustment to wealth at longevity
        return payout_matrix, wealth_at_start_of_annuity, wealth_at_death, is_ruined

    #Claculate the table
    def calc_table(self, options_dict, swrs=[.020, .025, .030, .035, .040, 0.042, 0.043, 0.044, 0.045, 0.046]):
        name = options_dict['Name of Run']
        out = pd.DataFrame(np.zeros((2*len(swrs)+8, 1)), columns=[name])
        inds = ['Average Utility'] + ['Certainty Equivalent'] + ['Median Wealth at Death']
        inds = inds + ['Median Wealth at Start of Annuity'] + ['Median Wealth at Death Cond Die Before Annuity']
        inds = inds + ['Prob Ruin'] + ['Prob Ruin Cond Live to 80'] + ['Prob Ruin Cond Live to 90']
        inds = inds + ['Prob Fails TWR ' + str(swr) for swr in swrs] + ['Prob Fails TWR ' + str(swr) + ' Cond Live to Annuity' for swr in swrs]
        out.index = inds

        #Define Helpful Terms
        payout_matrix = self.strat_performance[options_dict['Name of Run']]['Payout Matrix']
        wealth_at_start_of_annuity = self.strat_performance[options_dict['Name of Run']]['Wealth At Start Of Annuity']
        wealth_at_death = self.strat_performance[options_dict['Name of Run']]['Wealth At Death']
        is_ruined = self.strat_performance[options_dict['Name of Run']]['Is Ruined']
        n_scen = payout_matrix.shape[0]
        

        age_of_death = np.zeros((n_scen,1))
        age_of_death[:,0] = self.extract_age_of_death()

        #Define First values
        out.loc['Average Utility', name] = self.crra_utility(payout_matrix)
        out.loc['Certainty Equivalent'] = payout_from_utility(out.loc['Average Utility', name], self.gamma)
        out.loc['Median Wealth at Death', name] = np.median(wealth_at_death[wealth_at_death != -1])
        
        if(options_dict['Strategy Type'][1] == 'annuity'):
            start_of_annuity = options_dict['Strategy Specification']['start_of_annuity']
            out.loc['Median Wealth at Start of Annuity', name] = np.median(wealth_at_start_of_annuity[wealth_at_start_of_annuity != -1])
            out.loc['Median Wealth at Death Cond Die Before Annuity', name] = np.median(wealth_at_death[(age_of_death <  start_of_annuity)])
        else:
            out.loc['Median Wealth at Start of Annuity', name] = None
            out.loc['Median Wealth at Death Cond Die Before Annuity', name] = None
        
        #Calculate Prob of Ruin
        out.loc['Prob Ruin', name] = np.count_nonzero(is_ruined)/is_ruined.shape[0]
        #Calculate Prob of Ruin Cond Live to 80
        out.loc['Prob Ruin Cond Live to 80', name] = np.count_nonzero(is_ruined[age_of_death >= 80])/is_ruined[age_of_death >= 80].shape[0]
        #Calculate Prob of Ruin Cond Live to 90
        out.loc['Prob Ruin Cond Live to 90', name] = np.count_nonzero(is_ruined[age_of_death >= 90])/is_ruined[age_of_death >= 90].shape[0]

        #Ok, now I calculate the probability fails SWR
        n_survived_till_annuity = 0.0
        fails_swr_total = np.zeros((1,len(swrs)))
        fails_swr_cond = np.zeros((1,len(swrs)))

        min_payout_by_scen = np.zeros((n_scen, 1))
        #Calculate the minimum for each row
        for scen in range(n_scen):
            row = payout_matrix.loc[scen, :]
            min_payout_by_scen[scen, 0] = np.amin(row[row != -1])

        #Calculate Prob Fails SWR
        for swr in swrs:
            out.loc['Prob Fails TWR ' + str(swr), name] = min_payout_by_scen[min_payout_by_scen < self.w_0*swr].shape[0] / n_scen

        #calculate Prob Filas SWR conditional on living to annuity
        if(options_dict['Strategy Type'][1] == 'annuity'):
            for swr in swrs:
                temp = min_payout_by_scen[age_of_death >= start_of_annuity]
                out.loc['Prob Fails TWR ' + str(swr) + ' Cond Live to Annuity', name] = temp[temp < self.w_0*swr].shape[0] / temp.shape[0]
        else:
            for swr in swrs:
                out.loc['Prob Fails TWR ' + str(swr) + ' Cond Live to Annuity', name] = None

        return out

    def crra_utility(self, payout_table):
        util = payout_table[payout_table != -1].copy()/self.w_0
        util = util.apply(np.vectorize(crra_function), gamma=self.gamma)
        s = self.longevity_table[self.longevity_table != -1].sum().sum()
        prod = (util*self.longevity_table[self.longevity_table != -1]).sum().sum()
        return prod/s

    #Helper Functions for running a strategy
    def run_strat(self, options_dict):
        print('Beginning Run')
        print(options_dict['Name of Run'])
        payout_matrix, wealth_at_start_of_annuity, wealth_at_death, is_ruined = self.gen_payment_output(options_dict)
        self.strat_performance[options_dict['Name of Run']] = dict()
        self.strat_performance[options_dict['Name of Run']]['Payout Matrix'] = payout_matrix
        self.strat_performance[options_dict['Name of Run']]['Wealth At Start Of Annuity'] = wealth_at_start_of_annuity
        self.strat_performance[options_dict['Name of Run']]['Wealth At Death'] = wealth_at_death
        self.strat_performance[options_dict['Name of Run']]['Is Ruined'] = is_ruined
        self.strat_performance[options_dict['Name of Run']]['Metrics'] = self.calc_table(options_dict)

    #Stacks tables
    def stack_tables(self, list_of_strat_names):
        #list_of_strat_names: list object, string object
        out = self.strat_performance[list_of_strat_names[0]]['Metrics']
        for i in range(1, len(list_of_strat_names)):
            out = out.merge(self.strat_performance[list_of_strat_names[i]]['Metrics'], left_index=True, right_index=True)
        return out


def crra_function(x, gamma):
    if(x <= 0):
        return -np.Inf
    return ((x)**(1.-gamma)-1.)/(1.-gamma)

def payout_from_utility(util, gamma):
    return ((1.-gamma)*util + 1.)**(1/(1-gamma))






