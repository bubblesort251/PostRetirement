import numpy as np
import pandas as pd
import longevity_lib as lon
import scipy
import math
import time
import matplotlib.pyplot as plt


def create_options_dict(target_portfolio, name_of_strat):
    d = dict()
    d['Target Portfolio'] = target_portfolio
    d['Name of Run'] = name_of_strat
    d['Strategy Specification'] = dict()
    return d

#options_dict should also have a 'Strategy Type' element, list with three things, name of pre-annuity strategy, and annuity flag 'annuity' or 'no annuity', 3rd element is option strategy

#defines Seld Directed Strategy
class TONTINECLASS:
    def __init__(self, longevity_table, dt_table, scenarios, inflation_matrix, w_0, age_0, twr_levels=[.02, .025, .03, .035, .04, .045, .049], deaths=[80,90], gamma=30.0):
        self.longevity_table = longevity_table
        self.scenarios = scenarios
        self.dt_table = dt_table
        self.inflation_matrix = inflation_matrix
        self.w_0 = w_0
        self.age_0 = age_0
        self.n_in_tontine = longevity_table.iloc[0,0]
        self.twr_levels = twr_levels
        self.death_ages = deaths
        self.gamma = gamma
        self.strat_performance = dict()


    def extract_age_of_death(self):
        age_of_death = []
        for i in range(self.longevity_table.shape[0]):
            for age in range(self.age_0, self.longevity_table.columns[-1]+1):
                if(self.longevity_table.loc[i, age] == -1):
                    age_of_death.append(age-1)
                    break
        age_of_death = np.array(age_of_death)
        return age_of_death

    #Helper Functions for gen_payment_output
    def initialize_wealth(self, options_dict):
        if(options_dict['Strategy Type'][1] =='no annuity'):
            initial_wealth_pool = self.w_0*self.n_in_tontine
            amount_placed_in_annuity = -1
            return initial_wealth_pool, amount_placed_in_annuity

        if(options_dict['Strategy Type'][1] == 'annuity'):
            initial_wealth_poll = self.w_0*(1-options_dict['Strategy Specification']['amount_placed_in_annuity'])*self.n_in_tontine
            amount_placed_in_annuity = self.w_0*options_dict['Strategy Specification']['amount_placed_in_annuity']
            return initial_wealth_pool, amount_placed_in_annuity


        #update wealth given investment that year
    def update_wealth_with_investment(self, scen, age, wealth, options_dict):
        if(options_dict['Strategy Type'][2] == None):
            year = age - self.age_0
            wealth_vals = wealth*options_dict['Target Portfolio'][year]
            returns = np.cumprod(np.array(self.scenarios[scen][year*12:(year+1)*12]) + 1, axis=0)
            cum_returns = returns[11,:]
            #now add it back together
            wealth = np.sum(wealth_vals[0:7]*cum_returns[0:7]) + wealth_vals[7]*(cum_returns[7]-1) + wealth_vals[8]*cum_returns[8]

        if(options_dict['Strategy Type'][2] == 'collar'):
            #if you haven't hit the terminal condition, now increase wealth by amount
            premium_diff = options_dict['Strategy Specification']['premium_diff']
            ratios_bottom = options_dict['Strategy Specification']['ratios_bottom']
            ratios_top = options_dict['Strategy Specification']['ratios_top']
            target_portfolio = options_dict['Target Portfolio'] 

            year = age - self.age_0
            wealth_vals = wealth*target_portfolio[year]
            returns = np.cumprod(np.array(self.scenarios[scen][year*12:(year+1)*12]) + 1, axis=0)
            cum_returns = returns[11,:]

            #now add it back together
            wealth = 0
            for i in range(3):
                wealth = wealth + wealth_vals[i]*max(ratios_bottom[i] + premium_diff[i], min(cum_returns[i] + premium_diff[i], ratios_top[i] + premium_diff[i]))

            wealth = wealth + np.sum(wealth_vals[3:7]*cum_returns[3:7]) + wealth_vals[7]*(cum_returns[7]-1) + wealth_vals[8]*cum_returns[8]

        if(options_dict['Strategy Type'][2] == 'write_calls'):
            premium_diff = options_dict['Strategy Specification']['premium_diff']
            ratios_top = options_dcit['Strategy Specification']['ratios_top']


            wealth = 0
            for i in range(3):
                wealth = wealth + wealth_vals[i]*min(cum_returns[i] + premiums[i], ratios_top[i] + premiums[i])

            wealth = wealth + np.sum(wealth_vals[3:7]*cum_returns[3:7]) + wealth_vals[7]*(cum_returns[7]-1) + wealth_vals[8]*cum_returns[8]


        if(options_dict['Strategy Type'][2] == 'buy_puts'):
            premium_diff = options_dict['Strategy Specification']['premium_diff']
            ratios_top = options_dcit['Strategy Specification']['ratios_bottom']

            wealth = 0
            for i in range(3):
                wealth = wealth + wealth_vals[i]*max(cum_returns[i] + premium_diff[i], ratios_bottom[i] + premium_diff[i])

            wealth = wealth + np.sum(wealth_vals[3:7]*cum_returns[3:7]) + wealth_vals[7]*(cum_returns[7]-1) + wealth_vals[8]*cum_returns[8]

        return wealth


    def payout_calc(self, scen, wealth_pool, age, payout_matrix, options_dict):
        #This funciton takes all versions of the pre-annuity payout structure
        #TWR Strategy
        if(options_dict['Strategy Type'][0] == 'twr'):
            if(age == self.age_0):
                desired_drawdown = self.w_0*options_dict['Strategy Specification']['twr']*self.longevity_table.loc[scen, age]
            else:
                desired_drawdown = self.w_0*options_dict['Strategy Specification']['twr']*self.longevity_table.loc[scen, age]
                desired_drawdown = desired_drawdown*np.cumprod(self.inflation_matrix[scen, 0:age-self.age_0]+1)[-1]

            drawdown = min(desired_drawdown, wealth_pool)
            payout_matrix.loc[scen, age] = drawdown/self.longevity_table.loc[scen, age]
            wealth_pool = wealth_pool - drawdown

            return payout_matrix, wealth_pool

        if(options_dict['Strategy Type'][0] == 'crra'):
            drawdown = max(0,wealth_pool)*self.dt_table.loc[scen, age]

            payout_matrix.loc[scen, age] = drawdown/self.longevity_table.loc[scen, age]
            wealth_pool = wealth_pool - drawdown

        if(options_dict['Strategy Type'][0] == 'capped crra'):
            if(age == self.age_0):
                desired_drawdown = self.w_0*options_dict['Strategy Specification']['twr']*self.longevity_table.loc[scen, age]
            else:
                desired_drawdown = self.w_0*options_dict['Strategy Specification']['twr']*self.longevity_table.loc[scen, age]
                desired_drawdown = desired_drawdown*np.cumprod(self.inflation_matrix[scen, 0:age-self.age_0]+1)[-1]

            crra_drawdown = max(0,wealth_pool)*self.dt_table.loc[scen, age]

            drawdown = min(desired_drawdown, crra_drawdown)
            payout_matrix.loc[scen, age] = drawdown/self.longevity_table.loc[scen, age]

            wealth_pool = wealth_pool - drawdown

        return payout_matrix, wealth_pool



    def time_step_payout_over_time(self, scen, wealth_pool, age, payout_matrix, ppw_at_start_of_annuity, options_dict):
        #Time step the amount payed over time
        #if it is an annuity type, check if you've hit the annuity
        if(options_dict['Strategy Type'][1] == 'annuity'):
            #If you're at the start of the annuity, record the wealth
            if(age == options_dict['Strategy Specification']['start_of_annuity']):
                if(age > self.age_0):
                    ppw_at_start_of_annuity[scen, 0] = wealth_pool/np.cumprod(self.inflation_matrix[scen, 0:age-self.age_0]+1)[-1]
                else:
                     ppw_at_start_of_annuity[scen, 0] = wealth_pool

            #Check if you've made the annuity, look at the payout
            if(age >= options_dict['Strategy Specification']['start_of_annuity']):
                #If current age is after annuity starts
                amount_placed_in_annuity = options_dict['Strategy Specification']['amount_placed_in_annuity']
                annuity_payout = options_dict['Strategy Specification']['annuity_payout']
                if(age > self.age_0):
                    payout_matrix, wealth_pool = self.payout_calc(scen, wealth_pool, age, payout_matrix, options_dict)
                    payout_matrix.loc[scen, age] = payout_matrix.loc[scen, age] + self.w_0*amount_placed_in_annuity*annuity_payout*np.cumprod(self.inflation_matrix[scen, 0:age-self.age_0]+1)[-1]

                else:
                    payout_matrix, wealth_pool = self.payout_calc(scen, wealth_pool, age, payout_matrix, options_dict)
                    payout_matrix.loc[scen, age] = payout_matrix.loc[scen, age] + self.w_0*amount_placed_in_annuity*annuity_payout

                #zero at end says ruin_flag = 0
                return payout_matrix, wealth_pool,  ppw_at_start_of_annuity
            else: #you're before the annuity, determine the payout structure based on the strategy
                payout_matrix, wealt_pool= self.payout_calc(scen, wealth_pool, age, payout_matrix, options_dict)
                return payout_matrix, wealth, ppw_at_start_of_annuity

        else: #you aren't running an annuity
            payout_matrix, wealth_pool = self.payout_calc(scen, wealth_pool, age, payout_matrix, options_dict)
            return payout_matrix, wealth_pool, ppw_at_start_of_annuity


    #generates payment output, returns payment information, wealth_at_death and wealth_at_start_of_annuity
    def gen_payment_output(self, options_dict):
        #Begin Funciton
        m = self.longevity_table.columns[-1]
        n_scen = self.longevity_table.shape[0]

        cols = [self.age_0 + i for i in range(m-self.age_0+1)]
        payout_matrix = pd.DataFrame(-np.ones((n_scen, m-self.age_0+1)), columns=cols)

        per_dollar_invested_wealth_at_death = -1.*np.zeros((self.longevity_table.shape[0],1))

        #Find age of death
        age_of_death = self.extract_age_of_death()

        ppw_at_start_of_annuity = -np.ones((n_scen, 1))

        #Now loop over all the longevities and figure out the payout to the person
        for scen in range(n_scen):
            #we are in a specific scen

            #Step 1: put some amount into the annuity, if applicable, if not, just ignore
            initial_wealth_pool, amount_placed_in_annuity = self.initialize_wealth(options_dict)
            wealth_pool = initial_wealth_pool

            for age in range(self.age_0, age_of_death[scen]+1):
                #increment values
                payout_matrix, wealth_pool, ppw_at_start_of_annuity = self.time_step_payout_over_time(scen, wealth_pool, age, payout_matrix, ppw_at_start_of_annuity, options_dict)

                #If you haven't hit ruin, update wealth
                wealth_pool = self.update_wealth_with_investment(scen, age, wealth_pool, options_dict)

            cum_inflation = np.cumprod(self.inflation_matrix + 1, axis=1)
            per_dollar_invested_wealth_at_death[scen] = 1*wealth_pool/(self.w_0*self.n_in_tontine)/cum_inflation[scen, age-self.age_0]

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
        return payout_matrix, ppw_at_start_of_annuity, per_dollar_invested_wealth_at_death



       #Claculate the table
    def calc_table(self, payouts_per_person, per_dollar_invested_wealth_at_death,  options_dict):
        twrs = self.twr_levels

        name = options_dict['Name of Run']
        out = pd.DataFrame(np.zeros((2*len(twrs)+8, 1)), columns=[name])
        inds = ['Average Utility'] + ['Certainty Equivalent'] + ['Per Dollar Wealth Remaining At End of Tontine']
        inds = inds + ['Median Wealth at Start of Annuity'] + ['Median Wealth at Death Cond Die Before Annuity']
        inds = inds + ['Prob Ruin'] + ['Prob Ruin Cond Live to 80'] + ['Prob Ruin Cond Live to 90']
        inds = inds + ['Prob Fails TWR ' + str(twr) for twr in twrs] + ['Prob Fails TWR ' + str(twr) + ' Cond Live to Annuity' for twr in twrs]
        out.index = inds
    
        total_people = self.longevity_table.iloc[0,0]*self.longevity_table.shape[0]
    
        twr_vals = [self.w_0*twr for twr in twrs]
        start_retirement = self.longevity_table.columns[0]

        out.loc['Average Utility', name] = self.crra_utility(payouts_per_person)
        out.loc['Certainty Equivalent'] = payout_from_utility(out.loc['Average Utility', name], self.gamma)
        out.loc['Per Dollar Wealth Remaining At End of Tontine', name] = np.mean(per_dollar_invested_wealth_at_death)
    
        ruin_count = 0
        ruin_count_deaths = np.zeros((len(self.death_ages),1))
        fail_twr_counts = np.zeros((len(twrs),1))
        for scen in range(self.longevity_table.shape[0]):
            ruin_tag = 0
            ruin_tag_deaths = np.zeros((len(self.death_ages),1))
            twr_tags = np.zeros((len(twrs),1))
            for year in range(start_retirement, self.longevity_table.columns[-1]+1):
                if(self.longevity_table.loc[scen, year] == -1):
                    break
                #check to see if you need to update the number of people who hit ruin
                if((ruin_tag == 0) & (payouts_per_person.loc[scen, year] == 0)):
                    #print('Tagged')
                    #print('scen = ' + str(scen))
                    #print('year = ' + str(year))
                    ruin_tag = 1
                    ruin_count = ruin_count + self.longevity_table.loc[scen, year]
                    #print('Scen = ' + str(scen))
                    #print('Year = ' + str(year))
                    #print('longevity_table ' + str(self.longevity_table.loc[scen, year]))
                #Check to see if you need to update the number of people who hit ruin above a specific age
                for i in range(len(self.death_ages)):
                    if((ruin_tag_deaths[i] == 0) & (year >= self.death_ages[i]) & (payouts_per_person.loc[scen, year]==0)):
                        ruin_tag_deaths[i] = 1
                        ruin_count_deaths[i] = ruin_count_deaths[i] + self.longevity_table.loc[scen, year]
                    
                #Check to see if you need to update the number of people who failed the swr rate
                for i in range(len(twr_tags)):
                    if((twr_tags[i] == 0) & (payouts_per_person.loc[scen, year] < twr_vals[i])):
                        twr_tags[i] = 1
                        fail_twr_counts[i] = fail_twr_counts[i] + self.longevity_table.loc[scen, year]
                    
        #Now do the divising
        out.loc['Prob Ruin', name] = ruin_count/total_people
        out.loc['Prob Ruin Cond Live to ' + str(self.death_ages[0]), name] = ruin_count_deaths[0]/self.longevity_table[self.death_ages[0]].sum()
        out.loc['Prob Ruin Cond Live to ' + str(self.death_ages[1]), name] = ruin_count_deaths[1]/self.longevity_table[self.death_ages[1]].sum()
    
        for i in range(len(twrs)):
            out.loc['Prob Fails TWR ' + str(twrs[i])] = fail_twr_counts[i] / total_people

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
        payout_matrix, ppw_at_start_of_annuity, per_dollar_invested_wealth_at_death = self.gen_payment_output(options_dict)
        self.strat_performance[options_dict['Name of Run']] = dict()
        self.strat_performance[options_dict['Name of Run']]['Payout Matrix'] = payout_matrix
        self.strat_performance[options_dict['Name of Run']]['Wealth At Start Of Annuity'] = ppw_at_start_of_annuity
        self.strat_performance[options_dict['Name of Run']]['Wealth Remaining At Death (per dollar invested)'] = per_dollar_invested_wealth_at_death
        self.strat_performance[options_dict['Name of Run']]['Metrics'] = self.calc_table(payout_matrix, per_dollar_invested_wealth_at_death, options_dict)

    #Stacks tables
    def stack_tables(self, list_of_strat_names, printBool = False):
        #list_of_strat_names: list object, string object
        out = self.strat_performance[list_of_strat_names[0]]['Metrics']
        for i in range(1, len(list_of_strat_names)):
            out = out.merge(self.strat_performance[list_of_strat_names[i]]['Metrics'], left_index=True, right_index=True)

        self.performance_table = out
        if(printBool):
            return out


def crra_function(x, gamma):
    if(x <= 0):
        return -np.Inf
    return ((x)**(1.-gamma)-1.)/(1.-gamma)

def payout_from_utility(util, gamma):
    return ((1.-gamma)*util + 1.)**(1/(1-gamma))



def calc_percentile(longevity_table, payout_per_person, age_0, quantile):
    out = pd.DataFrame(np.zeros((longevity_table.shape[1], 4)), columns=['Ages', 'Bottom ' + str(quantile), 'Median', 'Top ' + str(quantile)])
    
    for i in range(longevity_table.shape[1]):
        age = age_0 + i
        n_alive = longevity_table[age].sum()
        inds = payout_per_person.sort_values(age).index
        sorted_longevity = longevity_table.loc[inds, age]
        sorted_longevity = sorted_longevity.cumsum()/n_alive
        
        bottom_ind = inds[len(sorted_longevity[sorted_longevity < quantile])]
        top_ind = inds[len(sorted_longevity[sorted_longevity < 1-quantile])]
        median_ind = inds[len(sorted_longevity[sorted_longevity < .5])]

        out.loc[i, 'Ages'] = age
        out.loc[i, 'Bottom ' + str(quantile)] = payout_per_person.loc[bottom_ind, age]
        out.loc[i, 'Median'] = payout_per_person.loc[median_ind, age]
        out.loc[i, 'Top ' + str(quantile)] = payout_per_person.loc[top_ind, age]
        
    return out





