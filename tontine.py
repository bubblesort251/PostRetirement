import numpy as np
import pandas as pd
import longevity_lib as lon

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
            income = wealth*(ir[i,t-age_0]) # grow by 1−yr treasury bond yield curve rate wealth = wealth+income
            if total_dead < n-1:
                dead = np.count_nonzero(members == t) # find number of members dead total dead += dead
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