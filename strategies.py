import numpy as np
import pandas as pd


# finds the probability of ruin across multiple longevity ages and the magnitude of failure at the 5th percentile across multiple longevity ages
def p_ruin(age_0, wealth, longevity, percentile):
    pos = 0 # pass away with positive wealth
    neg = 0 # pass away with negative wealth
    end = []
    for i in range(len(longevity)):
        amt = wealth[i][(int(longevity[i]-age_0)*12)+11]
        end.append(amt)
        
        if amt > 0:
            pos += 1
        else:
            neg += 1
        
        p_ruin = neg/(pos+neg) # probability of ruin
        mag = np.percentile(end, percentile) # magnitude of failure at 5th percentile
        mean_w = np.mean(end) # average wealth at time of death
    
    return(p_ruin, mag, mean_w)


# clean up the median wealth path
def clean_wealth(wealth):
    med_wealth = []

    for i in range(len(wealth[0])):
        boolArr = wealth[:,i] != None
        temp_wealth = wealth[boolArr,i]

        if temp_wealth.size != 0:
            med_wealth.append(np.median(temp_wealth))

    return med_wealth

# find profile of strategy . takes matrix of wealth across all scenarios , the age of death T, and the percentile for magnitude of ruin,
    #returns the probability of ruin, the mean wealth and volatility of wealth at time of death

def profile(age_0, wealth, death, percentile):
    pos = 0 # pass away with positive wealth
    neg = 0 # pass away with negative wealth
    end = []

    for i in range(wealth.shape[0]):
        amt = wealth[i][(int(death-age_0)*12)+11]
        end.append(amt)
        if amt > 0:
            pos += 1
        
        else:
            neg += 1

    p_ruin = neg/(pos+neg) # probability of ruin

    mag = np.percentile(end, percentile) # magnitude of failure at 5th percentile
    mean_w = np.mean(end) # average wealth at time of death
    std = np.std(end) # standard deviation of wealth at time of death across all scenarios 
    return(p_ruin, mag, mean_w, std)



# creates array of wealth level at time i
# takes matrix of returns where each asset returns are on a column, wealth = list of cash inflows each year, target = the optimal asset allocation proportion of starting wealth,
    # this can be either a list of target allocations each year or a (j , 1) matrix of allocations for each asset in J,
    # n asset = the number of assets , dynamic = False or True depending on if target is a list of dynamic target allocations over the years.
    # Returns the wealth over time as a list and the return for each month

def wealth_s(age_0, returns, target, w_0, swr, inflation, n_asset, longevity, dynamic = False):
    w=w_0
    y = 0 # counter for years of retirement
    wplt = [w_0] # array of total portfolio wealth for each month
    avg_returns = [] # portfolio return for each month
    drawdown = [swr*w_0]
    
    for i in range(int(longevity - age_0)):
        drawdown.append(drawdown[i]*(1+inflation[i]))
    
    # set portfolio allocation
    if dynamic:
        alloc = target[0]
    else:
        alloc = target

    # iterate through the time periods to rebalance each month
    for i in range(len(returns)):
        # calculate any yearly calculations (adding new cash−inflows/outflows)
         # when it’s january, do...
        if i%12 == 0:
            # if still alive
            if(y+age_0) <= longevity:
                w -= drawdown[y] # subtract drawdown cash flow
                
                # change target allocation for the next year
                if dynamic:
                    alloc = target[y]
            # update year counter
            y += 1

        w_t = w # wealth at start of month

        # if still alive
        if(y-1+age_0)<=longevity:

            #if still have money in account
            if w_t>0:
                # rebalancing
                amount = w*alloc
                # grow returns from each asset for the month
                for j in range(n_asset):
                    amount[j] = amount[j]*(1+returns[i][j])

                w = np.sum(amount) # wealth for start of next month
                wplt.append(w)
                avg_returns.append((w - w_t)/w_t) # return for this month

            # if no money left in account
            else:
                wplt.append(w_t)
                avg_returns.append(None)

        # if died , set wealth = None
        else:
            wplt.append(None)
            avg_returns.append(None)

    return(wplt, avg_returns, drawdown)



# using generated scenarios , find wealth over all simulations. Takes matrix of scenarios shape ( nscenarios , (periods , assets)), target is the allocation to each asset for fixed mix,
    # w in is the list of cash flows in each year, returns the wealth over all simulations as a matrix with each wealth for a given simulation on a row,
    # also returns a list of the average returns for each month over the scenarios


def wealth_path(age_0, scenarios, target, w_0, swr, inf_mat, n_asset, longevity, dynamic = False):
    wealths = []
    returns = [] # matrix of returns for each year under each scenario
    avg_returns = [] # list of the monthly average return across all scenarios
    drawdowns = [] # matrix of all drawdowns

    if isinstance(longevity, np.ndarray):
        wealth_return = list(map(lambda i: wealth_s(age_0, scenarios[i], target, w_0, swr, inf_mat[i], n_asset , longevity[i], dynamic), range(len(scenarios))))
    else:
        wealth_return = list(map(lambda i: wealth_s(age_0, scenarios[i], target, w_0, swr, inf_mat[i], n_asset, longevity, dynamic) , range(len(scenarios))))


    for i in range(len(wealth_return)):
        wealths.append(wealth_return[i][0])
        returns.append(wealth_return[i][1])
        drawdowns.append(wealth_return[i][2])

    wealths = np.array(wealths)
    returns = np.array(returns)
    drawdowns = np.array(drawdowns)

    for i in range(len(returns[0])):
        boolArr = returns[:,i] != None
        temp_returns = returns[boolArr,i]
        if temp_returns.size != 0:
            avg_returns.append(np.mean(temp_returns))

    return (wealths, avg_returns, drawdowns)


# same as wealth_s() but accounts for how the portfolio tracks with the goal portfolio
def wealth_s_dzgp(age_0, returns, target, w_0, d, inflation, n_asset, longevity):
    w=w_0
    y = 0 #counter for years
    wplt = [w_0] # array of total portfolio wealth for each month
    avg_returns = [] # portfolio return for each month


    # create drawdowns
    drawdown = [d*w_0]
    for i in range(int(longevity - age_0)):
        drawdown.append(drawdown[i]*(1+inflation[i]))
    
    # create goal portfolio
    goal=[w_0]
    for i in range(1, int(longevity - age_0)+1):
        temp_w = (goal[i-1]-drawdown[i-1])*1.065 #withdraw and then grow remaining by 6.5% 
        goal.append(temp w)

    # set portfolio allocation
    alloc = target[0].copy()

    # iterate through the time periods to rebalance each month
    for i in range(len(returns)):
        # calculate any yearly calculations (adding new cash−inflows/outflows)
        # when it’s january, do...
        if i%12 == 0:
            # if still alive
            if(y+age_0) <= longevity:
                w -= drawdown[y] # subtract drawdown
                # change target allocation for the next year
                alloc = target[y].copy()
                # if wealth is less than 90% of goal wealth, increase allocation to equity by 20% and decrease allocation to bonds by 20%
                if wplt[i] < goal[y]*0.9:
                    alloc[0] = np.minimum(alloc[0]+0.2 , 1)
                    alloc[1] = 1-alloc[0]
                # if wealth is greater than 110% of goal wealth, decrease allocation to equity by 20% and increase allocation to bonds by 20%
                elif wplt[i] >= goal[y]*1.1:
                    alloc[0] = np.maximum(alloc[0] - 0.2, 0)
                    alloc[1] = 1-alloc[0]
            y +=1
            







