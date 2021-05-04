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
                    if(j != 7):
                        amount[j] = amount[j]*(1+returns[i][j])
                    else:
                        amount[j] = amount[j]*(returns[i][j])

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
def wealth_s_dzgp(age_0, returns, target, w_0, d, inflation, n_asset, longevity, goal_return):
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
        temp_w = (goal[i-1]-drawdown[i-1])*(1+goal_return) #withdraw and then grow remaining by the goal 
        goal.append(temp_w)

    # set portfolio allocation
    alloc = target[0].copy()
    #change target allocation to whatever

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
                    if(y+age_0 < 100):
                        x = alloc[0] + alloc[1] + alloc[2] + alloc[6]
                        if(x + .2 >= 1):
                            equity_prop = 1
                        else:
                            equity_prop = x + .2
                            mult_e = equity_prop/x
                        bond_prop = 1- equity_prop
                        mult_b = bond_prop/np.sum(alloc[3] + alloc[5])

                        alloc[0] = alloc[0]*mult_e
                        alloc[1] = alloc[1]*mult_e
                        alloc[2] = alloc[2]*mult_e
                        alloc[6] = alloc[6]*mult_e

                        alloc[3] = alloc[3]*mult_b
                        alloc[5] = alloc[5]*mult_b
                # if wealth is greater than 110% of goal wealth, decrease allocation to equity by 20% and increase allocation to bonds by 20%
                elif wplt[i] >= goal[y]*1.1:
                    if(y+age_0 < 100):
                        x = alloc[0] + alloc[1] + alloc[2] + alloc[6]
                        if(x + .2 >= 1):
                            equity_prop = 1
                        else:
                            equity_prop = x - .2
                            mult_e = equity_prop/x
                        bond_prop = 1- equity_prop
                        mult_b = bond_prop/np.sum(alloc[3] + alloc[5])

                        alloc[0] = alloc[0]*mult_e
                        alloc[1] = alloc[1]*mult_e
                        alloc[2] = alloc[2]*mult_e
                        alloc[6] = alloc[6]*mult_e

                        alloc[3] = alloc[3]*mult_b
                        alloc[5] = alloc[5]*mult_b
            y +=1
            
        w_t = w #wealth at start of month 
        #print(’wealth at start of month: ’, w t)

        # if still alive
        if (y-1+age_0)<=longevity:
            # if still have money in account
            if w_t>0:
                # rebalancing
                amount = w*alloc
                # grow returns from each asset for the month
                for j in range(n_asset):
                    if(j != 7):
                        amount[j] = amount[j]*(1+returns[i][j])
                    else:
                        amount[j] = amount[j]*(returns[i][j])

                w = np.sum(amount) # wealth for start of next month
                wplt.append(w)
                avg_returns.append((w - w_t) / w_t) # return for this month
            
            # if no money left in account
            else:
                wplt.append(w_t)
                avg_returns.append(None)
        # if dead , set wealth = None
        else:
            wplt.append(None)
            avg_returns.append(None)

    return(wplt,avg_returns,drawdown)

# function for modeling the wealth of the DZGP strategy
def wealth_path_dzgp(age_0, scenarios, target, w_0, d, inf_mat , n_asset , longevity, goal_return):
    wealths = []
    returns = [] # matrix of returns for each year under each scenario 
    drawdowns = []
    avg_returns = [] # list of the monthly average return across all scenarios


    if isinstance(longevity , np.ndarray):
        wealth_return = list(map(lambda i: wealth_s_dzgp(age_0, scenarios[i], target, w_0, d, inf_mat[i], n_asset, longevity[i], goal_return), range(len(scenarios))))

    else:
        wealth_return = list(map(lambda i: wealth_s_dzgp(age_0, scenarios[i], target, w_0, d, inf_mat[i], n_asset , longevity, goal_return), range(len(scenarios))))

    for i in range(len(wealth_return)):
        wealths.append(wealth_return[i][0])
        returns.append(wealth_return[i][1])
        drawdowns.append(wealth_return[i][2])

    # calculate average return over scenarios only for returns that are not None
    wealths = np.array(wealths)
    returns = np.array(returns)
    drawdowns = np.array(drawdowns)
    for i in range(len(returns[0])):
        boolArr = returns[:,i] != None
        temp_returns = returns[boolArr,i]
        if temp_returns.size != 0:
            avg_returns.append(np.mean(temp_returns))
    
    return (wealths , avg_returns , drawdowns)



# wealth s() for dynamic drawdown strategy
def wealth_s_dd(age_0, returns, target, w_0, d, inflation, n_asset, longevity, goal_return, dynamic = True):
    w=w_0
    y = 0 #counter for years
    wplt = [w_0] # array of total portfolio wealth for each month
    avg_returns = [] # portfolio return for each month
    w_out = [] # array of all the drawdowns
    
    # create drawdowns
    swr_drawdown = [d*w_0]
    for i in range(int(longevity - age_0)):
        swr_drawdown.append(swr_drawdown[i]*(1+inflation[i]))
    
    # create goal portfolio
    goal=[w_0]
    for i in range(1, int(longevity - age_0)+1):
        temp_w = (goal[i-1]-swr_drawdown[i-1])*(1+goal_return) #withdraw and then grow remaining by goal_return
        goal.append(temp_w)
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
            if (y+age_0) <= longevity:
                drawdown = swr_drawdown[y]
                # if wealth is less than 90% of goal wealth, decrease drawdown by 10%
                if wplt[i] < goal[y]*0.9:
                    drawdown = swr_drawdown[y]*0.9

                # if wealth is greater than 110% of goal wealth , increase drawdown by 20%
                elif wplt[i] >= goal[y]*1.1:
                    drawdown = swr_drawdown[y]*1.25

                w_out.append(drawdown)
                w -= drawdown # subtract amount being drawndown

                # change target allocation for the next year
                if dynamic:
                    alloc = target[y]
            
            # update year counter
            y += 1

        w_t = w #wealth at start of month

        # if still alive
        if(y-1+age_0)<=longevity:

            #if still have money in account
            if w_t>0:
                # rebalancing
                amount = w*alloc
                # grow returns from each asset for the month
                for j in range(n_asset):
                    if(j != 7):
                        amount[j] = amount[j]*(1+returns[i][j])
                    else:
                        amount[j] = amount[j]*(returns[i][j])

                w = np.sum(amount) # wealth for start of next month
                wplt.append(w)
                avg_returns.append((w-w_t) / w_t) # return for this month

            # if no money left in account
            else:
                wplt.append(w_t)
                avg_returns.append(None)

        # if died , set wealth = None
        else:
            wplt.append(None)
            avg_returns.append(None)

    return(wplt, avg_returns, w_out)



# function for modeling the wealth of the DZGP strategy
def wealth_path_dd(age_0, scenarios, target, w_0, d, inf_mat, n_asset, longevity, goal_return):
    wealths = []
    returns = [] # matrix of returns for each year under each scenario 
    drawdowns = []
    avg_returns = [] # list of the monthly average return across all scenarios

    if isinstance(longevity , np.ndarray):
        wealth_return = list(map(lambda i: wealth_s_dd(age_0, scenarios[i], target, w_0, d, inf_mat[i], n_asset, longevity[i], goal_return), range(len(scenarios))))

    else:
        wealth_return = list(map(lambda i: wealth_s_dd(age_0, scenarios[i], target, w_0, d, inf_mat[i], n_asset, longevity, goal_return), range(len(scenarios))))

    for i in range(len(wealth_return)):
        wealths.append(wealth_return[i][0])
        returns.append(wealth_return[i][1])
        drawdowns.append(wealth_return[i][2])

    # calculate average return over scenarios only for returns that are not None
    wealths = np.array(wealths)
    returns = np.array(returns)
    drawdowns = np.array(drawdowns)
    for i in range(len(returns[0])):
        boolArr = returns[:,i] != None
        temp_returns = returns[boolArr,i]
        if temp_returns.size != 0:
            avg_returns.append(np.mean(temp_returns))
    
    return (wealths , avg_returns , drawdowns)





















