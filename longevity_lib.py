# returns the age of death of a given individual
import numpy as np
import pandas as pd

def mortality(age_0, prob):
    alive = 1
    age = age_0
    while alive > 0:
        rand = np.random.uniform()
        if rand<=prob[age]:
            alive = 0
        elif age > 118:
            alive = 0
        else:
            age += 1
    return age

# returns an array of lifespans for n scenarios starting from age_0 given mortality table prob
def longevity_n(nscen, age_0, prob):
    longevity = np.zeros(nscen) # array to generate longevity for each scenario
    for i in range(nscen):
        longevity[i] = mortality(age_0, prob)
    return longevity

# returns an array of lifespans for n scenarios starting from age_0 given mortality table prob, conditions on all ages above lb
def conditional_longevity(nscen, age_0, prob, lb):
    longevity = np.zeros(nscen)
    curr = 0
    while(curr < nscen):
        age = mortality(age_0, prob)
        if(age > lb):
            longevity[curr] = age
            curr += 1
    return longevity
