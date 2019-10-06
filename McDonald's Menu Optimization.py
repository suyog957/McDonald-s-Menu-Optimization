# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import os
import datetime
import seaborn as sns
from pulp import *
import re 
import matplotlib.pyplot as plt
from IPython.display import Image

df = pd.read_excel('C:/Users/suyog957/Desktop/Python/python Optimization/optimization example/MCDONALDS_menu.xlsx')
print(df.shape)
(df.head())

print(df['Category'].value_counts())

plt.figure(figsize=(16, 6))
sns.set(style="whitegrid")
ax = sns.barplot(x="Category", y="Calories", data=df)

prob = pulp.LpProblem('EatingWhat', pulp.LpMinimize)

decision_variables = []
for rownum, row in df.iterrows():
    variable = str('x' + str(rownum))
    variable = pulp.LpVariable(str(variable), lowBound = 0, upBound = 100, cat= 'Integer') #make variables binary
    decision_variables.append(variable)

print ("Total number of decision_variables: " + str(len(decision_variables)))
print ("Array with Decision Variables:" + str(decision_variables))


total_calorie = ""
for rownum, row in df.iterrows():
    for i, schedule in enumerate(decision_variables):
        if rownum == i:
            formula = row['Calories']*schedule
            total_calorie += formula
prob += total_calorie
print ("Optimization function: " + str(total_calorie))

fat_limit = 90
total_fat_selected = ""
for rownum, row in df.iterrows():
    for i, schedule in enumerate(decision_variables):
        if rownum == i:
            formula = row['Total Fat (% Daily Value)']*schedule
            total_fat_selected += formula
prob += (total_fat_selected == fat_limit)

print(prob)
prob.writeLP("EatingWhat.lp" )

optimization_result = prob.solve()
assert optimization_result == pulp.LpStatusOptimal
print("Status:", LpStatus[prob.status])
print("Optimal Solution to the problem: ", value(prob.objective))
print ("Individual decision_variables: ")
for v in prob.variables():
    print(v.name, "=", v.varValue)

variable_name = []
variable_value = []

for v in prob.variables():
    variable_name.append(v.name)
    variable_value.append(v.varValue)

dt = pd.DataFrame({'variable': variable_name, 'value': variable_value})
for rownum, row in dt.iterrows():
    value = re.findall(r'(\d+)', row['variable'])
    dt.loc[rownum, 'variable'] = int(value[0])

dt = dt.sort_index(by='variable')

#append results
for rownum, row in df.iterrows():
    for results_rownum, results_row in dt.iterrows():
        if rownum == results_row['variable']:
            df.loc[rownum, 'decision'] = results_row['value']
print(df.shape)
(df[:5])

print(df[df['decision'] == 1])

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from matplotlib import animation
from IPython.display import HTML

from autograd import elementwise_grad, value_and_grad
from scipy.optimize import minimize
from collections import defaultdict
from itertools import zip_longest
from functools import partial

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = list(range(0, 2000, 1))         # Calories
y = list(range(0, 2000, 1))      # Fat
z = list(range(0, 2000, 1))      # Proteins

ax.scatter(910, 58, 22, c='r', marker='o')
ax.set_xlabel('Calories')
ax.set_ylabel('Fats')
ax.set_zlabel('Protein')
plt.show()
