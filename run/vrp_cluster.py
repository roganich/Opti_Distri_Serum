## Import the necessary libraries
import numpy as np
import os
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import networkx as nx
import gurobipy as gp
from prettytable import PrettyTable
import csv
import time
import pickle
cwd = os.getcwd()

dataTime = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaTimeDistance.csv'), index_col=0)
##
def vrp_per_cluster(M, d, model_name):
    model = gp.Model(model_name)

    x = dict()
    u = dict()

    for i in M:
        for j in M:
            x[i,j] = model.addVar(vtype=gp.GRB.BINARY, name=f'x_{i}_{j}')

    for i in M:
        u[i] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f'u_{i}')

    for i in M:
        model.addConstr(x[i,i] == 0)

    for i in M:
        model.addConstr(gp.quicksum(x[i,j] for j in M)  == 1)

    for j in M:
        model.addConstr(gp.quicksum(x[i,j] for i in M) == 1)

    for i in M:
        for j in M:
            if j >= 2:
                model.addConstr(u[i] + x[i, j] <= u[j] + (len(M)-1)*(1 - x[i, j]))

    model.setObjective(gp.quicksum(d[i, j]*x[i, j] for i in M for j in M), gp.GRB.MINIMIZE)
    model.setParam('OutputFlag', 1)
    model.update()

    return model

def timeDistance_per_cluster(M):
    distance = dict()
    for i in M:
        for j in M:
            distance[i, j] = dataTime.loc[i][str(j)]

    return distance

##

clusters = pickle.load(open(os.path.join(cwd, 'parameters', 'clusters_GMM.pickle'), 'rb'))

M1 = clusters[1]
d1 = timeDistance_per_cluster(M1)

vrp1 = vrp_per_cluster(M1, d1, 'model_vrp_cluster1')
##
vrp1.optimize()

