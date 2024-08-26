##
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
from tqdm import tqdm
cwd = os.getcwd()


dataTime = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaTimeDistance.csv'), index_col=0)
dataCoordinates = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaCoordinates.csv'), index_col=0)

##

results_routes = pickle.load(open(os.path.join(cwd, 'results', 'tsp_routes.pickle'), 'rb'))
##
clusters = results_routes['clusters']
TSP_routes = results_routes['TSP_routes']
TSP_distance = results_routes['TSP_distance']
TSP_running_time = results_routes['TSP_running_time']
NN_routes = results_routes['NN_routes']
NN_distance = results_routes['NN_distance']
NN_running_time = results_routes['NN_running_time']
size_clusters = {key:len(value) for key, value in clusters.items() if not key in [15, 18, 20, 25, 28]}
cluster = {key:value for key, value in clusters.items() if not key in [15, 18, 20, 25, 28]}

##
fig, ax = plt.subplots(1,2, figsize=(10,6))

ax[0].hist(NN_distance.values(), color='blue', density=True)
ax[0].set_xlabel('Tiempo de viaje total [h]')
ax[0].set_title('NN')

ax[1].hist(TSP_distance.values(), color='red', density=True)
ax[1].set_xlabel('Tiempo de viaje total [h]')
ax[1].set_title('TSP')

fig.show()


##

fig, ax = plt.subplots(1,2, figsize=(10,6))

ax[0].hist(NN_running_time.values(), color='blue', density=False)
ax[0].set_xlabel('Tiempo computacional [s]')
ax[0].set_title('NN')

ax[1].hist(TSP_running_time.values(), color='red', density=False)
ax[1].set_xlabel('Tiempo computacional [s]')
ax[1].set_title('TSP')

fig.show()

##

plt.figure()
plt.scatter(size_clusters.values(), NN_distance.values(), color='blue', label='NN')
plt.scatter(size_clusters.values(), TSP_distance.values(), color='red', label='TSP')
plt.legend()
plt.ylabel('Tiempo total de la ruta')
plt.xlabel('Tamaño de cluster')
plt.grid()
plt.show()


##
df_results = pd.DataFrame(list(zip(size_clusters.values(), NN_routes.values())), columns=['size', 'NN_routes'])
df_results['NN_distance'] = list(NN_distance.values())
df_results['NN_running_time'] = list(NN_running_time.values())

df_results['TSP_routes'] = list(TSP_routes.values())
df_results['TSP_distance'] = list(TSP_distance.values())
df_results['TSP_running_time'] = list(TSP_running_time.values())

df_results.to_csv(os.path.join(cwd, 'results', 'df_routes.csv'))