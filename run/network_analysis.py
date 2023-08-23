##
import numpy as np
import os
import pandas as pd
from debugpy.adapter.servers import connections
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import networkx as nx
import gurobipy as gp
from prettytable import PrettyTable
import random

##
cwd = os.getcwd()
dataCoordinates = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaCoordinates.csv'), index_col=0)
dataTime = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaTimeDistance.csv'), index_col=0)
dataAccidents = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaOA2019.csv'), index_col=0)
dataDepartments = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaDepartments.csv'), index_col=0)
dataAccidentRates = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaAccidentRates.csv'), index_col=0)

##
def connections_fromGraph(G):
    connections = dict()
    for u, v in G.edges:
        connections[(u, v)] = True
        connections[(v, u)] = True

    return G, connections

def readParametersDet(filename: str):
        cwd = os.getcwd()
        txt = open(os.path.join(cwd, 'parameters', filename), 'r')

        M = list()  # Set of regions interconnected
        T = list()  # Time horizon [months]

        delta = int()  # Cantidad de viales promedio utilizados por persona
        p = int()  # Costo de producción de un vial de suero antiofífico
        V = int()  # Capacacidad máxima de viales para un centro médico
        mu = dict()  # Accidentes ofídicos esperados para todos los tipos
        k = int()  # Capacidad máxima de transporte terrestre
        g = int()  # Costo de una hora de transporte terreste
        h = dict()  # Distancia en tiempo entre los centros médicos
        I0 = dict()  # Inventario inicial de los centros médicos
        q = dict()  # Conexiones entre municipios, 1: Si existe, 0: No existe
        DW = float()  # Peso de la enfermedad
        alpha = float()  # Probabilidad de MUERTE?
        gamma = float()  # Probabilidad de COMORBILIDAD?
        b = float()  # Costo de año de vida
        L = float()  # Años de expectativa de vida

        currentParameter = 'ninguno'
        for line in txt:
            if len(line) == 2 or len(line) == 1:
                currentParameter = 'ninguno'
                continue
            else:
                if currentParameter == 'ninguno':
                    if line.find('M:=') != -1:
                        startChar = line.find('[')
                        endChar = line.find(']')
                        M = list(map(int, (line[startChar + 1:endChar].split(','))))
                    elif line.find('T:=') != -1:
                        startChar = line.find('[')
                        endChar = line.find(']')
                        T = list(map(int, line[startChar + 1:endChar].split(',')))
                    elif line.find('delta:=') != -1:
                        startChar = line.find('=')
                        endChar = line.find('//')
                        delta = int(line[startChar + 1:endChar])
                    elif line.find('p:=') != -1:
                        startChar = line.find('=')
                        endChar = line.find('//')
                        p = int(line[startChar + 1:endChar])
                    elif line.find('V:=') != -1:
                        startChar = line.find('=')
                        endChar = line.find('//')
                        V = int(line[startChar + 1:endChar])
                    elif line.find('k:=') != -1:
                        startChar = line.find('=')
                        endChar = line.find('//')
                        k = int(line[startChar + 1:endChar])
                    elif line.find('g:=') != -1:
                        startChar = line.find('=')
                        endChar = line.find('//')
                        g = int(line[startChar + 1:endChar])
                    elif line.find('DW:=') != -1:
                        startChar = line.find('=')
                        endChar = line.find('//')
                        DW = float(line[startChar + 1:endChar])
                    elif line.find('alpha:=') != -1:
                        startChar = line.find('=')
                        endChar = line.find('//')
                        alpha = float(line[startChar + 1:endChar])
                    elif line.find('gamma:=') != -1:
                        startChar = line.find('=')
                        endChar = line.find('//')
                        gamma = float(line[startChar + 1:endChar])
                    elif line.find('b:=') != -1:
                        startChar = line.find('=')
                        endChar = line.find('//')
                        b = float(line[startChar + 1:endChar])
                    elif line.find('L:=') != -1:
                        startChar = line.find('=')
                        endChar = line.find('//')
                        L = float(line[startChar + 1:endChar])
                    elif line.find('h:=') != -1:
                        currentParameter = 'h'
                    elif line.find('mu:=') != -1:
                        currentParameter = 'mu'
                    elif line.find('I0:=') != -1:
                        currentParameter = 'I0'
                    elif line.find('q:=') != -1:
                        currentParameter = 'q'
                else:
                    startKey = line.find('(')
                    endKey = line.find(')')
                    startValue = line.find('=')
                    endValue = line.find('//')
                    if currentParameter == 'mu':
                        keyTemp = list(line[startKey + 1:endKey].split(','))
                        mu[(int(keyTemp[0]), int(keyTemp[1]))] = float(line[startValue + 1:endValue])
                    elif currentParameter == 'h':
                        keyTemp = line[startKey + 1:endKey].split(',')
                        h[(int(keyTemp[0]), int(keyTemp[1]))] = float(line[startValue + 1:endValue])
                    elif currentParameter == 'I0':
                        keyTemp = line[startKey + 1:endKey]
                        I0[int(keyTemp)] = float(line[startValue + 1:endValue])
                    elif currentParameter == 'q':
                        keyTemp = line[startKey + 1:endKey].split(',')
                        q[(int(keyTemp[0]), int(keyTemp[1]))] = int(line[startValue + 1:endValue])

        txt.close()

        return M, T, delta, p, V, mu, k, g, h, I0, q, DW, alpha, gamma, b, L


list_degree = [10, 16, 20, 26, 30]
graph_names = [f'randD{i}' for i in list_degree]

graph_names += ['hier', 'GMM', 'GMMCapitals', 'GMMBridges']


for graph in graph_names:
    G_temp = nx.read_gpickle(os.path.join(cwd, 'parameters', f'{graph}.gpickle'))
    connections_temp = connections_fromGraph(G_temp)[1]
    M, T, delta, p, V, mu, k, g, h, I0, q, DW, alpha, gamma, b, L = readParametersDet(f'params_det_{graph}.txt')

    matGraph_temp = np.zeros((len(M), len(M)))
    for (M1,M2) in list(connections_temp.keys()):

        ind1 = M.index(M1)
        ind2 = M.index(M2)
        matGraph_temp[ind1, ind2] = 1

    plt.figure()
    plt.title(f'Matrices de viajes para {graph}')
    plt.imshow(matGraph_temp)
    plt.show()





