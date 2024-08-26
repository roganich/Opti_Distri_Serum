import numpy as np
import os
import pickle
import importlib
import pandas as pd
import matplotlib.pyplot as plt
import csv
import matplotlib.patches as mpatches
import geopandas as gdp
import networkx as nx
from sklearn.mixture import GaussianMixture

cwd = os.getcwd()
dataDepartments = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaDepartments.csv'), index_col=0)


def generateRandomGraph(M: list, degree: int):
    size = len(M)
    graph = nx.random_regular_graph(degree, size)

    mapping = {i: M[i] for i in range(0, size)}
    G = nx.relabel_nodes(graph, mapping)

    connections = dict()
    for u, v in G.edges:
        connections[(u,v)] = True
        connections[(v,u)] = True

    for i in M:
        for j in M:
            if (i,j) not in connections.keys():
                connections[(i,j)] = False

    return G, connections

#
def createGraph(R: list, Mr: dict):
    data = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaCoordinates.csv'), index_col=0)

    G = nx.Graph()
    for r in R:
        list = Mr[r]
        for m in list:
            lat = data.loc[m]['Latitud']
            long = data.loc[m]['Longitud']
            G.add_node(m, pos=(long, lat), color=r)

        for i in list:
            for j in list:
                if i != j:
                    G.add_edge(i, j)
    return G

#
def joinRegions(R1: list, Mr: dict, R2: list):
    data = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaTimeDistance.csv'), index_col=0)
    R1prima = [x for x in R1 if x not in R2]

    for ind2 in R2:
        listMunip = Mr[ind2]
        lessDistance = np.infty
        start = 0
        newR = 0
        for i in listMunip:
            for r in R1prima:
                for j in Mr[r]:
                    time = round((data.loc[i, [str(j)]]), 4)[0]                    
                    if time < lessDistance:
                        lessDistance = time
                        start = i
                        newR = r
        Mr.setdefault(newR, []).append(start)

    R = R1.copy()

    return R, Mr

#Function that generates clusters hierarchically
def generateClustersHierarchical(M: list):
    dataDepartments = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaDepartments.csv'), index_col=0)
    dataCoordinates = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaCoordinates.csv'), index_col=0)

    departments = list()
    for i in M:
        dept = dataCoordinates.loc[i]['Código Departamento']
        if not dept  in departments:
            departments.append(dept)

    regions = list()
    for j in departments:
        reg = dataDepartments.loc[j]['Región']
        if not reg in regions:
            regions.append(reg)

    R = list(range(1, len(departments) + len(regions) + 1))
    Mr = dict()

    for r in R:
        if r <= len(regions):
            listTemp = [11001]
            for i in dataDepartments['Código Centroide'][dataDepartments['Región'] == regions[r - 1]]:
                deptTemp = dataDepartments.index[dataDepartments['Código Centroide'] == i]
                if deptTemp in departments and i not in listTemp:
                    listTemp.append(i)
        else:
            deptTemp = departments[r - len(regions) - 1]
            listTemp = [dataDepartments.loc[deptTemp]['Código Centroide']]
            for i in M:
                if dataCoordinates.loc[i]['Código Departamento'] == deptTemp:
                    listTemp.append(i)
        Mr[r] = listTemp

    G = createGraph(R, Mr)

    connections = dict()
    for u, v in G.edges:
        connections[(u, v)] = True
        connections[(v, u)] = True

    for i in M:
        for j in M:
            if (i, j) not in connections.keys():
                connections[(i, j)] = False

    return G, connections, Mr, R

#
def generateClustersGMM(M: list, BICGraph: bool = False, RegionsGraph: bool = False):
    cwd = os.getcwd()
    data = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaCoordinates.csv'), index_col=0)

    if not 11001 in M:
        M.append(11001)

    departments = list(data.loc[M]['Código Departamento'])
    #X = pd.concat([dataAccidents, dataTime], axis=1).iloc[:,3:]
    #X = dataTime
    X = dataCoordinates[['Latitud', 'Longitud']]

    bestBIC_K = 0
    lowestBIC = np.infty
    vecBIC = list()

    bestAIC_K = 0
    lowestAIC = np.infty
    vecAIC = list()


    limInf = len(dict.fromkeys(list(data.loc[M]['Departamento'])))
    limSup = 50

    for i in range(limInf, limSup):
        GMM = GaussianMixture(n_components=i)
        GMM.fit(X)
        BIC = GMM.bic(X)
        vecBIC.append(BIC)

        if BIC < lowestBIC:
            bestBIC_K = i
            lowestBIC = BIC

    GMMBest = GaussianMixture(n_components=bestBIC_K, random_state=0)
    regions = GMMBest.fit_predict(X)

    if BICGraph == True:
        plt.figure()
        plt.subplot(1, 1, 1)
        plt.plot(list(range(limInf, limSup)), vecBIC, color='coral', marker='o')
        plt.plot(bestBIC_K, lowestBIC, color='red', marker='o')
        plt.grid(linestyle='dashed', color='lightgrey')
        plt.ylabel('BIC')
        plt.xlabel('Components')
        plt.title(f'BIC vs k \n K={bestBIC_K}')
        plt.show()

    matProb = GMMBest.predict_proba(X)
    threshold = 0.3
    R = list()
    Mr = dict()
    for j in range(bestBIC_K):
        listMr = list()
        for i, munip in enumerate(M):
            if matProb[i, j] >= threshold:
                listMr.append(munip)
        R.append(j + 1)
        Mr[j + 1] = listMr


    if RegionsGraph == True:
        fig1, axs1 = plt.subplots(1, 2)
        axs1[0].scatter(X[:, 1], X[:, 0], c=regions)
        axs1[0].set_title(f'Municipios agrupados en {bestBIC_K} regiones')

        axs1[1].scatter(X[:, 1], X[:, 0], c=departments)
        axs1[1].set_title(f'División Política de Colombia')
        fig1.show()

    G = createGraph(R, Mr)
    S = [G.subgraph(c).copy() for c in nx.connected_components(G)]

    if nx.is_connected(G) == True:
        connections = dict()
        for u, v in G.edges:
            connections[(u, v)] = True
            connections[(v, u)] = True

        for i in M:
            for j in M:
                if (i, j) not in connections.keys():
                    connections[(i, j)] = False
        return G, connections
    else:
        unconnectedMunip = M.copy()
        unconnectedReg = list()
        for s in S:
            nodes = list(s.nodes)
            if 11001 in nodes:
                unconnectedMunip = [x for x in unconnectedMunip if x not in nodes]

        for munip in unconnectedMunip:
            ind = M.index(munip)
            r = regions[ind] + 1

            if not r in unconnectedReg:
                unconnectedReg.append(r)

        R_final, Mr_final = joinRegions(R, Mr, unconnectedReg)

        connections = dict()
        G_final = createGraph(R_final, Mr_final)
        for u, v in G_final.edges:
            connections[(u, v)] = True

        for i in M:
            for j in M:
                if (i, j) not in connections.keys():
                    connections[(i, j)] = False

        return G_final, connections, Mr, R

#
def generateClustersGMMCapitals(M: list, BICGraph: bool = False, RegionsGraph: bool = False):
    cwd = os.getcwd()
    data = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaCoordinates.csv'), index_col=0)

    if not 11001 in M:
        M.append(11001)

    departments = list(data.loc[M]['Código Departamento'])
    #X = pd.concat([dataAccidents, dataTime], axis=1).iloc[:,3:]
    #X = dataTime
    X = dataCoordinates[['Latitud', 'Longitud']]

    bestBIC_K = 0
    lowestBIC = np.infty
    vecBIC = list()

    bestAIC_K = 0
    lowestAIC = np.infty
    vecAIC = list()


    limInf = len(dict.fromkeys(list(data.loc[M]['Departamento'])))
    limSup = 50

    for i in range(limInf, limSup):
        GMM = GaussianMixture(n_components=i)
        GMM.fit(X)
        BIC = GMM.bic(X)
        vecBIC.append(BIC)

        if BIC < lowestBIC:
            bestBIC_K = i
            lowestBIC = BIC

    GMMBest = GaussianMixture(n_components=bestBIC_K, random_state=0)
    regions = GMMBest.fit_predict(X)

    if BICGraph == True:
        plt.figure()
        plt.subplot(1, 1, 1)
        plt.plot(list(range(limInf, limSup)), vecBIC, color='coral', marker='o')
        plt.plot(bestBIC_K, lowestBIC, color='red', marker='o')
        plt.grid(linestyle='dashed', color='lightgrey')
        plt.ylabel('BIC')
        plt.xlabel('Components')
        plt.title(f'BIC vs k \n K={bestBIC_K}')
        plt.show()

    matProb = GMMBest.predict_proba(X)
    threshold = 0.3
    R = list()
    Mr = dict()

    for j in range(bestBIC_K):
        listMr = list()
        for i, munip in enumerate(M):
            if matProb[i, j] >= threshold:
                listMr.append(munip)
        R.append(j + 1)
        Mr[j + 1] = listMr

    departments = list()
    for i in M:
        dept = dataCoordinates.loc[i]['Código Departamento']
        if not dept in departments:
            departments.append(dept)

    listCapitals = [dataDepartments.loc[deptTemp]['Código Centroide'] for deptTemp in departments]

    R.append(len(R)+1)
    Mr[len(R)] = listCapitals


    if RegionsGraph == True:
        fig1, axs1 = plt.subplots(1, 2)
        axs1[0].scatter(X[:, 1], X[:, 0], c=regions)
        axs1[0].set_title(f'Municipios agrupados en {bestBIC_K} regiones')

        axs1[1].scatter(X[:, 1], X[:, 0], c=departments)
        axs1[1].set_title(f'División Política de Colombia')
        fig1.show()

    G = createGraph(R, Mr)
    S = [G.subgraph(c).copy() for c in nx.connected_components(G)]

    if nx.is_connected(G) == True:
        connections = dict()
        for u, v in G.edges:
            connections[(u, v)] = True
            connections[(v, u)] = True

        for i in M:
            for j in M:
                if (i, j) not in connections.keys():
                    connections[(i, j)] = False
        return G, connections, Mr, R
    else:
        unconnectedMunip = M.copy()
        unconnectedReg = list()
        for s in S:
            nodes = list(s.nodes)
            if 11001 in nodes:
                unconnectedMunip = [x for x in unconnectedMunip if x not in nodes]

        for munip in unconnectedMunip:
            ind = M.index(munip)
            r = regions[ind] + 1

            if not r in unconnectedReg:
                unconnectedReg.append(r)

        R_final, Mr_final = joinRegions(R, Mr, unconnectedReg)

        connections = dict()
        G_final = createGraph(R_final, Mr_final)
        for u, v in G_final.edges:
            connections[(u, v)] = True

        for i in M:
            for j in M:
                if (i, j) not in connections.keys():
                    connections[(i, j)] = False

        return G_final, connections, Mr, R

#
def generateClustersGMMBridges(M: list, BICGraph: bool = False, ReggionsGraph: bool = False):
    G_old, connections_old, Mr, R = generateClustersGMM(M, False, False)
    whole_list = list()
    for r in R:
        whole_list += Mr[r]
    bridges = list()
    for munip in whole_list:
        if whole_list.count(munip) >= 2:
            if not munip in bridges:
                bridges.append(munip)

    R.append(len(R)+1)
    Mr[len(R)] = bridges

    connections = dict()
    G_final = createGraph(R, Mr)
    for u, v in G_final.edges:
        connections[(u, v)] = True
        connections[(v, u)] = True

    for i in M:
        for j in M:
            if (i, j) not in connections.keys():
                connections[(i, j)] = False

    return G_final, connections, Mr, R

dataCoordinates = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaCoordinates.csv'), index_col=0)

filepath_munipCol = os.path.join(cwd, 'parameters', 'DataMUNIP.shp')

gdf_munipCol = gdp.read_file(filepath_munipCol, index='MPIO_CCNCT')

list_degree = [10, 16, 20, 26, 30]

list_Graphs = ['hier', 'GMM', 'GMMCapitals', 'GMMBridges']

M = list()
T = [1, 2, 3, 4, 5, 6]
listMunipPruebas = list(dataCoordinates['Departamento'])

for i, munip in enumerate(dataCoordinates['Departamento']):
    if munip in listMunipPruebas:
        M.append(list(dataCoordinates.index)[i])

    
G_hier, connections_hier, Mr_hier, R_hier = generateClustersHierarchical(M)

G_GMM, connections_GMM, Mr_GMM, R_GMM = generateClustersGMM(M)
G_GMMCapitals, connections_GMMCapitals, Mr_GMMCapitals, R_GMMCapitals = generateClustersGMMCapitals(M, False, False)
G_GMMBridges, connections_GMMBridges, Mr_GMMBridges, R_GMMBridges = generateClustersGMMBridges(M, False, False)
list_Mr = [Mr_hier, Mr_GMM, Mr_GMMCapitals, Mr_GMMBridges]
list_namesGraphs = ['hier', 'GMM', 'GMMCapitals', 'GMMBridges']

dict_networks = {}
for munip_temp in M:
    dict_temp = {}
    for idx, Mr_temp in enumerate(list_Mr):
        for key in Mr_temp.keys():
            listTemp = Mr_GMMBridges[key]
            if munip_temp in listTemp:
                dict_temp[list_namesGraphs[idx]] = key
                break

    dict_networks[munip_temp] = dict_temp

df_networks = pd.DataFrame(list(dict_networks.values()), index=dict_networks.keys())

print

#print(df_networks)

df_networks.to_csv(os.path.join(cwd, 'results', 'clusters.csv'))

#gdf_munipCol = gdf_munipCol.merge(df_networks, left_index=True, right_index=True)
#gdf_munipCol.to_file(os.path.join(cwd, 'results', 'munipColClusters.shp'))
