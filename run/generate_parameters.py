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
import pickle

##
cwd = os.getcwd()
dataCoordinates = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaCoordinates.csv'), index_col=0)
dataTime = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaTimeDistance.csv'), index_col=0)
dataAccidents = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaOA2019.csv'), index_col=0)
dataDepartments = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaDepartments.csv'), index_col=0)
dataAccidentRates = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaAccidentRates.csv'), index_col=0)


M = list()
T = [1, 2, 3, 4, 5, 6]
listMunipPruebas = list(dataCoordinates['Departamento'])


for i, munip in enumerate(dataCoordinates['Departamento']):
    if munip in listMunipPruebas:
        M.append(list(dataCoordinates.index)[i])



data = pd.concat([dataAccidents, dataTime], axis=1)
data = data.iloc[:,3:]

##

#Función que retira una cantidad de municipios según el promedio de distancia que se requiere para llegar
# def remove_munip_2far(M):
#     ave_distance = dataTime.mean(axis=1)
#     newM = [i for i in M if ave_distance[i] < 60]
#     return newM
#
# newM = remove_munip_2far(M)
##
#Functions that creates a random graph [Returns Graph and Connections associated]
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
                    time = round((data.loc[i, [str(j)]]), 4)
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

#Function to write down the distances between the cities [Doesnt Return]
def write_h(connections:dict, filename: str):
    cwd = os.getcwd()
    data = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaTimeDistance.csv'), index_col=0)

    file = open(os.path.join(cwd, 'parameters', filename), 'w+')
    file.write('h:={\n')

    for (i,j) in connections:
        if connections[(i,j)] == True:
            hTemp = round(float(data.loc[i, str(j)]), 4)
            file.write(f'\t({i},{j})={hTemp}//\n')

    file.write('}\n')
    file.close()

#Function to write down the ophidic accidents and the initial inventory [Doesnt Return]
def write_mu_I0(M:list, T: list, filename: str):
    cwd = os.getcwd()
    delta = 5
    data = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaOA2019.csv'), index_col=0)

    file = open(os.path.join(cwd, 'parameters', filename), 'w+')
    file.write('mu:={\n')

    for t in T:
        for m in M:
            value = data.loc[m, [str(t)]]
            test = value.loc[m]
            muTemp = np.round(value.loc[m], 4)
            file.write(f'\t({t},{m})={muTemp}//\n')


    file.write('}\n')

    file.write('\nI0:={\n')

    for m in M:
        if round((data.loc[m, ['1']]), 4) > 0:
            I0Temp = round((data.loc[m, ['1']]), 4)*delta
        else:
            I0Temp = delta
        file.write(f'\t({m})={I0Temp}//\n')

    file.write('}\n')
    file.close()

#Function to write down the connections and the time scale [Doesnt Return]
def write_M_q_T(M: list, connections: dict, T: list, filename: str):
    cwd = os.getcwd()
    file = open(os.path.join(cwd, 'parameters', filename), 'w+')

    strM = map(str, M)
    strM = ','.join(strM)
    file.write(f'\nM:=[{strM}]\n')

    file.write('\nq:={\n')
    for (i,j) in connections:
        if connections[(i,j)] == True:
            file.write(f'\t({i},{j})=1//\n')
        else:
            file.write(f'\t({i},{j})=0//\n')

    file.write('}\n')

    strT = map(str, T)
    strT = ','.join(strT)
    file.write(f'\nT:=[{strT}]\n')

    file.close()

#Function that joins the common parameters with the specific from each topology [Return Null]
def joinParametersDet(filename: str, h_file: str, mu_I0_file: str, q_T_file: str):
    cwd = os.getcwd()
    filenames = [q_T_file, h_file, mu_I0_file, 'delta_V_g_p_DALY.txt']
    with open(os.path.join(cwd, 'parameters', filename), 'w') as outfile:
        for fname in filenames:
            with open(os.path.join(cwd, 'parameters', fname)) as infile:
                for line in infile:
                    outfile.write(line)
            #if filename != 'delta_V_g_p_DALY.txt':
                #os.remove(os.path.join(cwd, 'parameters', filename))

#
def average_degree(G):
    nodes = G.nodes()
    ave_degree = 0
    for n in nodes:
        ave_degree += G.degree(n)
    return ave_degree/len(nodes)


##
list_degree = [10, 16, 20, 26, 30]
#
# for degree_temp in list_degree:
#     name_temp = f'randD{degree_temp}'
#     G_temp, connection_temp = generateRandomGraph(M, degree_temp)
#     nx.write_gpickle(G_temp, os.path.join(cwd, 'parameters', name_temp+'.gpickle'))
#     write_h(connection_temp, f'h_{name_temp}.txt')
#     write_mu_I0(M, T, f'mu_I0_{name_temp}.txt')
#     write_M_q_T(M, connection_temp, T, f'q_T_{name_temp}.txt')
#     joinParametersDet(f'params_det_{name_temp}.txt',f'h_{name_temp}.txt',  f'mu_I0_{name_temp}.txt', f'q_T_{name_temp}.txt')
#     print(f'Parámetros para redes aleatorias de grado {degree_temp} creada!')


G_hier, connections_hier, Mr_hier, R_hier = generateClustersHierarchical(M)
with open (os.path.join(cwd, 'parameters', 'hier.gpickle'), 'wb') as f:
    pickle.dump(G_hier, f, pickle.HIGHEST_PROTOCOL)
f.close()
write_h(connections_hier, 'h_hier.txt')
write_mu_I0(M, T, 'mu_I0_hier.txt')
write_M_q_T(M, connections_hier, T, 'q_T_hier.txt')
joinParametersDet('params_det_hier.txt', 'h_hier.txt', 'mu_I0_hier.txt', 'q_T_hier.txt')
print(f'Parámetros para redes jerárquicas creados!')


G_GMM, connections_GMM, Mr_GMM, R_GMM = generateClustersGMM(M)
with open (os.path.join(cwd, 'parameters', 'GMM.gpickle'), 'wb') as f:
    pickle.dump(G_GMM, f, pickle.HIGHEST_PROTOCOL)
f.close()
write_h(connections_GMM, 'h_GMM.txt')
write_mu_I0(M, T, 'mu_I0_GMM.txt')
write_M_q_T(M, connections_GMM, T, 'q_T_GMM.txt')
joinParametersDet('params_det_GMM.txt', 'h_GMM.txt', 'mu_I0_GMM.txt', 'q_T_GMM.txt')
print(f'Parámetros para redes GMM creados!')

##

file_ClustersGMM = open(os.path.join(cwd, 'parameters', 'clusters_GMM.pickle'), 'wb')
pickle.dump(Mr_GMM, file_ClustersGMM)
file_ClustersGMM.close()

##

G_GMMCapitals, connections_GMMCapitals, Mr_GMMCapitals, R_GMMCapitals = generateClustersGMMCapitals(M, False, False)
with open (os.path.join(cwd, 'parameters', 'GMMCapitals.gpickle'), 'wb') as f:
    pickle.dump(G_GMMCapitals, f, pickle.HIGHEST_PROTOCOL)
write_h(connections_GMMCapitals, 'h_GMMCapitals.txt')
write_mu_I0(M, T, 'mu_I0_GMMCapitals.txt')
write_M_q_T(M, connections_GMMCapitals, T, 'q_T_GMMCapitals.txt')
joinParametersDet('params_det_GMMCapitals.txt', 'h_GMMCapitals.txt', 'mu_I0_GMMCapitals.txt', 'q_T_GMMCapitals.txt')
print(f'Parámetros para redes GMM con capitales creados!')


G_GMMBridges, connections_GMMBridges, Mr_GMMBridges, R_GMMBridges = generateClustersGMMBridges(M, False, False)
with open (os.path.join(cwd, 'parameters', 'GMMBridges.gpickle'), 'wb') as f:
    pickle.dump(G_GMMBridges, f, pickle.HIGHEST_PROTOCOL)
write_h(connections_GMMBridges, 'h_GMMBridges.txt')
write_mu_I0(M, T, 'mu_I0_GMMBridges.txt')
write_M_q_T(M, connections_GMMBridges, T, 'q_T_GMMBridges.txt')
joinParametersDet('params_det_GMMBridges.txt', 'h_GMMBridges.txt', 'mu_I0_GMMBridges.txt', 'q_T_GMMBridges.txt')
print(f'Parámetros para redes GMM con puentes creados!')


##

tableGraphs = PrettyTable(['Type', 'Edges', 'Ave. Degree', 'Diameter', 'Radius', 'Connected components'])

list_Graphs = [f'randD{i}' for i in list_degree]
list_Graphs += ['hier', 'GMM', 'GMMCapitals', 'GMMBridges']

for G_name in list_Graphs:
    with open (os.path.join(cwd,'parameters',f'{G_name}.gpickle'), 'rb') as f:
        G_temp = pickle.load(f)
    tableGraphs.add_row([G_name, G_temp.size(), np.round(average_degree(G_temp), 3),
                         nx.diameter(G_temp), nx.radius(G_temp), nx.number_connected_components(G_temp)])

print(tableGraphs)

##
dfColor = pd.read_csv(os.path.join(cwd, 'parameters','ColoresDepartamento.csv'), sep=';')

colorMapHier = []
colorMapGMM = []
cluster = []

dictPositions = dict()

for cod in list(dataCoordinates.index):
    longitudTemp = dataCoordinates.loc[cod]['Longitud']
    latitudTemp = dataCoordinates.loc[cod]['Latitud']

    dictPositions[cod] = (float(longitudTemp), float(latitudTemp))

colorMapHier = []

for node in G_hier:
    deptTemp = dataCoordinates.loc[node]['Código Departamento']
    colorTemp = dfColor[dfColor['Código Departamento'] == deptTemp].iloc[0, 2]

    colorMapHier.append(f'#{colorTemp}')

with open (os.path.join(cwd, 'parameters', 'hier.gpickle'), 'rb') as f:
    G_hier = pickle.load(f)

colorMapGMM = []
for node in G_GMM:
    for key in Mr_GMM.keys():
        listTemp = Mr_GMM[key]
        if node in listTemp:
            colorTemp = dfColor.loc[key-1]['Color']
            colorMapGMM.append(f'#{colorTemp}')
            break

colorMapGMMCapitals = []
for node in G_GMMCapitals:
    for key in Mr_GMMCapitals.keys():
        listTemp = Mr_GMMCapitals[key]
        if node in listTemp:
            colorTemp = dfColor.loc[key-1]['Color']
            colorMapGMMCapitals.append(f'#{colorTemp}')
            break

colorMapGMMBridges = []
for node in G_GMMBridges:
    for key in Mr_GMMBridges.keys():
        listTemp = Mr_GMMBridges[key]
        if node in listTemp:
            colorTemp = dfColor.loc[key-1]['Color']
            colorMapGMMBridges.append(f'#{colorTemp}')
            break


sizeMap = [100]*len(dataCoordinates.index)
'''
plt.figure(figsize=(5,7))
ax = plt.gca()
ax.set_title('Clusters Jerárquicos', fontweight='bold')
nx.draw(G_hier, pos=dictPositions, node_color=colorMapHier, node_size=sizeMap, ax=ax)
ax.axis('off')
plt.show()

plt.figure(figsize=(5,7))
ax = plt.gca()
ax.set_title('Clusters GMM', fontweight='bold')
nx.draw(G_GMM, pos=dictPositions, node_color=colorMapGMM, node_size=sizeMap, ax=ax)
ax.axis('off')
plt.show()'''
#
# plt.figure(figsize=(5,7))
# ax = plt.gca()
# ax.set_title('Clusters GMM Capitals', fontweight='bold')
# nx.draw(G_GMMCapitals, pos=dictPositions, node_color=colorMapGMMCapitals, node_size=sizeMap, ax=ax)
# ax.axis('off')
# plt.show()
#
# plt.figure(figsize=(5,7))
# ax = plt.gca()
# ax.set_title('Clusters GMM Bridges', fontweight='bold')
# nx.draw(G_GMMBridges, pos=dictPositions, node_color=colorMapGMMBridges, node_size=sizeMap, ax=ax)
# ax.axis('off')
# plt.show()
