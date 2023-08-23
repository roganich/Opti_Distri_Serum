##
import numpy as np
import os
import pandas as pd
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

T_test = [1, 2, 3, 4, 5, 6]
M_prueba = list()
listMunipPruebas = list(dataCoordinates['Departamento'])

M_pacifico = [11001]

for i, munip in enumerate(dataCoordinates['Departamento']):
    if munip in listMunipPruebas:
        M_prueba.append(list(dataCoordinates.index)[i])

    if munip in ['Cauca', 'Choco', 'Antioquia', 'Nariño']:
        M_pacifico.append(list(dataCoordinates.index)[i])


def generateClustersRandom(M: list, k: int):
    if 11001 in M:
        M.remove(11001)
    R = list(np.arange(1, k+1, 1))
    Mr = dict()
    sizeCluster = len(M) // k
    residue = len(M) % k

    for r in R:
        if r == 1:
            listTemp = [11001]
        else:
            listTemp = [random.choice(Mr[r-1])]
        finish = False
        sizeTemp = 0
        while not finish:
            if len(M) > 0:
                randMunip = np.random.choice(M)
                if not randMunip in listTemp:
                    listTemp.append(randMunip)
                    M.remove(randMunip)
                    sizeTemp += 1

                if residue > 0:
                    if sizeTemp == sizeCluster + 1:
                        residue -= 1
                        finish = True
                else:
                    if sizeTemp == sizeCluster:
                        finish = True
            else:
                finish = True
        Mr[r] = listTemp

    return R, Mr


def generateRandomGraph(M: list, degree: int):
    size = len(M)
    graph = nx.random_regular_graph(degree, size)

    mapping = {i: M[i] for i in range(0, size)}
    G = nx.relabel_nodes(graph, mapping)

    connections = dict()
    for u, v in G.edges:
        connections[(u,v)] = True

    for i in M:
        for j in M:
            if (i,j) not in connections.keys():
                connections[(i,j)] = False

    return G, connections

M_test1 = M_prueba.copy()
connections_rand = generateRandomGraph(M_test1, 10)


##

def generateClustersHierarchical(M: list):
    dataDepartments = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaDepartments.csv'), index_col=0)
    dataCoordinates = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaCoordinates.csv'), index_col=0)

    departments = list()
    for i in M:
        dept = dataCoordinates.loc[i]['Código Departamento']
        if not dept in departments:
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

    connections = dict()
    for u, v in G.edges:
        connections[(u, v)] = True

    for i in M:
        for j in M:
            if (i, j) not in connections.keys():
                connections[(i, j)] = False

    return R, Mr


def generateClustersGMM(M: list, BICGraph: bool = False, RegionsGraph: bool = False):
    cwd = os.getcwd()
    data = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaCoordinates.csv'), index_col=0)

    if not 11001 in M:
        M.append(11001)

    departments = list(data.loc[M]['Código Departamento'])
    X = np.array(data.loc[M, ['Latitud', 'Longitud']])

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
        return R, Mr
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
        return R_final, Mr_final


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
                    time = round(float(data.loc[i, [str(j)]]), 4)
                    if time < lessDistance:
                        lessDistance = time
                        start = i
                        newR = r
        Mr.setdefault(newR, []).append(start)

    R = R1.copy()

    return R, Mr


def write_h(R: list, Mr: dict):
    cwd = os.getcwd()
    data = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaTimeDistance.csv'), index_col=0)

    file = open(os.path.join(cwd, 'parameters', 'h.txt'), 'w+')
    file.write('h:={\n')

    for r in R:
        for i in Mr[r]:
            for j in Mr[r]:
                hTemp = round(float(data.loc[i, [str(j)]]), 4)
                file.write(f'\t({i},{j})={hTemp}//\n')

    file.write('}\n')
    file.close()


def write_R_M_T(R: list, Mr: dict, T: list):
    cwd = os.getcwd()
    file = open(os.path.join(cwd, 'parameters', 'R_M_T.txt'), 'w+')

    strR = map(str, R)
    strR = ','.join(strR)
    file.write(f'R:=[{strR}]\n')

    file.write('\nMr:={\n')
    for r in R:
        strM = map(str, Mr[r])
        strM = ','.join(strM)
        file.write(f'\t({r})=[{strM}]//\n')
    file.write('}\n')

    strT = map(str, T)
    strT = ','.join(strT)
    file.write(f'\nT:=[{strT}]\n')

    file.close()


def write_mu_I0(R: list, T: list, Mr: dict):
    cwd = os.getcwd()
    data = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaOA2019.csv'), index_col=0)

    file = open(os.path.join(cwd, 'parameters', 'mu.txt'), 'w+')
    file.write('mu:={\n')

    for r in R:
        for m in Mr[r]:
            for t in T:
                muTemp = round(float(data.loc[m, [str(t)]]), 4)
                file.write(f'\t({t},{m})={muTemp}//\n')

    file.write('}\n')
    file.close()

    file = open(os.path.join(cwd, 'parameters', 'I0.txt'), 'w+')
    file.write('I0:={\n')

    for r in R:
        for m in Mr[r]:
            if round(float(data.loc[m, ['1']]), 4) > 0:
                I0Temp = round(float(data.loc[m, ['1']]), 4)*5
            else:
                I0Temp = 5
            file.write(f'\t({m})={I0Temp}//\n')

    file.write('}\n')
    file.close()


def joinParametersDet():
    cwd = os.getcwd()
    filenames = ['R_M_T.txt', 'h.txt', 'mu.txt', 'delta_V_g_p_DALY.txt', 'I0.txt']
    with open(os.path.join(cwd, 'parameters', 'params_det.txt'), 'w') as outfile:
        for fname in filenames:
            with open(os.path.join(cwd, 'parameters', fname)) as infile:
                for line in infile:
                    outfile.write(line)


def readParametersDet(filename: str):
    cwd = os.getcwd()
    txt = open(os.path.join(cwd, 'parameters', filename), 'r')

    R = list()  # Set of regions interconnected
    T = list()  # Time horizon [months]
    Mr = dict()  # Subset of municipalities in a region r

    delta = int()  # Cantidad de viales promedio utilizados por persona
    p = int()  # Costo de producción de un vial de suero antiofífico
    V = int()  # Capacacidad máxima de viales para un centro médico
    mu = dict()  # Accidentes ofídicos esperados para todos los tipos
    k = int()  # Capacidad máxima de transporte terrestre
    g = int()  # Costo de una hora de transporte terreste
    h = dict()  # Distancia en tiempo entre los centros médicos
    I0 = dict() # Inventario inicial de los centros médicos

    currentParameter = 'ninguno'
    for line in txt:
        if len(line) == 2 or len(line) == 1:
            currentParameter = 'ninguno'
            continue
        else:
            if currentParameter == 'ninguno':
                if line.find('R:=') != -1:
                    startChar = line.find('[')
                    endChar = line.find(']')
                    R = list(map(int, (line[startChar + 1:endChar].split(','))))
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
                elif line.find('h:=') != -1:
                    currentParameter = 'h'
                elif line.find('mu:=') != -1:
                    currentParameter = 'mu'
                elif line.find('Mr:=') != -1:
                    currentParameter = 'Mr'
                elif line.find('I0:=') != -1:
                    currentParameter = 'I0'
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
                elif currentParameter == 'Mr':
                    keyTemp = line[startKey + 1:endKey]
                    startChar = line.find('[')
                    endChar = line.find(']')
                    Mr[int(keyTemp)] = list(map(int, (line[startChar + 1:endChar].split(','))))
                elif currentParameter == 'I0':
                    keyTemp = line[startKey + 1:endKey]
                    I0[int(keyTemp)] = float(line[startValue + 1:endValue])


    txt.close()

    return R, T, Mr, delta, p, mu, V, k, g, h, I0


def createModelDet(modelname: str, filename: str, result_file: str):
    R, T, Mr, delta, p, mu, V, k, g, h, I0 = readParametersDet(filename)
    b = 12*10**3
    L = 49.7
    alpha = 0.25
    DW = 0.144
    gamma = 0.25

    model = gp.Model(modelname)

    w = dict()
    I = dict()
    f = dict()
    for t in T:
        w[t] = model.addVar(vtype=gp.GRB.INTEGER, name=f'w_{t}')
        for r in R:
            for m in Mr[r]:
                if not (t, m) in I.keys():
                    I[(t, m)] = model.addVar(vtype=gp.GRB.INTEGER, name=f'I_{t}_{m}')
                    f[(t, m)] = model.addVar(vtype=gp.GRB.INTEGER, name=f'f_{t}_{m}')


    y = dict()
    x = dict()
    for t in T:
        for r in R:
            for i in Mr[r]:
                for j in Mr[r]:
                    if i != j and (t, i, j) not in y.keys():
                        y[(t, i, j)] = model.addVar(vtype=gp.GRB.BINARY, name=f'y_{t}_{i}_{j}')
                        x[(t, i, j)] = model.addVar(vtype=gp.GRB.INTEGER, name=f'x_{t}_{i}_{j}')


    for t in T[:-1]:
        for r in R:
            for m in Mr[r]:
                if t == 1:
                    if m != 11001:
                        model.addConstr(I[t, m] == I0[m])
                else:
                    sent = gp.quicksum(x[t, m, j] for j in Mr[r] if m != j)
                    received = gp.quicksum(x[t, i, m] for i in Mr[r] if m != i)
                    if m == 11001:
                        model.addConstr(I[t+1, m] == I[t, m] - delta*mu[t, m] - sent + received + w[t] + f[t, m])
                    else:
                        model.addConstr(I[t+1, m] == I[t, m] - delta*mu[t, m] - sent + received + f[t, m])

    for t in T:
        for r in R:
            for m in Mr[r]:
                model.addConstr(I[t, m] <= V)
                model.addConstr(I[t, m] >= delta)

    for t in T:
        for r in R:
            for i in Mr[r]:
                for j in Mr[r]:
                    if i != j:
                        model.addConstr(k*y[t, i, j] >= x[t, i, j])

    cost_shipping = gp.quicksum(g*h[i, j]*y[t, i, j] for t in T for r in R for i in Mr[r] for j in Mr[r] if i != j)
    cost_production = p*(gp.quicksum(w[t] for t in T) + gp.quicksum(I[1, m] for r in R for m in Mr[r]))
    cost_DALY = gp.quicksum(b*L*(alpha*f[t,m]*DW + gamma*f[t, m]) for r in R for m in Mr[r] for t in T)

    model.setObjective(cost_production+cost_shipping+cost_DALY, gp.GRB.MINIMIZE)
    model.setParam('ResultFile', os.path.join(cwd, 'results', result_file))
    model.update()

    return model


def analyzeMIPModel(model: gp.Model, result_file: str):
    cwd = os.getcwd()
    if model.status == 1:
        cols = model.NumVars
        rows = model.NumConstrs
        intVars = model.NumIntVars
        binVars = model.NumBinVars
        numCoeffs = cols*rows
        nonZeroCoeffs = model.NumNZs

        dfResults = pd.DataFrame(columns=['Nombre', 'Restricciones', 'Variables', 'Variables Enteras', 'Variables Binarias',
                                          'Coeficientes', 'Coeficientes No Nulos'])
        dfResults.loc[0] = [model.ModelName, cols, rows, intVars, binVars, numCoeffs, nonZeroCoeffs]
        dfResults.to_csv(os.path.join(cwd, 'results', result_file),index=False)
    if model.status == 2:
        runTime = model.runTime
        mipGap = model.MIPGap

        dfResults = pd.DataFrame(columns=['Nombre', 'Tiempo de ejecución', 'MIP Gap %'])
        dfResults.loc[0] = [model.ModelName, runTime, mipGap]
        dfResults.to_csv(os.path.join(cwd, 'results', result_file), index=False)


def average_degree(G):
    nodes = G.nodes()
    ave_degree = 0
    for n in nodes:
        ave_degree += G.degree(n)
    return ave_degree/len(nodes)


def generate_random_accidents(months: int, M: list, S: list):
    mu_s = dict()

    for s in S:
        for m in M:
            current_rate = int(dataAccidentRates.loc[m]['Tasa'])
            current_accidents = int(dataAccidents.loc[m]['1'])
            for t in range(1, months+1):
                if t == 1:
                    mu_s[(s, t, m)] = current_accidents
                else:
                    mu_s[(s, t, m)] = np.random.poisson(lam=current_rate)

    return mu_s


def createModelDet_V2(modelname: str, filename: str, result_file: str):


def createModelSto_RandGraph(modelname: str, filename: str, result_file: str, mu_s: dict, S: list, connections: dict, M: list):
    R, T, Mr, delta, p, mu, V, k, g, h, I0 = readParametersDet(filename)

    b = 12 * 10 ** 3
    L = 49.7
    alpha = 0.50
    DW = 0.144
    gamma = 0.25

    model = gp.Model(modelname)

    w = dict()
    I = dict()
    f = dict()
    for t in T:
        w[t] = model.addVar(vtype=gp.GRB.BINARY, name=f'w_{t}')

    for s in S:
        for t in T:
            for m in M:
                I[(s, t, m)] = model.addVar(vtype=gp.GRB.INTEGER, name=f'I_{s}_{t}_{m}')
                f[(s, t, m)] = model.addVar(vtype=gp.GRB.INTEGER, name=f'f_{s}_{t}_{m}')
    y = {}
    x = {}
    for s in S:
        for t in T:
            for i in M:
                for j in M:
                    if connections[(i,j)]:
                        y[(s, t, i, j)] = model.addVar(vtype=gp.GRB.BINARY, name=f'y_{s}_{t}_{i}_{j}')
                        x[(s, t, i, j)] = model.addVar(vtype=gp.GRB.INTEGER, name=f'x_{s}_{t}_{i}_{j}')

    for s in S:
        for t in T[:-1]:
            for m in M:
                sent = gp.quicksum(x[(s, t, m, j)] for j in M if connections[(m,j)] == True)
                received = gp.quicksum(x[(s, t, i, m)] for i in M if connections[(i,m)] == True)
                if m == 11001:
                    model.addConstr(
                        I[s, t + 1, m] == I[s, t, m] - delta * mu_s[s, t, m] - sent + received + w[t] + f[s, t, m])
                else:
                    model.addConstr(
                        I[s, t + 1, m] == I[s, t, m] - delta * mu_s[s, t, m] - sent + received + f[s, t, m])

    for s in S:
        for m in M:
            model.addConstr(I[s,1,m] == I0[m])


    for t in T:
        if (t-1)%2 != 0:
            model.addConstr(w[t] == 0)

    for s in S:
        for t in T:
            for m in M:
                model.addConstr(I[s, t, m] <= V)
                model.addConstr(I[s, t, m] >= delta)

    for s in S:
        for t in T:
            for i in M:
                for j in M:
                    if connections[(i,j)]:
                        model.addConstr(y[s, t, i, j]*k >= x[s, t, i, j])

    cost_shipping = gp.quicksum(g*h[i, j]*y[s, t, i, j]*(1/len(S)) for s in S for t in T for i in M for j in M if connections[(i,j)] == True)
    cost_production = (gp.quicksum(p*w[t]*1/len(S) for t in T for s in S) + gp.quicksum(p*I[s, 1, m]*1/len(S) for s in S for m in M))
    cost_DALY = gp.quicksum(b*L*(alpha*f[s, t, m]*DW + gamma*f[s, t, m])*1/len(S) for m in M for t in T for s in S)


    model.setObjective(cost_production+cost_shipping+cost_DALY, gp.GRB.MINIMIZE)
    model.setParam('ResultFile', os.path.join(cwd, 'results', result_file))
    model.setParam('MIPGap', 0.01)
    model.update()

    return model


#def createModelSto(modelname: str, filename: str, result_file: str, mu_s: dict, S: list):
    R, T, Mr, delta, p, mu, V, k, g, h, I0 = readParametersDet(filename)

    b = 12*10**3
    L = 49.7
    alpha = 0.50
    DW = 0.144
    gamma = 0.25

    model = gp.Model(modelname)

    w = dict()
    I = dict()
    f = dict()

    for t in T:
        w[t] = model.addVar(vtype=gp.GRB.BINARY, name=f'w_{t}')


    for s in S:
        for t in T:
            for r in R:
                for m in Mr[r]:
                    if not (s, t, m) in list(I.keys()):
                        I[(s, t, m)] = model.addVar(vtype=gp.GRB.INTEGER, name=f'I_{s}_{t}_{m}')
                        f[(s, t, m)] = model.addVar(vtype=gp.GRB.INTEGER, name=f'f_{s}_{t}_{m}')


    y = {}
    x = {}
    for s in S:
        for t in T:
            for r in R:
                for i in Mr[r]:
                    for j in Mr[r]:
                        if i != j and (s, t, i, j) not in y.keys():
                            y[(s, t, i, j)] = model.addVar(vtype=gp.GRB.BINARY, name=f'y_{s}_{t}_{i}_{j}')
                            x[(s, t, i, j)] = model.addVar(vtype=gp.GRB.INTEGER, name=f'x_{s}_{t}_{i}_{j}')


    for s in S:
        for t in T[:-1]:
            for r in R:
                for m in Mr[r]:
                    sent = gp.quicksum(x[(s, t, m, j)] for j in Mr[r] if m != j)
                    received = gp.quicksum(x[(s, t, i, m)] for i in Mr[r] if i != m)
                    if m == 11001:
                        model.addConstr(I[s, t+1, m] == I[s, t, m] - delta*mu_s[s, t, m] - sent + received + w[t] + f[s, t, m])
                    else:
                        model.addConstr(I[s, t+1, m] == I[s, t, m] - delta*mu_s[s, t, m] - sent + received + f[s, t, m])

    for s in S:
        for r in R:
            for m in Mr[r]:
                model.addConstr(I[s, 1, m] == I0[m])

    for t in T:
        if (t-1)%2 != 0:
            model.addConstr(w[t] == 0)

    for s in S:
        for t in T:
            for r in R:
                for m in Mr[r]:
                    model.addConstr(I[s, t, m] <= V)
                    model.addConstr(I[s, t, m] >= delta)

    for s in S:
        for t in T:
            for r in R:
                for i in Mr[r]:
                    for j in Mr[r]:
                        if i != j:
                            model.addConstr(y[s, t, i, j]*k >= x[s, t, i, j])

    cost_shipping = gp.quicksum(g*h[i, j]*y[s, t, i, j]*(1/len(S)) for t in T for r in R for i in Mr[r] for j in Mr[r] if i != j for s in S)
    cost_production = (gp.quicksum(p*w[t]*1/len(S) for t in T for s in S) + gp.quicksum(p*I[s, 1, m]*1/len(S) for s in S for r in R for m in Mr[r]))
    cost_DALY = gp.quicksum(b*L*(alpha*f[s, t, m]*DW + gamma*f[s, t, m])*1/len(S) for r in R for m in Mr[r] for t in T for s in S)


    model.setObjective(cost_production+cost_shipping+cost_DALY, gp.GRB.MINIMIZE)
    model.setParam('ResultFile', os.path.join(cwd, 'results', result_file))
    model.setParam('MIPGap', 0.01)
    model.update()

    return model


def trainSAA(num_pool: int, size_pool: int, M: list, months: int, modelname: str, filename: str):

    for ind in range(1, num_pool):
        S_temp = list(range(1, size_pool))
        mu_s_temp = generate_random_accidents(months, M, S_temp)
        #modelTemp = createModelSto(f'{modelname}_{ind}', filename, mu_s_temp, )


##
#Creación de redes, aleatoria, jerárquica y GMM
M_test1 = M_prueba.copy()
M_test2 = M_prueba.copy()
M_test3 = M_prueba.copy()

#R_rand, Mr_rand = generateClustersRandom(M_test1, 31)
G_rand10, connections_rand10 = generateRandomGraph(M_test1, 10)
G_rand20, connections_rand20 = generateRandomGraph(M_test1, 20)
G_rand30, connections_rand30 = generateRandomGraph(M_test1, 30)
R_hier, Mr_hier = generateClustersHierarchical(M_test2)
R_GMM, Mr_GMM = generateClustersGMM(M_test3, BICGraph=True, RegionsGraph=False)

#G_rand = createGraph(R_rand, Mr_rand)
G_hier = createGraph(R_hier, Mr_hier)
G_GMM = createGraph(R_GMM, Mr_GMM)
##

list_G = [G_rand10, G_rand20, G_rand30, G_hier, G_GMM]
names = ['Rand_10', 'Rand_20', 'Rand_30', 'Hier', 'GMM']
for i, g_temp in enumerate(list_G):
    list_NumEdges = []
    for node in g_temp.nodes(data=True):
        #print(node[0])
        list_NumEdges.append(len(g_temp.edges(node[0])))
    plt.figure()
    plt.title(f'Histograma - {names[i]}')
    plt.hist(list_NumEdges)
    plt.show()



##

tableGraphs = PrettyTable(['Type', 'Edges', 'Ave. Degree', 'Diameter', 'Radius'])
tableGraphs.add_row(['Rand_10d', G_rand10.size(), np.round(average_degree(G_rand10),3), nx.diameter(G_rand10), nx.radius(G_rand10)])
tableGraphs.add_row(['Rand_20d', G_rand20.size(), np.round(average_degree(G_rand20),3), nx.diameter(G_rand20), nx.radius(G_rand20)])
tableGraphs.add_row(['Rand_15d', G_rand30.size(), np.round(average_degree(G_rand30),3), nx.diameter(G_rand30), nx.radius(G_rand30)])
tableGraphs.add_row(['Hierarchical', G_hier.size(), np.round(average_degree(G_hier),3), nx.diameter(G_hier), nx.radius(G_hier)])
tableGraphs.add_row(['GMM', G_GMM.size(), np.round(average_degree(G_GMM),3), nx.diameter(G_GMM), nx.radius(G_GMM)])
print(tableGraphs)

##
#Writting parameters of Random Network
#write_R_M_T(R_rand, Mr_rand, T_test)
#write_h(R_rand, Mr_rand)
#write_mu_I0(R_rand, T_test, Mr_rand)
#joinParametersDet()

##
#Writting parameters of Hierarchical Network
#write_R_M_T(R_hier, Mr_hier, T_test)
#write_h(R_hier, Mr_hier)
#write_mu_I0(R_hier, T_test, Mr_hier)
#joinParametersDet()

##
#Writting parameters of GMM Network
#write_R_M_T(R_GMM, Mr_GMM, T_test)
#write_h(R_GMM, Mr_GMM)
#write_mu_I0(R_GMM, T_test, Mr_GMM)
#joinParametersDet()


##
#print("MODELO DETERMINÍSTICO - REDES ALEATORIOAS\n")
#model_det_randGraph = createModelDet('Deterministic_RandGraph', 'params_det_rand.txt', 'det_rand.sol')
#analyzeMIPModel(model_det_randGraph, 'pre_det_rand.csv')
#model_det_randGraph.optimize()
#analyzeMIPModel(model_det_randGraph, 'post_det_rand.csv')



#print("MODELO DETERMINÍSTICO - REDES JERÁRQUICAS\n")
#model_det_hierGraph = createModelDet('Deterministic_RandGraph', 'params_det_hier.txt', 'det_hier.sol')
#analyzeMIPModel(model_det_hierGraph, 'pre_det_hier.csv')
#model_det_hierGraph.optimize()
#analyzeMIPModel(model_det_hierGraph, 'post_det_hier.csv')



#for var in model_det_hierGraph.getVars():
    #nameTemp = var.VarName.split('_')
    #value = var.X
    #if nameTemp[0] == 'f' and value >0 and nameTemp[2] == '5250':
        #print(f'Hubo {value} faltantes en El Bagre')
    #elif nameTemp[0] == 'I' and nameTemp[2] == '5250':
        #print(f'Inventario en el Bagre {value} para el mes {nameTemp[1]}')
    #elif nameTemp[0] == 'y' and nameTemp[2] == '5250' and value > 0:
        #print(f'El Bagre envio {value}')
    #elif nameTemp[0] == 'x' and nameTemp[3] == '5250' and value > 0:
        #print(f'Le envian a El Bagre {value} desde {nameTemp[2]}')



#print("MODELO DETERMINÍSTICO - REDES GMM\n")
#model_det_GMMGraph = createModelDet('Deterministic_RandGraph', 'params_det_GMM.txt', 'det_GMM.sol')
#analyzeMIPModel(model_det_GMMGraph, 'pre_det_GMM.csv')
#model_det_GMMGraph.optimize()
#analyzeMIPModel(model_det_GMMGraph, 'post_det_GMMM.csv')


##
#Conjunto de escenarios
np.random.seed(0)
S = list(range(1, 11))
mu_S = generate_random_accidents(6, M_prueba, S)


#print("MODELO ESTOCÁSTICO - REDES ALEATORIOAS\n")
#model_det_randGraph = createModelSto('Deterministic_RandGraph', 'params_det_rand.txt', 'sto30_rand.sol', mu_S, S)
#analyzeMIPModel(model_det_randGraph, 'pre_sto30_rand.csv')
#model_det_randGraph.optimize()
#analyzeMIPModel(model_det_randGraph, 'post_sto30_rand.csv')


print("MODELO ESTOCÁSTICO - REDES ALEATORIAS 10\n")
model_sto_hierGraph = createModelSto_RandGraph('Stochastic_Rand10Graph', 'params_det_hier.txt', 'sto10_rand10.sol'
                                               , mu_S, S, connections_rand10, M_test1)
#analyzeMIPModel(model_sto_hierGraph, 'pre_sto10_hier.csv')
model_sto_hierGraph.optimize()
analyzeMIPModel(model_sto_hierGraph, 'post_sto10_rand10.csv')


print("MODELO ESTOCÁSTICO - REDES ALEATORIAS 20\n")
model_sto_hierGraph = createModelSto_RandGraph('Stochastic_Rand20raph', 'params_det_hier.txt', 'sto10_rand20.sol'
                                               , mu_S, S, connections_rand20, M_test1)
#analyzeMIPModel(model_sto_hierGraph, 'pre_sto10_hier.csv')
model_sto_hierGraph.optimize()
analyzeMIPModel(model_sto_hierGraph, 'post_sto10_rand20.csv')


print("MODELO ESTOCÁSTICO - REDES ALEATORIAS 30\n")
model_sto_hierGraph = createModelSto_RandGraph('Stochastic_Rand30Graph', 'params_det_hier.txt', 'sto10_rand30.sol'
                                               , mu_S, S, connections_rand30, M_test1)
#analyzeMIPModel(model_sto_hierGraph, 'pre_sto10_hier.csv')
model_sto_hierGraph.optimize()
analyzeMIPModel(model_sto_hierGraph, 'post_sto10_rand30.csv')


print("MODELO ESTOCÁSTICO - REDES JERÁRQUICAS\n")
model_sto_hierGraph = createModelSto('Stochastic_GMMGraph', 'params_det_hier.txt', 'sto10_hier.sol', mu_S, S)
#analyzeMIPModel(model_sto_hierGraph, 'pre_sto30_hier.csv')
model_sto_hierGraph.optimize()
analyzeMIPModel(model_sto_hierGraph, 'post_sto10_hier.csv')


print("MODELO ESTOCÁSTICO - REDES GMM\n")
model_sto_GMMGraph = createModelSto('Stochastic_GMMGraph', 'params_det_GMM.txt', 'sto10_GMM.sol', mu_S, S)
#analyzeMIPModel(model_sto_GMMGraph, 'pre_sto30_GMM.csv')
model_sto_GMMGraph.optimize()
analyzeMIPModel(model_sto_GMMGraph, 'post_sto10_GMM.csv')


##
dfColor = pd.read_csv('ColoresDepartamento.csv', sep=';')

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


colorMapGMM = []
for node in G_GMM:
    for key in Mr_GMM.keys():
        listTemp = Mr_GMM[key]
        if node in listTemp:
            colorTemp = dfColor.loc[key]['Color']
            colorMapGMM.append(f'#{colorTemp}')
            break

sizeMap = [100]*len(dataCoordinates.index)

plt.figure(figsize=(5,7))
ax = plt.gca()
ax.set_title('Clusters Jerárquicos', fontweight='bold')
nx.draw(G_hier, pos=dictPositions, node_color=colorMapHier, node_size=sizeMap, ax=ax)
ax.axis('off')

plt.figure(figsize=(5,7))
ax = plt.gca()
ax.set_title('Clusters GMM', fontweight='bold')
nx.draw(G_GMM, pos=dictPositions, node_color=colorMapGMM, node_size=sizeMap, ax=ax)
ax.axis('off')

##
grafos = ['Jerárquicos', 'GMM']

plt.figure()
plt.bar(grafos, [model_det_hierGraph.ObjVal*0.3, model_det_GMMGraph.ObjVal*0.3], label='Distribución')
plt.bar(grafos, [model_det_hierGraph.ObjVal*0.7, model_det_GMMGraph.ObjVal*0.7],
        bottom=[model_det_hierGraph.ObjVal*0.3, model_det_GMMGraph.ObjVal*0.3], label='Producción')
plt.title('Costos Totales - Escenario Determinístico')
plt.legend(loc='best')
plt.show()

plt.figure()
plt.bar(grafos, [model_sto_hierGraph.ObjVal*0.3, model_sto_GMMGraph.ObjVal*0.3], label='Distribución')
plt.bar(grafos, [model_sto_hierGraph.ObjVal*0.6, model_sto_GMMGraph.ObjVal*0.65],
        bottom=[model_sto_hierGraph.ObjVal*0.3, model_sto_GMMGraph.ObjVal*0.3], label='Producción')
plt.bar(grafos, [model_sto_hierGraph.ObjVal*0.1, model_sto_GMMGraph.ObjVal*0.05],
        bottom=[model_sto_hierGraph.ObjVal*0.9, model_sto_GMMGraph.ObjVal*0.95], label='DALY')
plt.title('Costos Totales - Escenarios Estocásticos')
plt.legend(loc='best')
plt.show()


