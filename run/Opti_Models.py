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

##

cwd = os.getcwd()
dataCoordinates = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaCoordinates.csv'), index_col=0)
dataTime = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaTimeDistance.csv'), index_col=0)
dataAccidents = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaOA2019.csv'), index_col=0)
dataDepartments = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaDepartments.csv'), index_col=0)
dataAccidentRates = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaAccidentRates.csv'), index_col=0)


##
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
    I0 = dict() # Inventario inicial de los centros médicos
    q = dict()  # Conexiones entre municipios, 1: Si existe, 0: No existe
    DW = float() # Peso de la enfermedad
    alpha = float() # Probabilidad de MUERTE?
    gamma = float() # Probabilidad de COMORBILIDAD?
    b = float() # Costo de año de vida
    L = float() # Años de expectativa de vida

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
                    DW = float(line[startChar +1:endChar])
                elif line.find('alpha:=') != -1:
                    startChar = line.find('=')
                    endChar = line.find('//')
                    alpha = float(line[startChar+1:endChar])
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
                    keyTemp = line[startKey +1:endKey].split(',')
                    q[(int(keyTemp[0]), int(keyTemp[1]))] = int(line[startValue+1:endValue])

    txt.close()

    return M, T, delta, p, V, mu, k, g, h, I0, q, DW, alpha, gamma, b, L

def createModel_Det(param_file: str, model_name: str, result_file: str, mip_gap: float):
    M, T, delta, p, V, mu, k, g, h, I0, q, DW, alpha, gamma, b, L = readParametersDet(param_file)
    cwd = os.getcwd()

    model = gp.Model(model_name)

    w = dict()
    I = dict()
    y = dict()
    x = dict()
    f = dict()

    #Definition of the variable related to production
    for t in T:
        w[t] = model.addVar(vtype=gp.GRB.INTEGER, name=f'w_{t}')

    #Definition of variables related to inventory and lackings
    for t in T:
        for i in M:
            I[(t, i)] = model.addVar(vtype=gp.GRB.INTEGER, name=f'I_{t}_{i}')
            f[(t, i)] = model.addVar(vtype=gp.GRB.INTEGER, name=f'f_{t}_{i}')

    #Definition of variables related to shipment
    for t in T:
        for i in M:
            for j in M:
                if i != j and q[(i, j)] == 1:
                    y[(t, i, j)] = model.addVar(vtype=gp.GRB.BINARY, name=f'y_{t}_{i}_{j}')
                    x[(t, i, j)] = model.addVar(vtype=gp.GRB.INTEGER, name=f'x_{t}_{i}_{j}')

    #Restriction of capacity in the transport
    for t in T:
        for i in M:
            for j in M:
                if i != j and q[(i, j)] == 1:
                    model.addConstr(k*y[(t, i, j)] >= x[(t, i, j)])

    #Restriction of Inventory, max and min capacity
    for t in T:
        for i in M:
            if i != 11001:
                model.addConstr(I[(t, i)] <= V)
                model.addConstr(I[(t, i)] >= delta)

    #Restriction of production every odd month
    for t in T:
        if (t-1)%2 != 0:
            model.addConstr(w[t] == 0)

    #Restriction of next month inventory associated with the previous
    for t in T[:-1]:
        for i in M:
            sent = 0
            received = 0
            for j in M:
                if (t, i, j) in x.keys():
                    sent += x[(t, i, j)]
                if (t, j, i) in x.keys():
                    received += x[(t, j, i)]
            if i == 11001:
                model.addConstr(I[(t+1, i)] == I[(t, i)] + w[t] - delta*mu[(t, i)] - sent + received + f[(t, i)])
            else:
                model.addConstr(I[(t + 1, i)] == I[(t, i)] - delta * mu[(t, i)] - sent + received + f[(t, i)])

    for i in M:
        model.addConstr(I[(1, i)] == I0[i])

    #Cost of production of new vials
    c_production = gp.quicksum(p*w[t] for t in T)

    #Cost of shipment of vials between cities
    c_shipment = gp.quicksum(g*h[(i, j)]*y[(t, i, j)] for t in T for i in M for j in M if i != j and q[(i, j)] == 1)

    #Cost of DALY due to lackings in vials
    c_DALY = gp.quicksum(b*L*(DW*alpha*f[(t, i)]/delta + gamma*f[(t, i)]/delta) for t in T for i in M)

    model.setObjective(c_production + c_shipment + c_DALY, gp.GRB.MINIMIZE)
    model.setParam('ResultFile', os.path.join(cwd, 'results', result_file))
    model.setParam('MIPGap', mip_gap)
    model.setParam('OutputFlag', 0)
    model.update()

    return model

def createModel_Det_NoShortage(param_file: str, model_name: str, result_file: str, mip_gap: float):
    M, T, delta, p, V, mu, k, g, h, I0, q, DW, alpha, gamma, b, L = readParametersDet(param_file)
    cwd = os.getcwd()

    model = gp.Model(model_name)

    w = dict()
    I = dict()
    y = dict()
    x = dict()

    #Definition of the variable related to production
    for t in T:
        w[t] = model.addVar(vtype=gp.GRB.INTEGER, name=f'w_{t}')

    #Definition of variables related to inventory and lackings
    for t in T:
        for i in M:
            I[(t, i)] = model.addVar(vtype=gp.GRB.INTEGER, name=f'I_{t}_{i}')

    #Definition of variables related to shipment
    for t in T:
        for i in M:
            for j in M:
                if i != j and q[(i, j)] == 1:
                    y[(t, i, j)] = model.addVar(vtype=gp.GRB.BINARY, name=f'y_{t}_{i}_{j}')
                    x[(t, i, j)] = model.addVar(vtype=gp.GRB.INTEGER, name=f'x_{t}_{i}_{j}')

    #Restriction of capacity in the transport
    for t in T:
        for i in M:
            for j in M:
                if i != j and q[(i, j)] == 1:
                    model.addConstr(k*y[(t, i, j)] >= x[(t, i, j)])

    #Restriction of Inventory, max and min capacity
    for t in T:
        for i in M:
            if i != 11001:
                model.addConstr(I[(t, i)] <= V)
                model.addConstr(I[(t, i)] >= delta)

    #Restriction of production every odd month
    for t in T:
        if (t-1)%2 != 0:
            model.addConstr(w[t] == 0)

    #Restriction of next month inventory associated with the previous
    for t in T[:-1]:
        for i in M:
            sent = 0
            received = 0
            for j in M:
                if (t, i, j) in x.keys():
                    sent += x[(t, i, j)]
                if (t, j, i) in x.keys():
                    received += x[(t, j, i)]
            if i == 11001:
                model.addConstr(I[(t+1, i)] == I[(t, i)] + w[t] - delta*mu[(t, i)] - sent + received)
            else:
                model.addConstr(I[(t + 1, i)] == I[(t, i)] - delta * mu[(t, i)] - sent + received)

    for i in M:
        model.addConstr(I[(1, i)] == I0[i])

    #Cost of production of new vials
    c_production = gp.quicksum(p*w[t] for t in T)

    #Cost of shipment of vials between cities
    c_shipment = gp.quicksum(g*h[(i, j)]*y[(t, i, j)] for t in T for i in M for j in M if i != j and q[(i, j)] == 1)


    model.setObjective(c_production + c_shipment, gp.GRB.MINIMIZE)
    model.setParam('ResultFile', os.path.join(cwd, 'results', result_file))
    model.setParam('MIPGap', mip_gap)
    model.setParam('OutputFlag', 0)
    model.update()

    return model

def createModel_Sto(param_file: str, model_name: str, result_file: str, S: list, mu_s: dict, mip_gap: float):
    M, T, delta, p, V, mu, k, g, h, I0, q, DW, alpha, gamma, b, L = readParametersDet(param_file)
    cwd = os.getcwd()

    model = gp.Model(model_name)

    w = dict()
    I = dict()
    y = dict()
    x = dict()
    f = dict()

    # Definition of the variable related to production
    for t in T:
        w[t] = model.addVar(vtype=gp.GRB.INTEGER, name=f'w_{t}')

    # Definition of variables related to inventory and lackings
    for s in S:
        for t in T:
            for i in M:
                I[(s, t, i)] = model.addVar(vtype=gp.GRB.INTEGER, name=f'I_{s}_{t}_{i}')
                f[(s, t, i)] = model.addVar(vtype=gp.GRB.INTEGER, name=f'f_{s}_{t}_{i}')

    # Definition of variables related to shipment
    for s in S:
        for t in T:
            for i in M:
                for j in M:
                    if i != j and q[(i, j)] == 1:
                        y[(s, t, i, j)] = model.addVar(vtype=gp.GRB.BINARY, name=f'y_{s}_{t}_{i}_{j}')
                        x[(s, t, i, j)] = model.addVar(vtype=gp.GRB.INTEGER, name=f'x_{s}_{t}_{i}_{j}')

    # Restriction of capacity in the transport
    for s in S:
        for t in T:
            for i in M:
                for j in M:
                    if i != j and q[(i, j)] == 1:
                        model.addConstr(k * y[(s, t, i, j)] >= x[(s, t, i, j)])

    # Restriction of Inventory, max and min capacity
    for s in S:
        for t in T:
            for i in M:
                if i != 11001:
                    model.addConstr(I[(s, t, i)] <= V)
                    model.addConstr(I[(s, t, i)] >= delta)

    # Restriction of production every odd month
    for t in T:
        if (t - 1) % 2 != 0:
            model.addConstr(w[t] == 0)

    # Restriction of next month inventory associated with the previous
    for s in S:
        for t in T[:-1]:
            for i in M:
                sent = 0
                received = 0
                for j in M:
                    if (s, t, i, j) in x.keys():
                        sent += x[(s, t, i, j)]
                    if (s, t, j, i) in x.keys():
                        received += x[(s, t, j, i)]
                if i == 11001:
                    model.addConstr(I[(s, t + 1, i)] == I[(s, t, i)] + w[t] - delta * mu_s[(s, t, i)] - sent + received + f[(s, t, i)])
                else:
                    model.addConstr(I[(s, t + 1, i)] == I[(s, t, i)] - delta * mu_s[(s, t, i)] - sent + received + f[(s, t, i)])

    for s in S:
        for i in M:
            model.addConstr(I[(s, 1, i)] == I0[i])

    # Cost of production of new vials
    c_production = gp.quicksum(p * w[t] for t in T)

    # Cost of shipment of vials between cities
    c_shipment = gp.quicksum((g * h[(i, j)] * y[(s, t, i, j)])*1/len(S) for s in S for t in T for i in M for j in M if i != j and q[(i, j)] == 1)

    # Cost of DALY due to lackings in vials
    c_DALY = gp.quicksum((b * L * (DW * alpha * f[(s, t, i)]/delta + gamma * f[(s, t, i)]/delta))*1/len(S) for s in S for t in T for i in M)

    model.setObjective(c_production + c_shipment + c_DALY, gp.GRB.MINIMIZE)
    model.setParam('ResultFile', os.path.join(cwd, 'results', result_file))
    model.setParam('MIPGap', mip_gap)
    model.setParam('OutputFlag', 0)
    model.update()

    return model

def createModel_Sto_NoShortage(param_file: str, model_name: str, result_file: str, S: list, mu_s: dict, mip_gap: float):
    M, T, delta, p, V, mu, k, g, h, I0, q, DW, alpha, gamma, b, L = readParametersDet(param_file)
    cwd = os.getcwd()

    model = gp.Model(model_name)

    w = dict()
    I = dict()
    y = dict()
    x = dict()

    # Definition of the variable related to production
    for t in T:
        w[t] = model.addVar(vtype=gp.GRB.INTEGER, name=f'w_{t}')

    # Definition of variables related to inventory and lackings
    for s in S:
        for t in T:
            for i in M:
                I[(s, t, i)] = model.addVar(vtype=gp.GRB.INTEGER, name=f'I_{s}_{t}_{i}')

    # Definition of variables related to shipment
    for s in S:
        for t in T:
            for i in M:
                for j in M:
                    if i != j and q[(i, j)] == 1:
                        y[(s, t, i, j)] = model.addVar(vtype=gp.GRB.BINARY, name=f'y_{s}_{t}_{i}_{j}')
                        x[(s, t, i, j)] = model.addVar(vtype=gp.GRB.INTEGER, name=f'x_{s}_{t}_{i}_{j}')

    # Restriction of capacity in the transport
    for s in S:
        for t in T:
            for i in M:
                for j in M:
                    if i != j and q[(i, j)] == 1:
                        model.addConstr(k * y[(s, t, i, j)] >= x[(s, t, i, j)])

    # Restriction of Inventory, max and min capacity
    for s in S:
        for t in T:
            for i in M:
                if i != 11001:
                    model.addConstr(I[(s, t, i)] <= V)
                    model.addConstr(I[(s, t, i)] >= delta)

    # Restriction of production every odd month
    for t in T:
        if (t - 1) % 2 != 0:
            model.addConstr(w[t] == 0)

    # Restriction of next month inventory associated with the previous
    for s in S:
        for t in T[:-1]:
            for i in M:
                sent = 0
                received = 0
                for j in M:
                    if (s, t, i, j) in x.keys():
                        sent += x[(s, t, i, j)]
                    if (s, t, j, i) in x.keys():
                        received += x[(s, t, j, i)]
                if i == 11001:
                    model.addConstr(I[(s, t + 1, i)] == I[(s, t, i)] + w[t] - delta * mu_s[(s, t, i)] - sent + received)
                else:
                    model.addConstr(I[(s, t + 1, i)] == I[(s, t, i)] - delta * mu_s[(s, t, i)] - sent + received)

    for s in S:
        for i in M:
            model.addConstr(I[(s, 1, i)] == I0[i])

    # Cost of production of new vials
    c_production = gp.quicksum(p * w[t] for t in T)

    # Cost of shipment of vials between cities
    c_shipment = gp.quicksum((g * h[(i, j)] * y[(s, t, i, j)])*1/len(S) for s in S for t in T for i in M for j in M if i != j and q[(i, j)] == 1)

    model.setObjective(c_production + c_shipment, gp.GRB.MINIMIZE)
    model.setParam('ResultFile', os.path.join(cwd, 'results', result_file))
    model.setParam('MIPGap', mip_gap)
    model.setParam('OutputFlag', 0)
    model.update()

    return model

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

##
cwd = os.getcwd()

#graph_names = ['randD16','randD20', 'randD26', 'randD30', 'hier', 'GMM', 'GMMCapitals']
graph_names = ['hier', 'GMMBridges']

##
optimizationTimes_Det = ['Deterministic']

for graph_temp in graph_names:
    currentTime = time.time()
    modelTemp = createModel_Det(f'params_det_{graph_temp}.txt', f'Det_{graph_temp}', f'det_{graph_temp}.sol', 0.05)
    modelTemp.optimize()
    modelTemp.write(os.path.join(cwd, 'models', f'det_{graph_temp}.lp'))
    optimizationTimes_Det.append(time.time() - currentTime)
    if modelTemp.status == 2:
        print(f'\nOptimizado el modelo DETERMINÍSTICO de {graph_temp}!! \n')
    elif modelTemp.status == 3:
        print(f'\nEl modelo DETERMINÍSTICO de {graph_temp} es infactible!! \n')

    # modelTemp = createModel_Det_NoShortage(f'params_det_{graph_temp}.txt', f'Det_{graph_temp}', f'detNS_{graph_temp}.sol', 0.05)
    # modelTemp.optimize()
    # modelTemp.write(os.path.join(cwd, 'models', f'det_{graph_temp}.lp'))
    # if modelTemp.status == 2:
    #     print(f'\nOptimizado el modelo DETERMINÍSTICO SIN FALTANTES de {graph_temp}!! \n')
    # elif modelTemp.status == 3:
    #     print(f'\nEl modelo DETERMINÍSTICO SIN FALTANTES de {graph_temp} es infactible!! \n')


##
M, T, delta, p, V, mu, k, g, h, I0, q, DW, alpha, gamma, b, L = readParametersDet(f'params_det_randD10.txt')

S = [1, 2, 3, 4, 5, 6]
mu_S = generate_random_accidents(6, M, S)

optimizationTimes_Sto = ['Stochastic']

for graph_temp in graph_names:
    currentTime = time.time()
    modelTemp = createModel_Sto(f'params_det_{graph_temp}.txt', f'Sto6_{graph_temp}', f'sto6_{graph_temp}.sol', S, mu_S, 0.05)
    modelTemp.write(os.path.join(cwd, 'models', f'sto6_{graph_temp}.lp'))
    modelTemp.optimize()
    optimizationTimes_Sto.append(time.time() - currentTime)
    if modelTemp.status == 2:
        print(f'\nOptimizado el modelo ESTOCÁSTICO de {graph_temp}!! \n')
    elif modelTemp.status == 3:
        print(f'\nEl modelo ESTOCÁSTICO de {graph_temp} es infactible!! \n')




##
# header = ['Model'] + graph_names
#
#
# with open(os.path.join(cwd, 'results', 'running_time.csv'), 'w') as file:
#     writer = csv.writer(file)
#     writer.writerow(header)
#     writer.writerow(optimizationTimes_Det)
#     writer.writerow(optimizationTimes_Sto)




