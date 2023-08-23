##
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import csv
import matplotlib.patches as mpatches


##

cwd = os.getcwd()
dataCoordinates = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaCoordinates.csv'), index_col=0)
dataTime = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaTimeDistance.csv'), index_col=0)
dataAccidents = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaOA2019.csv'), index_col=0)
dataDepartments = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaDepartments.csv'), index_col=0)
dataAccidentRates = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaAccidentRates.csv'), index_col=0)

arr_Time = dataTime.to_numpy()

T = [1, 2, 3, 4, 5, 6]
S = [1, 2, 3, 4, 5, 6]
list_degree = [10, 16, 20, 26, 30]
M = list()
listMunipPruebas = list(dataCoordinates['Departamento'])
cwd = os.getcwd()

for i, munip in enumerate(dataCoordinates['Departamento']):
    if munip in listMunipPruebas:
        M.append(list(dataCoordinates.index)[i])

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

M, T, delta, p, V, mu, k, g, h, I0, q, DW, alpha, gamma, b, L = readParametersDet(
    os.path.join(cwd, 'parameters', 'params_det_hier.txt'))

#Functions for deterministic results
def readResults(list_path: list, deterministic: bool):
    dict_f = dict()
    dict_I = dict()
    dict_y = dict()
    dict_x = dict()
    dict_w = dict()
    if deterministic:
        for path in list_path:
            arr_f = np.zeros((len(T), len(M)))
            arr_I = np.zeros((len(T), len(M)))
            arr_y = np.zeros((len(T), len(M), len(M)))
            arr_x = np.zeros((len(T), len(M), len(M)))
            arr_w = np.zeros((len(T)))

            with open(os.path.join(cwd, 'results', path), newline='\n') as csvfile:
                reader = csv.reader((line.replace('  ', ' ') for line in csvfile), delimiter=' ')
                next(reader)
                for line in reader:
                    var = line[0].split('_')
                    var_name = var[0]
                    if var_name == 'f':
                        indT = T.index(int(var[1]))
                        indM = M.index(int(var[2]))
                        arr_f[indT, indM] = float(line[1])
                    elif var_name == 'I':
                        indT = T.index(int(var[1]))
                        indM = M.index(int(var[2]))
                        arr_I[indT, indM] = float(line[1])
                    elif var_name == 'y':
                        indT = T.index(int(var[1]))
                        indM1 = M.index(int(var[2]))
                        indM2 = M.index(int(var[3]))
                        arr_y[indT, indM1, indM2] = float(line[1])
                    elif var_name == 'x':
                        indT = T.index(int(var[1]))
                        indM1 = M.index(int(var[2]))
                        indM2 = M.index(int(var[3]))
                        arr_x[indT, indM1, indM2] = float(line[1])
                    elif var_name == 'w':
                        indT = T.index(int(var[1]))
                        arr_w[indT] = float(line[1])
            csvfile.close()

            dict_f[f'{path}'] = arr_f
            dict_I[f'{path}'] = arr_I
            dict_y[f'{path}'] = arr_y
            dict_x[f'{path}'] = arr_x
            dict_w[f'{path}'] = arr_w
    else:
        for path in list_path:
                arr_f = np.zeros((len(S), len(T), len(M)))
                arr_I = np.zeros((len(S), len(T), len(M)))
                arr_y = np.zeros((len(S), len(T), len(M), len(M)))
                arr_x = np.zeros((len(S), len(T), len(M), len(M)))
                arr_w = np.zeros((len(T)))

                with open(os.path.join(cwd, 'results', path), newline='\n') as csvfile:
                    reader = csv.reader((line.replace('  ', ' ') for line in csvfile), delimiter=' ')
                    next(reader)
                    for line in reader:
                        var = line[0].split('_')
                        var_name = var[0]
                        if var_name == 'f':
                            indS = S.index(int(var[1]))
                            indT = T.index(int(var[2]))
                            indM = M.index(int(var[3]))
                            arr_f[indS, indT, indM] = float(line[1])
                        elif var_name == 'I':
                            indS = S.index(int(var[1]))
                            indT = T.index(int(var[2]))
                            indM = M.index(int(var[3]))
                            arr_I[indS, indT, indM] = float(line[1])
                        elif var_name == 'y':
                            indS = S.index(int(var[1]))
                            indT = T.index(int(var[2]))
                            indM1 = M.index(int(var[3]))
                            indM2 = M.index(int(var[4]))
                            arr_y[indS, indT, indM1, indM2] = float(line[1])
                        elif var_name == 'x':
                            indS = S.index(int(var[1]))
                            indT = T.index(int(var[2]))
                            indM1 = M.index(int(var[3]))
                            indM2 = M.index(int(var[4]))
                            arr_x[indS, indT, indM1, indM2] = float(line[1])
                        elif var_name == 'w':
                            indT = T.index(int(var[1]))
                            arr_w[indT] = float(line[1])

                dict_f[f'{path}'] = arr_f
                dict_I[f'{path}'] = arr_I
                dict_y[f'{path}'] = arr_y
                dict_x[f'{path}'] = arr_x
                dict_w[f'{path}'] = arr_w

    return dict_f, dict_I, dict_y, dict_x, dict_w


def graph_BarProduction(dict_w: dict, deterministic: bool):
    if deterministic:
        for (key, value) in dict_w.items():
            plt.figure()
            months = list(range(1,len(value)+1))
            plt.bar(months, value, color='darkgreen')
            plt.xlabel('Month')
            plt.ylabel('Vials')
            plt.title(f'Production of antiophidic vials in Bogotá D.C. \n ({key[:-4]})')
            plt.savefig(os.path.join(cwd, 'plots', f'{key[:-4]}_Bar_Production'))
            plt.show()
            plt.draw()
    else:
        for (key, value) in dict_w.items():
            plt.figure()
            months = list(range(1,len(value)+1))
            plt.bar(months, value, color='darkgreen')
            plt.xlabel('Month')
            plt.ylabel('Vials')
            plt.title(f'Production of antiophidic vials in Bogotá D.C. \n ({key[:-4]})')
            plt.savefig(os.path.join(cwd, 'plots', f'{key[:-4]}_Bar_Production'))
            plt.show()
            plt.draw()


def graph_HistShortage(dict_f: dict, deterministic: bool):
    if deterministic:
        for (key, value) in dict_f.items():
            plt.figure()
            plt.hist(value.flatten(), color='darkred')
            plt.ylabel('Frequency')
            plt.xlabel('Vials')
            plt.title(f'Histogram of shortages of vials in Colombia \n({key[:-4]})')
            plt.savefig(os.path.join(cwd, 'plots', f'{key[:-4]}_Hist_Shortage'))
            plt.show()
            plt.draw()
    else:
        for (key, value) in dict_f.items():
            plt.figure()
            average_value = np.mean(value, axis=0)
            plt.hist(average_value.flatten(), color='darkred')
            plt.ylabel('Frequency')
            plt.xlabel('Vials')
            plt.title(f'Histogram of average shortages of vials in Colombia \n({key[:-4]})')
            plt.savefig(os.path.join(cwd, 'plots', f'{key[:-4]}_Hist_Shortage'))
            plt.show()
            plt.draw()


def graph_HistShipments(dict_y: dict, deterministic: bool):
    if deterministic:
        for (key, value) in dict_y.items():
            histArray = np.empty(0)
            for indT in range(value.shape[0]):
                timeArray_temp = np.multiply(arr_Time, value[indT, :, :])
                histArray = np.concatenate((histArray, timeArray_temp.flatten()))

            fig, axs = plt.subplots(1,2, figsize=(12, 6))

            axs[0].hist(histArray, bins=80, color='indigo')
            axs[0].set_ylabel('Frequency')
            axs[0].set_xlabel('Time [hour]')
            axs[0].set_title(f'Histogram of Travelling Time \n({key[:-4]})')

            axs[1].hist(histArray[np.where(histArray>3.5)], bins=80, color='indigo')
            axs[1].set_ylabel('Frequency')
            axs[1].set_xlabel('Time [hour]')
            axs[1].set_title(f'Histogram of Travelling Time > 3.5 h \n({key[:-4]})')

            plt.savefig(os.path.join(cwd, 'plots', f'{key[:-4]}_Hist_TravelTime'))
            plt.show()
            plt.draw()
    else:
        for (key, value) in dict_y.items():
            histArray = np.empty(0)
            value = np.mean(value, axis=0)
            for indT in range(value.shape[0]):
                timeArray_temp = np.multiply(arr_Time, value[indT, :, :])
                histArray = np.concatenate((histArray, timeArray_temp.flatten()))

            fig, axs = plt.subplots(1, 2, figsize=(12, 6))

            axs[0].hist(histArray, bins=80, color='indigo')
            axs[0].set_ylabel('Frequency')
            axs[0].set_xlabel('Time [hour]')
            axs[0].set_title(f'Histogram of Travelling Time \n({key[:-4]})')

            axs[1].hist(histArray[np.where(histArray>3.5)], bins=80, color='indigo')
            axs[1].set_ylabel('Frequency')
            axs[1].set_xlabel('Time [hour]')
            axs[1].set_title(f'Histogram of Travelling Time > 3.5 h \n({key[:-4]})')

            plt.savefig(os.path.join(cwd, 'plots', f'{key[:-4]}_Hist_TravelTime'))
            plt.show()
            plt.draw()


def writeResults(dict_y: dict, dict_f: dict, dict_w: dict, deterministic: bool):
    list_CostShipments = ['Shipment cost']
    list_CostShortage = ['Shortage cost']
    list_CostProduction = ['Production cost']
    list_TotalCost = ['Total cost']
    list_TotalProduction = ['Total Production']
    list_TotalShipments = ['Total shipments']
    list_TotalTravelTime = ['Total travelling time']
    list_TotalShortage = ['Total Shortage']

    if deterministic:
        for (key, value) in dict_w.items():
            production_temp = np.sum(value)
            list_TotalProduction.append(production_temp)
            list_CostProduction.append(int(p*production_temp))

        for (key, value) in dict_f.items():
            shortage_temp = np.sum(value)
            list_TotalShortage.append(shortage_temp)
            list_CostShortage.append(int(b*L*(DW*alpha*shortage_temp + gamma*shortage_temp)))

        for (key, value) in dict_y.items():
            shipments_temp = np.sum(value)
            travelTime_temp = 0
            for indT in range(value.shape[0]):
                travelTime_temp += np.sum(np.multiply(value[indT,: , :], arr_Time))

            list_TotalShipments.append(int(shipments_temp))
            list_TotalTravelTime.append(np.round(travelTime_temp,3))
            list_CostShipments.append(int(g*travelTime_temp))

        for i in range(1, len(list_CostShipments)):
            totalCost_temp = list_CostShipments[i] + list_CostShortage[i] + list_CostProduction[i]
            list_TotalCost.append(totalCost_temp)


    else:
        for (key, value) in dict_w.items():
            production_temp = np.sum(value)
            list_TotalProduction.append(production_temp)
            list_CostProduction.append(p * production_temp)

        for (key, value) in dict_f.items():
            shortage_temp = np.sum(np.mean(value, axis=0))
            list_TotalShortage.append(shortage_temp)
            list_CostShortage.append(b * L * (DW * alpha * shortage_temp + gamma * shortage_temp))

        for (key, value) in dict_y.items():
            shipments_temp = np.sum(np.mean(value, axis=0))
            travelTime_temp = 0
            for indT in range(value.shape[0]):
                travelTime_temp += np.sum(np.multiply(value[indT, :, :], arr_Time))

            list_TotalShipments.append(shipments_temp)
            list_TotalTravelTime.append(travelTime_temp)
            list_CostShipments.append(g * travelTime_temp)

        for i in range(1, len(list_CostShipments)):
            totalCost_temp = list_CostShipments[i] + list_CostShortage[i] + list_CostProduction[i]
            list_TotalCost.append(totalCost_temp)

    if deterministic:
        plot_name = 'Deterministic'
        list_x = [i[4:-4] for i in list(dict_w.keys())]
    else:
        plot_name = 'Stochastic'
        list_x = [i[5:-4] for i in list(dict_w.keys())]

    header = ['Metrics'] + list(dict_w.keys())
    dataCSV = [list_CostShipments, list_CostShortage, list_CostProduction, list_TotalCost, list_TotalProduction,
               list_TotalShipments, list_TotalTravelTime, list_TotalShortage]
    with open(os.path.join(cwd, 'results', f'results_{plot_name}.csv'), 'w') as file:
        writer = csv.writer(file)

        writer.writerow(header)

        writer.writerows(dataCSV)


    fig, axs = plt.subplots(1, 2, figsize=(14,6))

    axs[0].bar(list_x, list_CostShipments[1:], color='indigo', label='Shipment')
    axs[0].bar(list_x, list_CostProduction[1:], bottom=list_CostShipments[1:], color='darkgreen', label='Production')
    axs[0].set_xlabel('Networks')
    axs[0].set_title('Cost of production & shipment of vials')
    axs[0].set_ylabel('Thousands of COP ($)')

    axs[1].bar(list_x, list_CostShortage[1:], color='darkred')
    axs[1].set_title('Cost of shortages of vials')
    axs[1].set_xlabel('Networks')
    axs[1].set_ylabel('Thousands of COP ($)')

    plt.suptitle(f'{plot_name} Optimization')
    plt.savefig(os.path.join(cwd, 'plots', f'{plot_name}_Bar_Cost'))
    plt.show()
    plt.draw()
    plt.close()


##

list_degree = [10, 16, 20, 26, 30]


#list_graphsDet = ['det_randD16.sol', 'det_randD20.sol', 'det_randD26.sol',
                  #'det_hier.sol', 'det_GMM.sol', 'det_GMMCapitals.sol', 'det_GMMBridges.sol']

list_graphsDet = ['det_hier.sol', 'det_GMM.sol', 'det_GMMCapitals.sol', 'det_GMMBridges.sol']

list_graphsSto = ['sto6_hier.sol','sto6_GMM.sol', 'sto6_GMMCapitals.sol', 'sto6_GMMBridges.sol']

#list_graphsSto = ['sto6_randD16.sol', 'sto6_randD20.sol', 'sto6_randD26.sol',
                  #'sto6_hier.sol','sto6_GMM.sol', 'sto6_GMMCapitals.sol', 'sto6_GMMBridges.sol']


dictDet_f, dictDet_I, dictDet_y, dictDet_x, dictDet_w = readResults(list_graphsDet, True)

dictSto_f, dictSto_I, dictSto_y, dictSto_x, dictSto_w = readResults(list_graphsSto, False)


##
#Plotting and printing the results for the deterministic optimization
writeResults(dictDet_y, dictDet_f, dictDet_w, True)

#graph_BarProduction(dictDet_w, True)

#graph_HistShortage(dictDet_f, True)

#graph_HistShipments(dictDet_y, True)


##
#Plotting and printing the results for the stochastic optimization
writeResults(dictSto_y, dictSto_f, dictSto_w, False)
#
#graph_BarProduction(dictSto_w, False)
#
#graph_HistShortage(dictSto_f, False)
#
#graph_HistShipments(dictSto_y, False)



##
labels1 = []

def add_label(violin, label):
    color = violin["bodies"][0].get_facecolor().flatten()
    labels1.append((mpatches.Patch(color=color), label))

string_graph = ['hier', 'GMMBridges']
list_histDet = []

for str in string_graph:
    histArray = np.empty(0)
    value = dictDet_y[f'det_{str}.sol']
    for indT in range(value.shape[0]):
        timeArray_temp = np.multiply(arr_Time, value[indT, :, :])
        histArray = np.concatenate((histArray, timeArray_temp.flatten()))
    list_histDet.append(histArray)

plt.style.use('seaborn-deep')


x1 = list_histDet[0][np.where(list_histDet[0] > 4)]
y1 = list_histDet[1][np.where(list_histDet[1] > 4)]

fig, ax = plt.subplots(figsize=(7,6))
add_label(plt.violinplot(x1), 'Hier')
add_label(plt.violinplot(y1, positions=[1.7]), 'GMM+Bridges')
plt.ylabel('# de Envíos con una duración mayor a 4 horas')
ax.set_xticks([1, 1.7])
ax.set_xticklabels(['Jerárquico', 'GMM+Puentes'])

plt.show()
plt.title('Caso Determinístico', fontweight='bold')


labels2 = []

list_histSto = []

def add_label(violin, label):
    color = violin["bodies"][0].get_facecolor().flatten()
    labels2.append((mpatches.Patch(color=color), label))


for str in string_graph:
    histArray = np.empty(0)
    value = np.mean(dictSto_y[f'sto6_{str}.sol'], axis=0)
    for indT in range(value.shape[0]):
        timeArray_temp = np.multiply(arr_Time, value[indT, :, :])
        histArray = np.concatenate((histArray, timeArray_temp.flatten()))
    list_histSto.append(histArray)


x2 = list_histSto[0][np.where(list_histSto[0] > 4)]
y2 = list_histSto[1][np.where(list_histSto[1] > 4)]

plt.style.use('seaborn-deep')


fig, ax = plt.subplots(figsize=(7,6))
add_label(plt.violinplot(x2), 'Hier')
add_label(plt.violinplot(y2, positions=[1.7]), 'GMM+Bridges')
plt.ylabel('# de Envíos con una duración mayor a 4 horas')
ax.set_xticks([1, 1.7])
ax.set_xticklabels(['Jerárquico', 'GMM+Puentes'])
plt.show()
plt.title('Caso Estocástico (DE)', fontweight='bold')

##

