##
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd
from scipy.stats import poisson, linregress

cwd = os.getcwd()
dataCoordinates = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaCoordinates.csv'), index_col=0)
dataTime = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaTimeDistance.csv'), index_col=0)
dataAccidents = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaOA2019.csv'), index_col=0)
dataDepartments = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaDepartments.csv'), index_col=0)

dataRateAccidents = (dataAccidents.iloc[:, 3:]).mean(axis=1)
dataStaDevAccidents = np.std(dataAccidents.iloc[:, 3:], axis=1)

M_test = [73148, 73001, 41872, 73168, 73268, 73275, 73043, 73283, 73349]
CD_test= 73001

GAMMA = 0.8
TOLERANCE = 0.001

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

def policy_evaluation(states, rewards, transition_probabilities, discount_factor, threshold, policy, value_function):
    converge = False
    while converge == False:
        delta = 0
        for s_current in states:
            temp_value = value_function[s_current]
            a_current = policy[s_current]
            new_value = rewards[s_current, a_current]
            for s_next in states:
                new_value += discount_factor*transition_probabilities[(s_current, a_current, s_next)]*value_function[s_next]
            value_function[s_current] = new_value
            delta = max(delta, np.abs(temp_value - new_value))

        if delta < threshold:
            converge = True

    return value_function

def policy_improvement(states, actions, transition_probabilities, policy, value_function):
    change = False
    for s_current in states:
        temp_policy = policy[s_current]
        max_value = value_function[s_current]
        max_a = policy[s_current]
        for a in actions[s_current]:
            temp_value = 0
            for s_next in states:
                temp_value += transition_probabilities[(s_current, a, s_next)]*value_function[s_next]

            if temp_value > max_value:
                max_value = temp_value
                max_a = a

        policy[s_current] = max_a
        if temp_policy != policy[s_current]:
            change = True
    return policy, change

def policy_iteration(states, actions, rewards, transition_probabilities, discount_factor, threshold):
    policy = dict()
    value_function = dict()
    for s in states:
        policy[s] = np.random.choice(actions[s])
        value_function[s] = -10**10

    stable = False
    iter = 0
    print(f'\nDiscount Factor: {discount_factor} - Threshold: {threshold}')
    while stable == False:
        print(f'Iteration: {iter}')
        value_function = policy_evaluation(states, rewards, transition_probabilities, discount_factor, threshold, policy, value_function)
        policy, change = policy_improvement(states, actions, transition_probabilities, policy, value_function)

        iter += 1
        if change == False:
            stable = True
            print(f'Optimized!!!')

    print(f'Optimal policy: {policy}')
    return policy, value_function

def make_MDP(negative_inventory, M, CD, max_probability):
    b = 12000 * 3
    L = 49.7
    alpha = 0.55
    DW = 0.144
    p = 200 * 5
    g = 262
    beta = 0.1
    h = dataTime.loc[CD]["11001"]
    i = 0.2

    sigma = 0
    mu = 0
    for ind in M:
        mu += dataRateAccidents[ind]
        sigma += dataStaDevAccidents[ind] ** 2

    sigma = sigma**(1/2)
    K = int(np.ceil(mu + 2*sigma))

    actions = dict()
    rewards = dict()
    transition_prob = dict()

    if negative_inventory == True:
        states = list(range(-K, K+1))

        #Generation of actions
        for s in states:
            actions[s] = list(range(0, K-s+1))

        #Estimation of rewards
        for s in states:
            for a in actions:
                if s < 0:
                    CI = b * L * (DW * alpha + beta)*s
                else:
                    CI = -p*s*i
                CE = 0
                if a > 0:
                    CE = -g * h
                CP = -p * a

                rewards[s, a] = CI + CE + CP


        #Estimation of matrix of probabilities
        for s_current in states:
            for a in actions:
                for s_next in states:
                    w = s_current + a - s_next
                    if s_next == -K:
                        transition_prob[s_current, a, s_next] = 1 - poisson.cdf(s_current+a+K-1, mu)
                    elif s_next == K:
                        transition_prob[s_current, a, s_next] = poisson.cdf(s_current+a-K, mu)
                    else:
                        transition_prob[s_current, a, s_next] = poisson.pmf(w, mu)

    else:
        states = [ind for ind in range(0, K+1)]

        #Iterations to generate the posible actions given the state s
        for s in states:
            actions[s] =  list(range(0, K-s+1))

        #Iterations to estimate the rewards of being in state s and taking the action a
        for s in states:
            for a in actions:
                temp_reward = 0
                cum_prob = 0
                w = 0
                CP = -p * a
                CE = 0
                if a > 0:
                    CE = -g * h

                while cum_prob < max_probability:
                    prob_temp = poisson.pmf(w, mu)
                    if s + a - w < 0:
                        CI = b * L * (DW * alpha + beta) * (s + a - w)
                    else:
                        CI = -p*s*i

                    temp_reward += CI*prob_temp
                    cum_prob += prob_temp
                    w += 1

                rewards[s, a] = temp_reward + CE + CP

        #Iterations to estimate the matrixes of probabilities of going to s' given that a previuos state is s
        #and took action a
        for s_current in states:
            for a in actions:
                for s_next in states:
                    if s_next == 0:
                        prob_temp = 1 - poisson.cdf(s_current + a - 1, mu)
                    elif s_next == K:
                        prob_temp = poisson.cdf(s_current + a - K, mu)
                    else:
                        prob_temp = poisson.pmf(s_current + a - s_next, mu)
                    transition_prob[s_current, a, s_next] = prob_temp

    return states, actions, rewards, transition_prob, mu, sigma

def plot_policy(policy, mu, sigma, title):
    plt.figure()
    plt.scatter(list(policy.keys()), list(policy.values()), color='teal')
    plt.vlines([mu, mu+sigma], 0, np.ceil(mu+2*sigma), color='black', linestyles='dashed')
    plt.ylabel('Orders')
    plt.xlabel('Inventory')
    plt.title(title)
    plt.savefig(os.path.join(cwd, 'plots', f'policy_{title}'), dpi=300)
    plt.close()

def local_nearest(M):
    data = dataTime.loc[M]
    str_M = list(map(str, M))
    data = data[str_M]
    sum_distance = data.sum()

    return int(sum_distance.index[np.argmin(sum_distance)])

def find_breaking_point(policy):
    for key, value in policy.items():
        if value == 0:
            return key

##
states2, actions2, rewards2, transition_prob2, mu_test, sigma_test = make_MDP(False, M_test, CD_test, 0.95)

array_probabilites2 = np.zeros((len(states2), len(actions2), len(states2)))
array_rewards2 = np.zeros((len(states2), len(actions2)))
for i, s1 in enumerate(states2):
    for j, a in enumerate(actions2):
        array_rewards2[i, j] = rewards2[s1, a]
        for k, s2 in enumerate(states2):
            array_probabilites2[i, j, k] = transition_prob2[s1, a, s2]


##
TOLERANCE = 0.005
GAMMA = 0.9
clusters = pickle.load(open(os.path.join(cwd, 'parameters', 'clusters_GMM.pickle'), 'rb'))
opt_policies = dict()
means_accidents = dict()
desv_accidents = dict()
ratios_accidents = dict()

for key, value in clusters.items():
    print(f'\nCluster: {key}')
    CD_temp = local_nearest(value)
    states_temp, actions_temp, rewards_temp, transition_prob_temp, mu_temp, sigma_temp = make_MDP(False, value, CD_temp, 0.9)
    policy_temp, V_temp = policy_iteration(states_temp, actions_temp, rewards_temp, transition_prob_temp, GAMMA, TOLERANCE)
    opt_policies[key] = policy_temp
    means_accidents[key] = mu_temp
    desv_accidents[key] = sigma_temp
    ratios_accidents[key] = mu_temp/sigma_temp

print('\nAll clusters optimized!!!')

##
for p in opt_policies:
    plot_policy(opt_policies[p], means_accidents[p], desv_accidents[p], f'Cluster {p}')

##
plt.figure()
plt.scatter(list(opt_policies[23].keys()), list(opt_policies[23].values()), color='teal')
plt.vlines([means_accidents[23], means_accidents[23]+desv_accidents[23]], 0,
           np.ceil(means_accidents[23]+2*desv_accidents[23]), color='black', linestyles='dashed')
plt.ylabel('Órdenes')
plt.xlabel('Inventario')
plt.title('Cluster 23', fontweight='bold')


plt.figure()
plt.scatter(list(opt_policies[5].keys()), list(opt_policies[5].values()), color='teal')
plt.vlines([means_accidents[5], means_accidents[5]+desv_accidents[5]], 0,
           np.ceil(means_accidents[5]+2*desv_accidents[5]), color='black', linestyles='dashed')
plt.ylabel('Órdenes')
plt.xlabel('Inventario')
plt.title('Cluster 5', fontweight='bold')


##
cluster_size = {key: len(value) for key, value in clusters.items()}

plt.figure()
plt.title('')
plt.xlabel('Cluster size')
plt.ylabel('Standard Deviation')
plt.scatter(cluster_size.values(), desv_accidents.values())
plt.grid()
plt.show()

r_squared = dict()
breaking_point = dict()
for p in opt_policies.keys():
    x = list(opt_policies[p].keys())
    y = list(opt_policies[p].values())
    slope_temp, intercept_temp, r2_temp, pValue_temp, stdErr_temp = linregress(x, y)
    breaking_point[p] = find_breaking_point(opt_policies[p])
    r_squared[p] = r2_temp

##
plt.figure()
plt.title('')
plt.scatter(cluster_size.values(), r_squared.values())
plt.ylabel(r'$R^{2}$')
plt.xlabel('Size of cluster')
plt.grid()
plt.show()

##

plt.figure(figsize=(7,5))
plt.title('')
plt.scatter(desv_accidents.values(), means_accidents.values(), c=list(r_squared.values()), cmap='magma')
cbar = plt.colorbar()
cbar.ax.set_ylabel(r'$\rho$')
plt.ylabel('Media')
plt.xlabel('Desviación Estándar')
plt.grid()
plt.savefig(os.path.join(cwd, 'plots', 'disp_pearsonCorrelation'), dpi=300)
plt.show()

##
plt.figure()
plt.title('')
plt.scatter(desv_accidents.values(), r_squared.values(), c='darkred')
plt.ylabel(r'$R^{2}$')
plt.xlabel('Standard deviation')
plt.grid()
plt.show()

##
plt.figure()
plt.title('')
plt.scatter(ratios_accidents.values(), r_squared.values(), c='darkred')
plt.ylabel(r'$R^{2}$')
plt.xlabel(r'Ratio $\mu/\sigma$')
plt.grid()
plt.show()

##
plt.figure()
plt.title('')
plt.scatter(desv_accidents.values(), breaking_point.values(), c='darkgreen')
plt.ylabel('Breaking point')
plt.xlabel(r'Size of clusters')
plt.grid()
plt.show()

##
list_municipalities = list(dataCoordinates.index)
list_cluster = list()
list_rational_policy = list()

for ind, munip in enumerate(list_municipalities):
    for key, value in clusters.items():
        if munip in value:
            list_cluster.append(key)
            break

for ind, c in enumerate(list_cluster):
    if np.abs(r_squared[c]) >= 0.7:
        list_rational_policy.append(1)
    else:
        list_rational_policy.append(0)

for ind, munip in enumerate(list_municipalities):
    if len(str(munip)) == 4:
        list_municipalities[ind] = '0'+str(munip)
    else:
        list_municipalities[ind] = str(munip)

df_maps = pd.DataFrame(list(zip(list_municipalities, list_cluster, list_rational_policy)),
                       columns=['DIVIPOLA', 'CLUSTER', 'RATIONAL_POLICY'])
df_maps.to_csv(os.path.join(cwd, 'results', 'df_mapping.csv'))