##
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.stats import poisson

##

cwd = os.getcwd()
dataCoordinates = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaCoordinates.csv'), index_col=0)
dataTime = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaTimeDistance.csv'), index_col=0)
dataAccidents = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaOA2019.csv'), index_col=0)
dataDepartments = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaDepartments.csv'), index_col=0)

TOLERANCE = 0.05
GAMMA = 0.9

dataRateAccidents = (dataAccidents.iloc[:, 3:]).mean(axis=1)
dataStaDevAccidents = np.std(dataAccidents.iloc[:, 3:], axis=1)

M_test = [73148, 73001, 41872, 73168, 73268, 73275, 41001, 41013, 73319, 73449]
CD_test= 73001


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

def makeClusters(M: list, q: dict):
    R = list()
    Mr = dict()

    return R, Mr

def make_MDP1(M, CD):
    states = list()
    actions = dict()
    rewards = dict()
    value_func = dict()

    b = 12000*3
    L = 49.7
    alpha = 0.55
    DW = 0.144
    p = 200*5
    g = 262
    gamma = 0.1
    h = dataTime.loc[CD]["11001"]
    i = 1


    V = 0
    sigma = 0
    for i in M:
        V += dataRateAccidents[i]
        sigma += dataStaDevAccidents[i]**2

    lamdbaTotal = V
    V = int(np.ceil(V))
    sigma = np.sqrt(sigma)
    V += int(np.ceil(2*sigma))


    states = [ind for ind in range(-V, V+1)]

    for s in states:
        actions[s] = [ind for ind in range(0, V-s+1)]

    for s in states:
        for a in actions[s]:
            r_temp = -(p*a)
            if s < 0:
                r_temp += b*L*(alpha*DW*s + gamma*s)
            else:
                r_temp -= 0
                #r_temp -= 1.5 * p * (s+a-lamdbaTotal)
            rewards[s, a] = r_temp

    for s in states:
        vals_temp = list()
        for a in actions[s]:
            sum_temp = 0
            for s_next in states:
                w = s + a - s_next
                sum_temp += poisson.pmf(w, lamdbaTotal)*(rewards[s, a])
            vals_temp.append(sum_temp)


    return states, actions, rewards, lamdbaTotal, V

def policy_evaluation1(states, rewards, lambda_total, gamma, threshold, policy, value_fun):
    converge = False
    while converge == False:
        delta = 0
        for s_act in states:
            temp = value_fun[s_act]
            curren_act = policy[s_act]
            new_v = rewards[s_act, curren_act]
            for s_nxt in states:
                w = s_act + curren_act - s_nxt
                new_v += gamma*poisson.pmf(w, lambda_total)*value_fun[s_nxt]
            value_fun[s_act] = new_v
            delta = max(delta, np.abs(temp - value_fun[s_act]))

        if delta < threshold:
            converge = True

    return value_fun

def policy_improvement1(states, actions, lambda_total, policy, value_fun):
    change = False
    for s_act in states:
        temp_pol = policy[s_act]
        max_value = -10**10
        for a in actions[s_act]:
            value_a = 0
            for s_nxt in states:
                w = s_act + a - s_nxt
                value_a += poisson.pmf(w, lambda_total)*value_fun[s_nxt]
            if value_a > max_value:
                policy[s_act] = a
                max_value = value_a
        if temp_pol != policy[s_act]:
            change = True

    return policy, change

def policy_iteration1(states, actions, rewards, lambda_total, gamma, threshold):
    policy = dict()
    for s in states:
        policy[s] = np.random.choice(actions[s])

    value_fun = dict()
    for s in states:
        value_fun[s] = -10 ** 10
    stable = False
    iter = 0
    while stable == False:
        print(iter)
        value_fun = policy_evaluation1(states, rewards, lambda_total, gamma, threshold, policy, value_fun)
        policy, change = policy_improvement1(states, actions, lambda_total, policy, value_fun)

        if change == False:
            stable = True
        iter +=1
    return policy, value_fun

states_test1, actions_test1, rewards_test1, lambda_test1, V_test1 = make_MDP1(M_test, CD_test)

array_rewards1 = np.zeros((len(states_test1), len(actions_test1)))
for i, s in enumerate(states_test1):
    for j, a in enumerate(actions_test1[s]):
        array_rewards1[i, j] = rewards_test1[s, a]


opt_policy1, value_fun1 = policy_iteration1(states_test1, actions_test1, rewards_test1, lambda_test1, GAMMA, TOLERANCE)

plt.figure()
plt.plot(value_fun1.keys(), value_fun1.values())
plt.show()

def MDP2(M):
    V = 0
    sigma = 0
    for i in M:
        V += dataRateAccidents[i]
        sigma += dataStaDevAccidents[i] ** 2

    lamdbaTotal = V
    V = int(np.ceil(V))
    sigma = np.sqrt(sigma)
    V += int(np.ceil(2 * sigma))

    states = [ind for ind in range(0, V + 1)]
    actions = [ind for ind in range(0, V + 1)]

    return states, actions, lamdbaTotal, V

def rewards2(s, a, CD, lambda_total, V, p_max):
    b = 12000 * 3
    L = 49.7
    alpha = 0.55
    DW = 0.144
    p = 200 * 5
    g = 262
    beta = 0.1
    h = dataTime.loc[CD]["11001"]
    i = 1.5

    reward = 0
    cum_prob = 0
    w = 0
    while cum_prob < p_max:
        prob_temp = poisson.pmf(w, lambda_total)
        CP = -p*a
        CE = 0
        if a > 0:
            CE = -g*h
        s_nxt = max(0, s+a-w, V)
        if s+a-w < 0:
            CI = b*L*(DW*alpha+beta)*(s+a-w)
        else:
            CI = 0

        reward += (CP+CI+CE)*prob_temp
        cum_prob += prob_temp
        w += 1

    return reward

states_test2, actions_test2, lambda_test2, V_test2 = MDP2(M_test)

rewards_test2 = dict()
array_rewards2 = np.zeros((len(states_test2), len(actions_test2)))
for i, s in enumerate(states_test2):
    for j, a in enumerate(actions_test2):
        rewards_test2[(s, a)] = rewards2(s, a, CD_test, lambda_test2, V_test2, 0.95)
        array_rewards2[i, j] = rewards2(s, a, CD_test, lambda_test2, V_test2, 0.95)


def policy_evaluation2(states, rewards, lambda_total, gamma, threshold, policy, value_fun, V):
    converge = False
    while converge == False:
        delta = 0
        for s_act in states:
            temp = value_fun[s_act]
            current_a = policy[s_act]
            new_v = rewards[s_act, current_a]
            for s_nxt in states:
                if s_nxt == 0:
                    prob_temp = 1 - poisson.cdf(s_act + current_a - 1, lambda_total)
                elif s_nxt == V:
                    prob_temp = poisson.cdf(s_act + current_a - V, lambda_total)
                else:
                    prob_temp = poisson.pmf(s_act + current_a - s_nxt, lambda_total)
                new_v += gamma*prob_temp*value_fun[s_nxt]
            value_fun[s_act] = new_v
            delta = max(delta, np.abs(temp - value_fun[s_act]))

        if delta < threshold:
            converge = True

    return value_fun

def policy_improvement2(states, actions, lambda_total, policy, value_fun, V):
    change = False
    for s_act in states:
        current_a = policy[s_act]
        max_value = -10**10
        for a in actions:
            value_a = 0
            for s_nxt in states:
                if s_nxt == 0:
                    prob_temp = 1 - poisson.cdf(s_act + current_a - 1, lambda_total)
                elif s_nxt == V:
                    prob_temp = poisson.cdf(s_act + current_a - V, lambda_total)
                else:
                    prob_temp = poisson.pmf(s_act + current_a - s_nxt, lambda_total)
                value_a += prob_temp*value_fun[s_nxt]
            if value_a > max_value:
                policy[s_act] = a
                max_value = value_a
        if current_a != policy[s_act]:
            change = True

    return policy, change

def policy_iteration2(states, actions, rewards, lambda_total, gamma, threshold, V):
    policy = dict()
    for s in states:
        policy[s] = np.random.choice(actions)

    value_fun = dict()
    for s in states:
        value_fun[s] = -10 ** 10
    stable = False
    iter = 0
    while stable == False:
        print(f'Iteración: {iter} \n Política: {policy}\n Función de Valor: {value_fun}')
        value_fun = policy_evaluation2(states, rewards, lambda_total, gamma, threshold, policy, value_fun, V)
        policy, change = policy_improvement2(states, actions, lambda_total, policy, value_fun, V)
        if change == False:
            stable = True
        iter +=1
    return policy, value_fun

opt_policy2, value_fun2 = policy_iteration2(states_test2, actions_test2, rewards_test2, lambda_test2, GAMMA, TOLERANCE, V_test2)

plt.figure()
plt.plot(value_fun2.keys(), value_fun2.values())
plt.show()

##
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
        max_value = -10 ** 10
        max_a = 0
        for a in actions:
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
        policy[s] = np.random.choice(actions)
        value_function[s] = -10**10

    stable = False
    iter = 0
    while stable == False:
        print(f'Iteration - {iter}')
        value_function = policy_evaluation(states, rewards, transition_probabilities, discount_factor, threshold, policy, value_function)
        policy, change = policy_improvement(states, actions, transition_probabilities, policy, value_function)

        iter += 1
        if change == False:
            stable = True

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
    for i in M:
        mu += dataRateAccidents[i]
        sigma += dataStaDevAccidents[i] ** 2

    sigma = sigma**(1/2)
    K = int(np.ceil(mu + 2*sigma))

    rewards = dict()
    transition_prob = dict()

    if negative_inventory == True:
        states = list(range(-K, K+1))

        #Generation of actions
        actions = list(range(0, K+1))

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
        actions = [ind for ind in range(0, K+1)]

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

    return states, actions, rewards, transition_prob

states1, actions1, rewards1, transition_prob1 = make_MDP(True, M_test, CD_test, 0.95)
states2, actions2, rewards2, transition_prob2 = make_MDP(False, M_test, CD_test, 0.95)

##

array_probabilites1 = np.zeros((len(states1), len(actions1), len(states1)))
array_rewards1 = np.zeros((len(states1), len(actions1)))
for i, s1 in enumerate(states1):
    for j, a in enumerate(actions1):
        array_rewards1[i, j] = rewards1[s1, a]
        for k, s2 in enumerate(states1):
            array_probabilites1[i, j, k] = transition_prob1[s1, a, s2]

array_probabilites2 = np.zeros((len(states2), len(actions2), len(states2)))
array_rewards2 = np.zeros((len(states2), len(actions2)))
for i, s1 in enumerate(states2):
    for j, a in enumerate(actions2):
        array_rewards2[i, j] = rewards2[s1, a]
        for k, s2 in enumerate(states2):
            array_probabilites2[i, j, k] = transition_prob2[s1, a, s2]


GAMMA = 0.7
TOLERANCE = 0.01

policy1, V1 = policy_iteration(states1, actions1, rewards1, transition_prob1, GAMMA, TOLERANCE)
policy2, V2 = policy_iteration(states2, actions2, rewards2, transition_prob2, GAMMA, TOLERANCE)

##
plt.figure()
plt.title('Estados sin inventario negativo')
plt.plot(V2.keys(), V2.values())
plt.show()

plt.figure()
plt.title('Estados con inventario negativo')
plt.plot(V1.keys(), V1.values())
plt.show()
##

