##
import numpy as np
import pandas as pd
import scipy.stats as stats
import os


##
cwd = os.getcwd()
dataCoordinates = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaCoordinates.csv'), index_col=0)
dataTime = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaTimeDistance.csv'), index_col=0)
dataAccidents = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaOA2019.csv'), index_col=0)
dataDepartments = pd.read_csv(os.path.join(cwd, 'parameters', 'ColombiaDepartments.csv'), index_col=0)

##
accidents_rate = []

for i in range(len(dataAccidents)):
    average_temp = np.sum(np.array(dataAccidents.iloc[i,3:15].to_numpy()))/12
    accidents_rate.append(average_temp*2)

rateAccidents = dataAccidents.iloc[:,0:3]
rateAccidents = rateAccidents.assign(Tasa=accidents_rate)



rateAccidents.to_csv(os.path.join(cwd, 'parameters', 'ColombiaAccidentRates.csv'), index=True)
