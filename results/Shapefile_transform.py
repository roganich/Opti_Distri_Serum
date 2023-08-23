##
import geopandas as gdp
import matplotlib.pyplot as plt
import dbf
import pandas as pd
import os
import rasterio
import numpy as np
import rasterio.mask

import mapclassify

maps = pd.read_csv(os.path.join('results','df_mapping.csv'), index_col=1)
shape = gdp.read_file("mpio.shp")
departamentos = gdp.read_file("depto.shp")
casos = gdp.read_file("DataMUNIP.shp")

src = rasterio.open('TravelTime.tif')

out_image, out_transform = rasterio.mask.mask(src, casos["geometry"], crop=True)
out_meta = src.meta

with rasterio.open("TravelTimeCol.tif", "w", **out_meta) as dest:
    dest.write(out_image)

antioquia = departamentos[departamentos['NOMBRE_DPT']=='ANTIOQUIA']
##
from rasterio.plot import show

src2 = rasterio.open('TravelTimeCol.tif')

fig, ax = plt.subplots(figsize=(6,7))

image_hidden = ax.imshow(src.read()[0,:,:],
                         cmap='magma')

extent=[src2.bounds[0], src2.bounds[2], src2.bounds[1], src2.bounds[3]]
ax = rasterio.plot.show(src2, extent=extent, ax=ax, cmap='magma')

fig.colorbar(image_hidden, ax=ax)
ax.axis('off')

##
list_cluster = []
list_policy = []
list_newDept = []
deptos_reales = 0

for munip in shape.get('MPIOS'):
    if int(munip) in list(maps.index):
        deptos_reales += 1
        cluster_temp = maps.loc[int(munip)]['CLUSTER']
        policy_temp = maps.loc[int(munip)]['RATIONAL_POLICY']
        dept_temp = shape[shape['MPIOS'] == (munip)]['NOMBRE_DPT'].iloc[0]
    else:
        cluster_temp = float('nan')
        policy_temp = float('nan')
        dept_temp = float('nan')

    list_cluster.append(cluster_temp)
    list_policy.append(policy_temp)
    list_newDept.append(dept_temp)

##

shape = shape.assign(CLUSTER=list_cluster)
shape = shape.assign(POLICY=list_policy)
shape = shape.assign(DPTO_NEW=list_newDept)

list_newCases2019 = []
for ind in casos.index:
    if casos.loc[ind]['C2019'] == 0:
        list_newCases2019.append(float('nan'))
    else:
        list_newCases2019.append(casos.loc[ind]['C2019'])

casos = casos.assign(C2019_new=list_newCases2019)
##
fig, ax = plt.subplots(figsize=(6, 7))

casos.plot(column='C2019_new',
            categorical=False,
            scheme='Quantiles',
            markersize=45,
            k=4,
            legend=True,
            legend_kwds={"loc": 3},
            cmap="magma", ax=ax)
ax.axis('off')
ax.set_title('Accidente Ofídico \n Promedio Mensual (2019)', fontweight='bold')

##

fig, ax = plt.subplots(figsize=(6, 7))

base = antioquia.boundary.plot(color='black', ax=ax)
shape.plot(column='DPTO_NEW',
                         categorical=True,
                         markersize=45,
                         cmap="Set2", ax=base)
ax.axis('off')
ax.set_title('Conexiones Jerárquicas', fontweight='bold')

##

fig, ax = plt.subplots(figsize=(6, 7))

base = antioquia.boundary.plot(color='black', ax=ax)
shape.plot(column='CLUSTER',
                         categorical=True,
                         markersize=45,
                         cmap="Set2", ax=base)

ax.set_title('Conexiones mediante GMM \n+ Municipios Puentes', fontweight='bold')
ax.axis('off')


##
fig, ax = plt.subplots(figsize=(6, 7))

shape.plot(column='POLICY',
                         categorical=True,
                         figsize=(10,6),
                         markersize=45,
                         cmap="Set2", ax=ax)
ax.set_title('Mapa de Racionalidad de Políticas', fontweight='bold')
ax.axis('off')
##

