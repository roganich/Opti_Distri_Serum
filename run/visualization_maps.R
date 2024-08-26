library(terra)
library(rstudioapi)
library(readxl)

setwd(dirname(getActiveDocumentContext()$path))
main_path = dirname(getwd())

df_clusters = read.csv(paste0(main_path, '//results//clusters.csv'))
gdf_munipCol = vect(paste0(main_path, '/parameters/DataMUNIP.shp'))

gdf_munipCol$Hier=NA
gdf_munipCol$GMM=NA
gdf_munipCol$GMMCapitals=NA
gdf_munipCol$GMMBridges=NA

count=gdf_munipCol$MPIO_CCNCT[gdf_munipCol$MPIO_CCNCT %in% df_clusters$X]

cont = 1
for (i in count){
  print(paste0("Porcentaje abarcado del ciclo: ",round(cont/length(count)*100,3),"%" ))
  gdf_munipCol$Hier[gdf_munipCol$MPIO_CCNCT==i] <- df_clusters$hier[df_clusters$X==i]
  gdf_munipCol$GMM[gdf_munipCol$MPIO_CCNCT==i] <- df_clusters$GMM[df_clusters$X==i]
  gdf_munipCol$GMMCapitals[gdf_munipCol$MPIO_CCNCT==i] <- df_clusters$GMMCapitals[df_clusters$X==i]
  gdf_munipCol$GMMBridges[gdf_munipCol$MPIO_CCNCT==i] <- df_clusters$GMMBridges[df_clusters$X==i]
  cont = cont + 1
}

writeVector(gdf_munipCol, "gdf_munipColClusters.shp", overwrite=TRUE)

#plot(gdf_munipCol, "Hier",plg=list(title="Hierarchical"))
#plot(gdf_munipCol, "GMM",plg=list(title="GMM"))
#plot(gdf_munipCol, "GMMBridges",plg=list(title="GMMBridges"))
#plot(gdf_munipCol, "GMMCapitals",plg=list(title="GMMCapitals"))
