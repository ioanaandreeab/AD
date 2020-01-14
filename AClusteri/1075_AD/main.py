import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as hclust #algorimi ierarhici
import grafice
import sklearn.cluster.hierarchical as skhclust
#1. aplicare model & construie grafic dendograma
#2. construire partitie
mortalitate = pd.read_csv("MortalitateEU/MortalitateEU.csv",index_col=0)
variabile = list(mortalitate)[1:]
nume_instante = list(mortalitate.index)

x = mortalitate[variabile].values
if np.isnan(x).any():
    print("Valori lipsa!!")
    #pentru ca exista valori lipsa, le inlocuim cu media
    k = np.where(np.isnan(x))
    x[k] = np.nanmean(x[:,k[1]],axis=0)
#print(variabile,x,sep="\n")


#aplicam modelul - 2 module care fac clusterizare - sklearn& scipy
#creare ierarhie
metoda = "ward"
#matricea ierarhie - de ex - clusterul 26 face jonctiune cu 31, distanta e de 11.59 si reszulta un cluster cu 2 elemente
h = hclust.linkage(x,method=metoda)
print("Matrice ierarhie:",h,sep="\n")
#Plot ierarhie - graficul dendograma
grafice.dendrograma(h,nume_instante,"Plot ierarhie. Metoda: "+metoda+" Metrica euclidiana")
#grafice.show()
#Determinare partitie optimala
m = np.shape(h)[0] #liniile de la shape; shape intoarce nr de linii si nr de coloane

# nr de clustere din partitia  optimala - trb sa vad unde este ceas mai mare distanta --ma intereseaza pozitia nu valoarea in sine
k_opt = m-np.argmax(h[:1,2]- h[:(m-1),2])
print("Partitia optimala are "+str(k_opt)+" clusteri")
#di si Pi
# di+1 - d2
#di+1 - di

#cobstruire model sklearn
model_clusterizare_sk =skhclust.AgglomerativeClustering(n_clusters = 5, linkage = metoda)
model_clusterizare_sk.fit(x)
coduri = model_clusterizare_sk.labels_
partitie =np.array(["Cluster"+ str(cod + 1) for cod in coduri])
tabel_partitie = pd.DataFrame(
    data={"Partitie ":partitie},
    index = mortalitate.index
)
print("Partitia optimala: ",tabel_partitie, sep="\n")
partitie = np.array("Cluster "+ str(cod+1) for cod in coduri)

#Plot partitie in axele principale
grafice.plot_partitie(x,partitie,nume_instante,"Partitie optimala")
grafice.show()