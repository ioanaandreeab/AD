import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as hclust
import grafice
import functii

tabel = pd.read_csv("MiscareaNaturalaAPop2016/MiscNatAPop_Judete.csv", index_col=0)
variabile = list(tabel)[1:]
nume_instante = list(tabel.index)

x = tabel[variabile].values
if np.isnan(x).any():
    print("Valori lipsa! Inlocuire cu media.")
    k = np.where(np.isnan(x))
    x[k] = np.nanmean(x[:, k[1]], axis=0)

# Construire ierarhie
h = hclust.linkage(x, method="complete")
# Vizualizare ierarhie - grafic dendrograma
print("Matricea ierarhie:",h,sep="\n")
grafice.plot_ierarhie(h,nume_instante)
# Identificare partitie optimala
m = np.shape(h)[0]
k = m - np.argmax(h[1:, 2] - h[:(m - 1), 2])
print("Partitia optimala are ",k," clusteri")
# Determinare partitie optimala
partitie = functii.partitie(h, k)
partitie_optimala = pd.DataFrame(
    data={"Partitie": partitie},
    index=nume_instante
)
partitie_optimala.to_csv("PartitaOptimala.csv")
clusteri = set(partitie)
partitie_ = np.array(partitie)
instante_ = np.array(nume_instante)
print("Partitia optimala:",partitie,sep="\n")
for cluster in clusteri:
    print("Cluterul " + cluster)
    print(instante_[partitie_==cluster])
# Afisare partitie in axe principale
grafice.plot_partitie(x,partitie,nume_instante)
partitie_5 = functii.partitie(h,k=5)
grafice.plot_partitie(x,partitie_5,nume_instante,titlu="Plot partitie cu 5 clusteri")
grafice.show()
