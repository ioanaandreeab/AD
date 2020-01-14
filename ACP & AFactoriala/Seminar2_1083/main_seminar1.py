import pandas as pd
import functii
import grafice
import numpy as np

nume_fisier = "ADN/ADN_Total.csv"
# Citire date din fisier in obiect pandas DataFrame
tabel = pd.read_csv(nume_fisier,index_col=0)
# print(tabel)
variabile = list(tabel)
instante = list(tabel.index)
# print(variabile,instante,sep="\n")
x = tabel[variabile].values
# print(x,type(x),sep="\n")
r,alpha,a,c = functii.acp(x)
t_r = pd.DataFrame(data=r,index=variabile,columns=variabile)
t_r.to_csv("R.csv")
grafice.corelograma(t_r)
tabel_varianta = functii.tabelare(alpha)
tabel_varianta.to_csv("Varianta.csv")
# Calcul corelatii factoriale
rxc = a*np.sqrt(alpha)
nume_componente = ["Comp"+str(i) for i in range(1,len(alpha)+1)]
t_rxc = pd.DataFrame(data=rxc,index=variabile,columns=nume_componente)
t_rxc.to_csv("rxc.csv")
# Trasare corelograma corelatii factoriale
grafice.corelograma(t_rxc)
# Trasare plot corelatii
grafice.plot_corelatii(rxc,nume_variabile=variabile)
# Calcul scoruri
s = c/np.sqrt(alpha)
t_s = pd.DataFrame(data=s,index=instante,columns=nume_componente)
t_s.to_csv("s.csv")
# Plot instante
grafice.plot_instante(s,nume_instante=instante)
# grafice.plot_instante(c,nume_instante=instante)
# Calcul cosinusuri
c2 = c*c
cos = np.transpose(np.transpose(c2)/np.sum(c2,axis=1))
t_cos = pd.DataFrame(data=cos,index=instante,columns=nume_componente)
t_cos.to_csv("cos.csv")
# Calcul contributii
n = len(instante)
beta = c2/(n*alpha)
t_beta = pd.DataFrame(data=beta,index=instante,columns=nume_componente)
t_beta.to_csv("contributii.csv")
# Calcul comunalitati
comm = np.cumsum(rxc*rxc,axis=1)
t_comm = pd.DataFrame(data=comm,index=variabile,columns=nume_componente)
t_comm.to_csv("comm.csv")
# Corelograma comunalitati
grafice.corelograma(t_comm)
