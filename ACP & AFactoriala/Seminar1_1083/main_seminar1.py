import pandas as pd
import functii
import grafice

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
# grafice.corelograma(t_r)
tabel_varianta = functii.tabelare(alpha)
tabel_varianta.to_csv("Varianta.csv")
