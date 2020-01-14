import sklearn.decomposition as dec
import pandas as pd
import numpy as np
import grafice
import factor_analyzer as fact

t = pd.read_csv("ADN/ADN/ADN_Total.csv",index_col=0)
variabile = list(t)
instante = list(t.index)
x = t.values
n,m = np.shape(x)
isna = np.isnan(x)
assert isinstance(isna,np.ndarray)
if isna.any():
    print("Valori lipsa")
# Construire model An. in Comp. Principale
model_acp = dec.PCA()
x_std = (x - np.mean(x,axis=0))/np.std(x,axis=0)
model_acp.fit(x_std)
# Aplicare model si extragere rezultate
# Calcul scoruri
s = model_acp.transform(x_std)
t_s = pd.DataFrame(data=s,index=instante,columns=["C"+str(i) for i in range(1,m+1)])
t_s.to_csv("scoruri_acp.csv")
grafice.plot_scoruri(s[:,0],s[:,1],instante,"C1","C2","Plot scoruri - Componente")
# Preluare axe factoriale
a = model_acp.components_
alpha = model_acp.explained_variance_
print("Axe componente (sklearn):",a,sep="\n")
print("Varianta componente (sklearn):",alpha,sep="\n")
print("Componente (sklearn):",s,sep="\n")


# ANALIZA FACTORIALA
model_afact = fact.FactorAnalyzer(rotation=None)
model_afact.fit(x)
l = model_afact.loadings_
# Calcul scoruri
f = model_afact.transform(x)
grafice.plot_scoruri(f[:,0],f[:,1],instante,"F1","F2","Plot scoruri - Factori")
print("Legatura variabile - factori comuni (factor_analyzer):", l, sep="\n")
print("Factori comuni (factor_analyzer):", f, sep="\n")
grafice.show()