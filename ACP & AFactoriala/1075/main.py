#calcul scoruri si grafic instante in primele doua axe ale componentelor 1 si 2
import pandas as pd
import sklearn.decomposition as dec #aici se afla pca - principal component analysis -- aici e pe esantion -> 1/n-1 * Xt * X
import functions
import numpy as np
import factor_analyzer as fact

#preluare date & prelucrare valori lipsa
t=pd.read_csv("Freelancer/FreelancerT.csv",index_col=1)
variabile = list(t)
variabile_prelucrate = variabile[2:]
#print variabile
x=t[variabile_prelucrate].values
#luam nr de linii si de coloane
n,m = np.shape(x)
#inlocuim valorile lipsa
functions.inlocuire_nan(x)
print(x)

#construire model PCA - Analiza in componente principale
#pt a calcula componentele principale avem nevoie de vectorul A - pe care il are modelul
model_pca = dec.PCA()
#calcul matrice standardizata - OPTIONAL
x_std = (x-np.mean(x,axis=0))/np.std(x,axis=0)
model_pca.fit(x_std)
#calcul scoruri
s = model_pca.transform(x_std)

#apelam plotul
functions.plot_scoruri(s[:,0],s[:,1],list(t.index),"C1","C2","Plot Scoruri - ACP") #coloana 1 din scoruri, aferenta primei componente principale


#corelatii factoriale & trasati o corelograma
#am nev de scoruri - sunt calculate
#corelatiile dintre X si s
rxs_ = np.corrcoef(x,s,rowvar=False) #matricea asta o sa fie o matrice mai mare
rxs = rxs_[:m,m:] #matricea de corelatie
print(rxs)
#corelograma matrice de corelatii in seminarele 1 sau 2

#sa se calc varianta componentelor principale si sa se tabeleze
alpha = model_pca.explained_variance_
#valorile astea le trimit in functie si le tabelez
a = model_pca.components_


#analiza factoriala
#elimin rotatia
model_fa = fact.FactorAnalyzer(rotation=None)
#construiesc modelul - standardizeaza modelul
model_fa.fit(x)
#calcul scoruri
f = model_fa.transform(x)
#plot al scorurilor
functions.plot_scoruri(f[:,0],f[:,1],list(t.index),"F1","F2","Plot Scoruri - Analiza Factoriala") #coloana 1 din scoruri, aferenta primei componente principale


#matrice de corelatii
l= model_fa.loadings_

#varianta factorilor
alpha_fa = model_fa.get_factor_variance() #aici merge facuta si tabelare

# model factorial sklearn - daca e fara factor_analyzer -- alta metoda de factorizare folosita
model_fa_sk = dec.FactorAnalysis(n_components=3)
model_fa_sk.fit(x)
#extragem scorurile
f_sk = model_fa_sk.transform(x)
functions.plot_scoruri(f_sk[:,0],f_sk[:,1],list(t.index),"F1","F2","Plot Scoruri SK - Analiza Factoriala") #coloana 1 din scoruri, aferenta primei componente principale
functions.show()

