import numpy as np
import pandas as pd
import sklearn.decomposition as dec
import matplotlib.pyplot as plt

# citire din fisier csv si incarcare in obiect DataFrame
nume_fisier="ADN/ADN/ADN_Total.csv"
# preluare variabile -- aici sunt si valorile (tot csv-ul)
tabel = pd.read_csv(nume_fisier, index_col=0)
# lista cu denumirile variabilelor
variabile = list(tabel)
# lista cu toate inregistrarile - aici sunt tari
instante = list(tabel.index)
# valorile inregistrate pentru fiecare variabila
x = tabel.values
#luam dimensiunile matricei
n,m=np.shape(x)

# construire model ACP
m_ACP = dec.PCA()
m_ACP.fit(x)

# calcul tabel scoruri
scoruri = m_ACP.transform(x)
tabel_scoruri = pd.DataFrame(data=scoruri, index=instante,columns=["C"+str(i) for i in range (1,m+1)])
# tabel_scoruri.to_csv("tabel_scoruri_acp.csv")

# corelatii factoriale
# am nev de scoruri - sunt calculate
# corelatiile dintre X si s
rxs_ = np.corrcoef(x,scoruri,rowvar=False) # matricea asta o sa fie o matrice mai mare
rxs = rxs_[:m,m:] # matricea de corelatie
t_rxs = pd.DataFrame(data=rxs,index=variabile,columns=["Comp"+str(i) for i in range (1,m+1)]) # asta nu sunt sigura ca e bine dar corelatiile sunt ok
# t_rxs.to_csv("CorelatiiFactoriale.csv")


# plotul instantelor in 2 axe principale
def plot_scoruri(x,y,etichete,labelX,labelY,titlu="Plot scoruri"):
    fig = plt.figure(figsize=(11, 8))
    assert isinstance(fig, plt.Figure)
    ax = fig.add_subplot(1, 1, 1)
    assert isinstance(ax, plt.Axes)
    ax.set_title(titlu, fontsize=16, color='b')
    ax.set_xlabel(labelX)
    ax.set_ylabel(labelY)
    ax.scatter(x, y, c='r')
    n = len(etichete)
    for i in range(n):
        ax.text(x[i], y[i], etichete[i])
    plt.show()


plot_scoruri(scoruri[:,0],scoruri[:,1],instante,"C1","C2","Plot scoruri - componente")



