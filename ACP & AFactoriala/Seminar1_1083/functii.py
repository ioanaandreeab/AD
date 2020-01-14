import numpy as np
import pandas as pd

# Definire functie pentru implementarea analizei in componente principale
def acp(x):
    n,m = np.shape(x)
    # Standardizare tabel x
    x_std = (x - np.mean(x,axis=0))/np.std(x,axis=0)
    # Calcul matrice de corelatii
    r = (1/n)*np.transpose(x_std)@x_std
    # Calcul vectori si valori proprii pentru matricea de corelatie
    valp,vecp = np.linalg.eig(r)
    k = np.flipud(np.argsort(valp))
    alpha = valp[k]
    a = vecp[:,k]
    c = x_std@a
    return r,alpha,a,c

# Definire functie pentru tabelarea variantei
def tabelare(alpha):
    m = len(alpha)
    alpha_cum = np.cumsum(alpha)
    alpha_proc = alpha*100/sum(alpha)
    alpha_proc_cum = np.cumsum(alpha_proc)
    tabel_varianta = pd.DataFrame(data={"Varianta":alpha,
                                        "Varianta cumulata":alpha_cum,
                                        "Procent varianta":alpha_proc,
                                        "Procent cumulat":alpha_proc_cum},
                                  index=["C"+str(i) for i in range(1,m+1)])
    return tabel_varianta