# Analiza canonica
import pandas as pd
import sklearn.cross_decomposition as cdec
import numpy as np
import grafice
import sklearn.preprocessing as pp

pd.set_option("display.max_columns", None)
# preluare primul set de date
t1_consum = pd.read_csv("EnergieEU/EnergieEU/FinalFeg.csv", index_col=0)
# preluare al doilea set de date
t2_prod = pd.read_csv("EnergieEU/EnergieEU/FinalIceg.csv", index_col=0)
var_x = list(t1_consum)[1:]
var_y = list(t2_prod)[1:]
# creare tabel cu cele doua seturi de date
t = t1_consum.join(other=t2_prod, how="inner", lsuffix="_x", rsuffix="_y")
# print(t)
# numele variabilelor
nume_instante = list(t.index)

# valorile variabilelor
x = t[var_x].values
y = t[var_y].values
# nr linii si coloane x
n, p = x.shape
# nr coloane y
q = y.shape[1]
# nr minim de coloane dintre cele doua vb
m = min(p, q)
# print(n, p, q, m, x, y, sep="\n")

# Creare model CCA - Canonical Correlation Analysis
cca_model = cdec.CCA(m)
cca_model.fit(x, y)

# Extragere rezultate
# Preluare scoruri
z = cca_model.x_scores_
u = cca_model.y_scores_
# Normalizare scoruri
pp.normalize(z, axis=0, copy=False)
pp.normalize(u, axis=0, copy=False)
# Calcul corelatii canonice

r = np.diagonal(np.corrcoef(z, u, rowvar=False)[:m, m:])
print("Corelatii canonice:", r)

# Calcul corelatii dintre variabilele observate si variabilele canonice
z_std = np.std(z, axis=0)
u_std = np.std(u, axis=0)
rxz = cca_model.x_loadings_ * z_std
ryu = cca_model.y_loadings_ * u_std
print("Corelatii variabile-variabile canonice:")
print(rxz, ryu, sep="\n")
# Plot scoruri
grafice.plot_scoruri(z[:, 0], z[:, 1], u[:, 0], u[:, 1], nume_instante)
# Corelograma corelatii
t_rxz = pd.DataFrame(data=rxz, index=var_x,
                     columns=['z' + str(i) for i in range(1, m + 1)])
t_rxz.to_csv("rxz.csv")
grafice.corelograma(t_rxz, titlu="Corelograma X-Z")
t_ryu = pd.DataFrame(data=ryu, index=var_y,
                     columns=['u' + str(i) for i in range(1, m + 1)])
t_ryu.to_csv("ryu.csv")
grafice.corelograma(t_ryu, titlu="Corelograma Y-U")
grafice.show()
