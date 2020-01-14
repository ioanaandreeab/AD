import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hclust
import seaborn as sb
import sklearn.decomposition as dec #resursa pentru componenetele principale

def dendrograma(h,nume_instante,titlu):
    fig = plt.figure(figsize=(12,8))
    #asertie facuta pentru autodocumentare - ca sa o recunoasca python drept apartinand acelei clase si pentru a avea autocomplete la functii
    assert isinstance(fig, plt.Figure)
    axa = fig.subplots()
    assert isinstance(axa, plt.Axes)
    axa.set_title(titlu, fontsize = 16, color = 'b')
    hclust.dendrogram(h,labels = nume_instante,ax=axa)

#x - setul de date
def plot_partitie(x,partitie,nume_instante,titlu):
    fig = plt.figure(figsize=(12, 8))
    assert isinstance(fig, plt.Figure)
    axa = fig.subplots()
    assert isinstance(axa, plt.Axes)
    axa.set_title(titlu, fontsize=16, color='b')
    pca = dec.PCA(n_components=2) #ma intereseaza primele 2 componenete
    z = pca.fit_transform(x)
    sb.scatterplot(z[:,0],z[:,1], hue=partitie,ax = axa)
    n = len(nume_instante)
    for i in range(n):
        axa.text(z[i,0],z[i,1],nume_instante[i])

def show():
    plt.show()