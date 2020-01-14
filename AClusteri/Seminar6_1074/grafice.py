import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hclust
import sklearn.decomposition as dec
import seaborn as sb


def plot_ierarhie(h, nume_instante):
    fig = plt.figure(figsize=(12, 8))
    assert isinstance(fig, plt.Figure)
    ax = fig.subplots()
    assert isinstance(ax, plt.Axes)
    ax.set_title("Plot ierarhie - Dendrograma", fontsize=16, color='b')
    hclust.dendrogram(h, labels=nume_instante, ax=ax)
    # plt.show()


def plot_partitie(x, partitie, nume_instante,titlu="Plot partitie optimala"):
    fig = plt.figure(figsize=(12, 8))
    assert isinstance(fig, plt.Figure)
    ax = fig.subplots()
    assert isinstance(ax, plt.Axes)
    ax.set_title(titlu, fontsize=16, color='b')
    acp = dec.PCA(n_components=2)
    z = acp.fit_transform(x)
    sb.scatterplot(z[:, 0], z[:, 1], hue=partitie, ax=ax, s=100)
    n = len(nume_instante)
    for i in range(n):
        ax.text(z[i, 0], z[i, 1], nume_instante[i],fontsize=14)
    # plt.show()


def show():
    plt.show()
