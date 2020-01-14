import matplotlib.pyplot as plt
import seaborn as sb

def plot_scoruri(z1, z2, u1, u2, nume_instante,titlu="Plot scoruri"):
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(1, 1, 1)
    assert isinstance(ax, plt.Axes)
    ax.set_title(titlu,fontsize=16)
    ax.scatter(z1, z2, c='r', label='Spatiul 1 (X)')
    ax.scatter(u1, u2, c='b', label="Spatiul 2 (Y)")
    n = len(nume_instante)
    for i in range(n):
        ax.text(z1[i], z2[i], nume_instante[i])
        ax.text(u1[i], u2[i], nume_instante[i])
    ax.legend()
    # plt.show()

def corelograma(t,vmin=-1,vmax=1,titlu="Corelograma"):
    fig = plt.figure(figsize=(11,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_title(titlu, fontsize=16)
    sb.heatmap(t,vmin=vmin,vmax=vmax,cmap='RdYlBu',ax=ax)
    # plt.show()

def show():
    plt.show()

