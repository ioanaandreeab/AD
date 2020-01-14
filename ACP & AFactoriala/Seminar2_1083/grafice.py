import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

def corelograma(t,vmin=-1,vmax=1):
    sb.heatmap(t,vmin,vmax,"RdYlBu")
    plt.show()

# Plot corelatii - Cercul corelatiilor
def plot_corelatii(r,k1=0,k2=1,nume_variabile=None):
    fig = plt.figure(figsize=(8,8))
    assert isinstance(fig,plt.Figure)
    ax = fig.add_subplot(1,1,1)
    assert isinstance(ax,plt.Axes)
    ax.set_title("Plot corelatii",fontsize = 16)
    t = np.arange(0,np.pi*2,0.01)
    ax.plot(np.cos(t),np.sin(t))
    ax.axhline(0,c='k')
    ax.axvline(0,c='k')
    ax.set_xlabel("Componenta "+str(k1+1))
    ax.set_ylabel("Componenta "+str(k2+1))
    ax.scatter(r[:,k1],r[:,k2],c='r')
    if nume_variabile is not None:
        m = np.shape(r)[0]
        for i in range(m):
            ax.text(r[i,k1],r[i,k2],nume_variabile[i])
    plt.show()

# Plot instante
def plot_instante(t,k1=0,k2=1,nume_instante=None):
    fig = plt.figure(figsize=(11,8))
    assert isinstance(fig,plt.Figure)
    ax = fig.add_subplot(1,1,1)
    assert isinstance(ax,plt.Axes)
    ax.set_title("Plot instante",fontsize = 16)
    ax.set_xlabel("a"+str(k1+1))
    ax.set_ylabel("a"+str(k2+1))
    ax.scatter(t[:,k1],t[:,k2],c='r')
    if nume_instante is not None:
        m = np.shape(t)[0]
        for i in range(m):
            ax.text(t[i,k1],t[i,k2],nume_instante[i])
    plt.show()

