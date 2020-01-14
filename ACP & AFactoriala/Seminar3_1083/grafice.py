import matplotlib.pyplot as plt

def plot_scoruri(x,y,etichete,tx,ty,titlu="Plot scoruri"):
    fig = plt.figure(figsize=(11,8))
    assert isinstance(fig,plt.Figure)
    ax = fig.add_subplot(1,1,1)
    assert isinstance(ax,plt.Axes)
    ax.set_title(titlu,fontsize = 16, color = 'b')
    ax.set_xlabel(tx)
    ax.set_ylabel(ty)
    ax.scatter(x,y,c='r')
    n = len(etichete)
    for i in range(n):
        ax.text(x[i],y[i],etichete[i])
    # plt.show()

def show():
    plt.show()
