import numpy as np
import matplotlib.pyplot as plt

def inlocuire_nan(x):
    isnan = np.isnan(x)
    assert isinstance(isnan,np.ndarray)
    if isnan.any():
        k = np.where(isnan) #am identificat pozitiile cu valori lipsa
        x[k] = np.nanmean(x[:,k[1]],axis=0) #media fara pozitiile lipsa

#x - scoruri comp 1 , y - scoruri comp2, label-etichete, lx- label pt axa x, ly - label pt axa y, titlu
def plot_scoruri(x,y,label,lx,ly,title):
    fig = plt.figure(figsize=(10,7)) #construim figura
    assert isinstance(fig,plt.Figure)
    ax = fig.add_subplot(1,1,1) #subplotul ax - matrice cu o linie, o coloana, elem indexate de la 1
    assert isinstance(ax,plt.Axes)
    ax.set_title(title,fontsize = 16, color = 'b')
    ax.set_xlabel(lx)
    ax.set_ylabel(ly)

    ax.scatter(x,y,c='r') #c e culoare, dar puteam sa zic si color
    #asociem fiecarui punct textul
    n = len(label) #nr de etichete
    for i in range(n):
        ax.text(x[i],y[i],label[i])
    #plt.show()

#functie pt a face show la grafice in acelasi timp adica sa nu mai inchid unul ca sa apara celalalt - pt comparatii
def show():
    plt.show()