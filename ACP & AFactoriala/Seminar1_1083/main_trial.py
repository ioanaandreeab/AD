import pandas as pd
import numpy as np

nume_fisier="ADN/ADN_Total.csv"

#preluare date din csv in obiect DataFrame
tabel=pd.read_csv(nume_fisier,index_col=0)
print(tabel)