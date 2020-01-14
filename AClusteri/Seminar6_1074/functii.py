import numpy as np
import pandas as pd

def partitie(h,k):
    n = np.shape(h)[0] + 1
    c = np.arange(n)
    for i in range(n-k):
        k1 = h[i,0]
        k2 = h[i,1]
        c[ c==k1 ] = n+i
        c[ c==k2 ] = n+i
    c_ = pd.Categorical(c).codes
    return ["c"+str(i+1) for i in c_]