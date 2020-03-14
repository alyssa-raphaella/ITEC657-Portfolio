import numpy as np
from matplotlib import pyplot as plt

# pearson value - correlation
def corr(x,y, **kwargs):
    coef = np.corrcoef(x,y)[0][1]
    label = r'$\rho$ = ' + str(round(coef, 2))
    
    ax = plt.gca()
    ax.annotate(label, xy = (0.2, 0.95), size = 20, xycoords = ax.transAxes)