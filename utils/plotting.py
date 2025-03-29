import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

def plot_time_series(data, x, t, cmap='copper', vert_exag=8, fig=None, ax=None):
    if (fig is None) and (ax is None):
        fig, ax = plt.subplots()
    elif ((fig is None) and not(ax is None)) or (not(fig is None) and (ax is None)):
        raise RuntimeError('Figure and axis handle must both simultaneously given, or None')
    
    if cmap == 'redblue':
        ls = LightSource(azdeg=90, altdeg=90)
        rgb = ls.shade(data, cmap=plt.cm.RdBu, vert_exag=vert_exag, blend_mode='overlay')
        im = ax.imshow(data, cmap=plt.cm.RdBu, origin='lower')
    elif cmap == 'copper':
        ls = LightSource(azdeg=90, altdeg=85)
        rgb = ls.shade(data, cmap=plt.cm.copper, vert_exag=vert_exag, blend_mode='overlay')
        im = ax.imshow(data, cmap=plt.cm.copper, origin='lower')
    else:
        raise ValueError('Invalid colormap type. Choose between `copper` and `redblue`')
    
    im.remove()
    fig.colorbar(im, ax=ax)
    ax.imshow(rgb, origin='lower', aspect='auto', 
              extent=[x.min(), x.max(), t.min(), t.max()], 
              interpolation=None)