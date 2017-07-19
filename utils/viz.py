"""
Commonly used plotting tools
"""
import numpy as np

def scatter2D(ax, X, Y, d_col= None, uniq_col = None, size = 10, cmap = None):
    """
    size: array (X.shape[0],) or scalar - size of dots
    uniq_col: None or dict [keys - unique colors, values - name of color]
            if dict, expecting d_col: (X.shape[0],)
    """
    if uniq_col is None and d_col is None:
        ax.scatter(X, Y, s = size, cmap = cmap)
    elif uniq_col is None and d_col is not None:
        ax.scatter(X, Y, s = size, cmap = cmap, c = d_col)
    else:
        assert d_col is not None,'Expecting specification of colors'
        assert np.all(np.sort(np.unique(d_col))==np.sort(np.unique(uniq_col.keys()))),'Expecting all keys to have assigned colors'
        for k in uniq_col:
            idx = np.where(np.array(d_col)==k)[0]
            ax.scatter(X[idx], Y[idx], s = size, c = uniq_col[k])      
