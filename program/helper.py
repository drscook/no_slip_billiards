## Some stuff in here is irrelevent for this project.  This is a collection of convenience commands I use across many projects.

import os
import time
from copy import copy as copy
import math
import numpy as np
import pandas as pd
import itertools as it
import ipywidgets as widg

# Graphics imports
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import IPython.display as ipd
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)

#import seaborn as sns
#sns.set_palette('deep')

#plt.style.use("fivethirtyeight")
#plt.rc("figure", figsize=(3,2))


### Display
import traceback
import warnings
import sys
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))
warnings.showwarning = warn_with_traceback


pd.options.display.show_dimensions = True
decimals = 4
tol = 1e-6
def num_format(x, tol=tol, decimals = decimals):
    y = copy(x)
    if np.abs(y) < tol:
        y = 0
    if np.abs(np.imag(y)) < tol:
        y = np.real(y)
    return np.round(y, decimals)
    
import webbrowser
def display(X, rows=None, where="inline", filename="df"):
    if(rows == 'all'):
        rows = 2000
    elif(type(rows) is int):
        rows *= 2
    else:
        rows = 100

    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series) or (isinstance(X, np.ndarray) and X.ndim <=2):
        Y = pd.DataFrame(copy(X))
        num = Y.select_dtypes(include=['number'])
        num = num.applymap(num_format)
        Y[num.columns] = num
        if(where == "popup"):
            filename = name + ".html"
            Y.to_html(filename)
            webbrowser.open(filename,new=2)
        else:
            pd.set_option('display.max_rows', rows)
            ipd.display(Y)
            pd.reset_option('display.max_rows')
    else:
        ipd.display(X)

        
        
### Data Science
def margins(df):
    df = pd.DataFrame(df)
    col_sums = df.sum(axis=0)
    df.loc['TOTAL'] = col_sums
    row_sums = df.sum(axis=1)
    df['TOTAL'] = row_sums
    return df
        

def get_summary_stats(v):    
    ss = pd.DataFrame(v).describe().T
    ss['SE'] = ss['std'] / np.sqrt(ss['count'])
    return ss



    






### linear algebra
def eigs(A, orient='col'):
    """
    Returns eigenvalues and eigenvectors.  Eigenvectors scaled to sum to 1 (probability distributions) where possible.
    """
    if orient == 'row':
        B = A.T
    elif orient == 'col':
        B = A
    else:
        raise Exception("orient must be 'row' or 'col'")
    evals, evecs = np.linalg.eig(B)
    s = evecs.sum(axis=0)
    transients = np.abs(s) < 0.01
    s[transients] = 1
    evecs = evecs / s
    if orient == 'row':
        evecs = evecs.T
    return evals, evecs

def make_symmetric(A):
    """
    Returns symmetric matrix by copying upper triangular onto lower.
    """
    A = np.asarray(A)
    U = np.triu(A,1)
    return np.triu(A,0) + U.T

def make_skew_symmetric(A):
    """
    Returns symmetric matrix by copying -1*upper triangular onto lower.
    """
    A = np.asarray(A)
    U = np.triu(A,1)
    return U - U.T


### The functions parse_ax, mdot, mdist, mmag, munit, make_orth_norm_basis work together to do basic linear algebra jobs.  
def parse_ax(ax):
    ax = listify(ax)
    l = len(ax)
    if l == 2:
        pass
    elif l == 1:
        ax.append(ax[0])
    elif l == 0:
        ax = [-1, -1]
    else:
        raise Exception('axis list must be length 0, 1, or 2')
    return ax

def mdot(A, B, ax=[]):
    A = np.asarray(A)
    B = np.asarray(B)
    dA = A.ndim
    dB = B.ndim
    ax = parse_ax(ax)
    ax = [ax[0]%dA, ax[1]%dB]
    if ax[0] == dA-1:
        pass
    else:
        A = np.rollaxis(A, ax[0], dA)
    if ax[1] == dB-2:
        pass    
    elif ax[1] == dB-1:
        B = np.rollaxis(B, ax[1], dB-2)
    else:
        B = np.rollaxis(B, ax[1], dB-1)
    return A.dot(B)
    
def mdist(A, B, ax=[], p=2):
    A = np.asarray(A)
    B = np.asarray(B)
    dA = A.ndim
    dB = B.ndim
    ax = parse_ax(ax)
    ax = [ax[0]%dA, dA + ax[1]%dB]
    C = np.subtract.outer(A,B)
    if p%2 == 0:
        C **= p
    else:
        C = np.abs(C)
        if p != 1:
            C **= p
    C = np.rollaxis(C,ax[0],0)
    C = np.rollaxis(C,ax[1],1)
    D = np.einsum('ii...',C)
    if p != 1:
        D **= (1/p)
    return D

def mmag(A, ax=-1, p=2):
    A = np.asarray(A)
    B = np.zeros(A.shape[ax])
    return mdist(A, B, ax=[ax,0], p=p)

def munit(A, ax=-1, p=2):
    A = np.asarray(A)
    l = mmag(A, ax=ax, p=p)
    return A / np.expand_dims(l, ax)


def make_orth_norm_basis(v, preserve_direction=True):
    """
    Converts list of n vectors in dimension d into an orthonormal basis using Graham=Schmidt.  Assumes vectors written across rows.  If n < d, completes basis by selecting d-n uniformly random vectors.  If preserve_direction==False, the input vectors will be freely modified so that the resulting basis will be fully orthonormal.  If preserve_direction==True, the input vectors will only be rescaled.  In this case, the first n ouput vectors may not be mutually orthogonal (but the d-n new vectors will be orthogonal to all others).
    """
    if v.ndim == 0:
        try:
            return np.eye(v)
        except:
            raise Exception('error in make_orth_norm_basis')
    elif v.ndim == 1:
        v = v[np.newaxis,:]
    v = munit(v,ax=1)
    n,d = v.shape
    if n > d:
        raise Exception("Shape error {}: must not increase".format(v.shape))    

    max_attempts = 20
    def Graham_Schmidt(w=None):
        """
        Let b1,b2,...,bk be the vectors produced in the first k iterations of GS.  Then proj = (b1.w)b1 + (b2.w)b2 + ... +(bk.w)bk.  Returns scaled orth = w - proj
        """
        try:
            c = onb.dot(w)
        except:
            c = 0
        for attempt in range(max_attempts):
            if np.any(c!=0):
                break
            else:
                w = np.random.rand(d)
                c = onb.dot(w)
        proj = c.dot(onb)
        orth = munit(w - proj)
        return orth

    onb = np.zeros([d,d])
    onb[0,:] = v[0]
    for i in range(1,n):        
        onb[i,:] = Graham_Schmidt(v[i])
    for i in range(n,d):
        onb[i,:] = Graham_Schmidt()
    if preserve_direction == True:
        onb[:n] = v
    return onb




### Solvers
def solve_linear(a, b):
    """
    Finds real solutions to ax+b=0 handling degenerate cases
    """
    a = np.asarray(a,dtype=float)
    b = np.asarray(b,dtype=float)
    if not (a.shape == b.shape):
        raise Exception('a, b must have same shape')
    else:        
        t = np.full_like(a, np.inf)
        idx = np.abs(a) > 1e-8
        t[idx] = b[idx] / a[idx]
    return t

def solve_quadratic(a, b, c):
    """
    Finds real solutions to ax**2+bx+c=0 handling degenerate cases (complex solutions reported as np.inf).  Solutions sorted by increading absolute value.
    """
    a = np.asarray(a,dtype=float)
    b = np.asarray(b,dtype=float)
    c = np.asarray(c,dtype=float)
    if not (a.shape == b.shape == c.shape):
        raise Exception('a, b, c must have same shape')
    else:
        no_a_idx = (np.abs(a) < 1e-8)
        no_b_idx = (np.abs(b) < 1e-8)
        lin_idx = no_a_idx & ~no_b_idx
        quad_idx = ~no_a_idx
        
        d = np.full_like(a, np.inf)
        t_small = d.copy()
        t_small[lin_idx] = -1*c[lin_idx] / b[lin_idx]
        t_big = t_small.copy()

        d[quad_idx] = b[quad_idx]**2 - 4*a[quad_idx]*c[quad_idx]
        real_idx = (d >= 0) & quad_idx
        d[real_idx] = np.sqrt(d[real_idx])
        t_small[real_idx] = (-1*b[real_idx] - d[real_idx]) / (2*a[real_idx])
        t_big[real_idx]  = (-1*b[real_idx] + d[real_idx]) / (2*a[real_idx])
        swap_idx = real_idx & (b > 0) # small := closest to 0 in absolute value.  If b>0, the "minus" root is father from 0 than the "plus" root.
        t_small[swap_idx], t_big[swap_idx] = t_big[swap_idx], t_small[swap_idx]
    return t_small, t_big



### Misc Utilities
def atrunc(A, tol=1e-8):
    A[np.abs(A) < tol] = 0
    A = np.real_if_close(A)
    return A

def marange(*sh):
    return np.arange(np.prod(sh)).reshape(*sh)

def mexpand(A, length=1, axis=None):
    """
    Creates new axis before axis with length l by "tiling" (copying) existing array
    """
    A = np.asarray(A)
    if axis is None:
        axis = A.ndim
    elif axis < 0:
        axis %= A.ndim
    sh = list(A.shape)
    sh.insert(axis, 1)
    B = A.reshape(sh)
    re_sh = np.ones_like(sh)
    re_sh[axis] = length
    return np.tile(B, re_sh)


def listify(x):
    if isinstance(x,str):
        x = [x]
    try:
        return list(x)
    except:
        return [x]
    
### Helpers for stochastic process and graphs
#import networkx as nx
def make_stochastic(A, orient='col'):    
    if orient == 'row':
        T = A.T
    elif orient == 'col':
        T = A
    else:
        raise Exception("orient must be 'row' or 'col'")
    s = T.sum(axis=0)
    sinks = np.abs(s) < 1e-4
    T[sinks,sinks] = 1
    s[sinks] = 1
    T = T / s
    if orient == 'row':
        T = T.T
    return T


def page_rankify(A, p=0.9, orient='col'):
    T = make_stochastic(A, orient)
    n = len(A)
    r = (1-p)/n
    T = p*T + r
    return T

def render_graphviz(G, node_labels=None, edge_labels=None, w='2in', fmt='svg', show=True): 
    if node_labels is not None:
        for v in G.nodes():
            try:
                l = round(node_labels[v],2)
            except:
                l = node_labels[v]
            G.nodes[v]['label'] = l
        
    if edge_labels is not None:
        for e in G.edges():
            try:
                l = round(edge_labels[e],2)
            except:
                l = edge_labels[e]
            G.edges[e]['label'] = l
    fn = 'im\\graph'+str(round(time.time()*1000))
    dot_file = fn+'.dot'
    im_file = fn+'.'+fmt

    nx.drawing.nx_pydot.write_dot(G,dot_file)
    os.system('dot -T'+fmt+' '+dot_file+' -o '+im_file)
    if show == True:
        display_ims(im_file,w)
    return im_file

def html_call(fn,w='2in'):
    return "<img src='"+fn+"' style='width:"+w+"'>"

def display_images(Files, w='2in'):
    if isinstance(Files,str):
        Files = [Files]
    html = "</td><td>".join([html_call(file,w) for file in Files])
    display(ipd.HTML("<table><tr><td>"+html+"</td></tr></table>"))
