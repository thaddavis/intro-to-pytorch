import copy
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

np.set_printoptions(precision=2)

dlc = dict(dlblue = '#0096ff', dlorange = '#FF9300', dldarkred='#C00000', dlmagenta='#FF40FF', dlpurple='#7030A0')
dlblue = '#0096ff'; dlorange = '#FF9300'; dldarkred='#C00000'; dlmagenta='#FF40FF'; dlpurple='#7030A0'
dlcolors = [dlblue, dlorange, dldarkred, dlmagenta, dlpurple]
plt.style.use('./intro_to_pytorch.mplstyle')

def sigmoid(z):
    z = np.clip( z, -500, 500 )
    g = 1.0/(1.0+np.exp(-z))

    return g

def log_1pexp(x, maximum=20):
    out  = np.zeros_like(x,dtype=float)
    i    = x <= maximum
    ni   = np.logical_not(i)

    out[i]  = np.log(1 + np.exp(x[i]))
    out[ni] = x[ni]
    return out


def compute_cost_matrix(X, y, w, b, logistic=False, lambda_=0, safe=True):
    m = X.shape[0]
    y = y.reshape(-1,1)
    w = w.reshape(-1,1)
    if logistic:
        if safe:
            z = X @ w + b
            cost = -(y * z) + log_1pexp(z)
            cost = np.sum(cost)/m
        else:
            f    = sigmoid(X @ w + b)
            cost = (1/m)*(np.dot(-y.T, np.log(f)) - np.dot((1-y).T, np.log(1-f)))
            cost = cost[0,0]
    else:
        f    = X @ w + b
        cost = (1/(2*m)) * np.sum((f - y)**2)

    reg_cost = (lambda_/(2*m)) * np.sum(w**2)

    total_cost = cost + reg_cost

    return total_cost

def compute_gradient_matrix(X, y, w, b, logistic=False, lambda_=0):
    m = X.shape[0]
    y = y.reshape(-1,1)
    w = w.reshape(-1,1)

    f_wb  = sigmoid( X @ w + b ) if logistic else  X @ w + b
    err   = f_wb - y
    dj_dw = (1/m) * (X.T @ err)
    dj_db = (1/m) * np.sum(err)

    dj_dw += (lambda_/m) * w

    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, alpha, num_iters, logistic=False, lambda_=0, verbose=True):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    w = w.reshape(-1,1)
    y = y.reshape(-1,1)

    for i in range(num_iters):
        dj_db,dj_dw = compute_gradient_matrix(X, y, w, b, logistic, lambda_)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i<100000:
            J_history.append( compute_cost_matrix(X, y, w, b, logistic, lambda_) )

        if i% math.ceil(num_iters / 10) == 0:
            if verbose: print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")

    return w.reshape(w_in.shape), b, J_history

def plot_data(X, y, ax, pos_label="y=1", neg_label="y=0", s=80, loc='best' ):
    pos = y == 1
    neg = y == 0
    pos = pos.reshape(-1,)
    neg = neg.reshape(-1,)

    ax.scatter(X[pos, 0], X[pos, 1], marker='x', s=s, c = 'red', label=pos_label)
    ax.scatter(X[neg, 0], X[neg, 1], marker='o', s=s, label=neg_label, facecolors='none', edgecolors=dlblue, lw=3)
    ax.legend(loc=loc)

    ax.figure.canvas.toolbar_visible = False
    ax.figure.canvas.header_visible = False
    ax.figure.canvas.footer_visible = False

def plt_data(x, y, ax):
    pos = y == 1
    neg = y == 0

    ax.scatter(x[pos], y[pos], marker='x', s=80, c = 'red', label="Cool")
    ax.scatter(x[neg], y[neg], marker='o', s=100, label="Wack", facecolors='none', edgecolors=dlblue,lw=3)
    ax.set_ylim(-0.175,1.1)
    ax.set_ylabel('y')
    ax.set_xlabel('Number line')
    ax.set_title("Cool Numbers")

    ax.figure.canvas.toolbar_visible = False
    ax.figure.canvas.header_visible = False
    ax.figure.canvas.footer_visible = False

def draw_threshold(ax,x):
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    ax.fill_between([xlim[0], x], [ylim[1], ylim[1]], alpha=0.2, color='white')
    ax.fill_between([x, xlim[1]], [ylim[1], ylim[1]], alpha=0.2, color=dldarkred)
    ax.annotate(">= 0.5", xy= [x,0.5], xycoords='data',
                xytext=[30,5],textcoords='offset points')
    d = FancyArrowPatch(
        posA=(x, 0.5), posB=(x+3, 0.5), color=dldarkred,
        arrowstyle='simple, head_width=5, head_length=10, tail_width=0.0',
    )
    ax.add_artist(d)
    ax.annotate("< 0.5", xy= [x,0.5], xycoords='data',
                 xytext=[-50,5],textcoords='offset points', ha='left')
    f = FancyArrowPatch(
        posA=(x, 0.5), posB=(x-3, 0.5), color=dlblue,
        arrowstyle='simple, head_width=5, head_length=10, tail_width=0.0',
    )
    ax.add_artist(f)
