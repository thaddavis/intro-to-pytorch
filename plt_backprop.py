"""
plt_backprop.py
"""

import time
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button
from lab_utils_common import np, plt, dlc, sigmoid, compute_cost_matrix, gradient_descent

dlc["dllightblue"] = '#add8e6'

class plt_backprop:
    def __init__(self, x_train,y_train, w_range, b_range):
        fig = plt.figure( figsize=(10,6))
        fig.canvas.toolbar_visible = False
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        fig.set_facecolor('#ffffff')
        gs  = GridSpec(1, 2, figure=fig)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0,1])
        pos = ax1.get_position().get_points()
        h = 0.05 
        width = 0.2
        axcalc   = plt.axes([pos[1,0]-width, pos[1,1]-h, width, h])
        ax = np.array([ax0, ax1, axcalc])
        self.fig = fig
        self.ax = ax
        self.x_train = x_train
        self.y_train = y_train

        self.w = 0.
        self.b = 0.

        self.dplot = data_plot(ax[0], x_train, y_train, self.w, self.b)
        self.cplot = cost_plot(ax[1])

        self.bcalc = Button(axcalc, '\nRun backpropagation\n', color=dlc["dllightblue"])
        self.bcalc.on_clicked(self.backpropagate)

    def backpropagate(self, event):
        for it in [1, 8,16,32,64,128,256,512,1024,2048,4096]:
            w, self.b, J_hist = gradient_descent(self.x_train.reshape(-1,1), self.y_train.reshape(-1,1),
                                                 np.array(self.w).reshape(-1,1), self.b, 0.1, it,
                                                 logistic=True, lambda_=0, verbose=False)
            self.w = w[0,0]
            self.dplot.update(self.w, self.b)
            self.cplot.add_cost(J_hist)

            time.sleep(0.3)
            self.fig.canvas.draw()


class data_plot:
    def __init__(self, ax, x_train, y_train, w, b):
        self.ax = ax
        self.x_train = x_train
        self.y_train = y_train
        self.m = x_train.shape[0]
        self.w = w
        self.b = b

        self.plt_data()
        self.draw_logistic_lines(firsttime=True)
        self.mk_cost_lines(firsttime=True)

        self.ax.autoscale(enable=False)

    def plt_data(self):
        x = self.x_train
        y = self.y_train
        pos = y == 1
        neg = y == 0
        self.ax.scatter(x[pos], y[pos], marker='x', s=80, c = 'red', label="Cool")
        self.ax.scatter(x[neg], y[neg], marker='o', s=100, label="Wack", facecolors='none',
                   edgecolors=dlc["dlblue"],lw=3)
        self.ax.set_ylim(-0.175,1.1)
        self.ax.set_ylabel('y')
        self.ax.set_xlabel('Number line')
        self.ax.set_title("Backpropagation on Cool/Wack Number data")

    def update(self, w, b):
        self.w = w
        self.b = b
        self.draw_logistic_lines()
        self.mk_cost_lines()

    def draw_logistic_lines(self, firsttime=False):
        if not firsttime:
            self.aline[0].remove()
            self.bline[0].remove()
            self.alegend.remove()

        xlim  = self.ax.get_xlim()
        x_hat = np.linspace(*xlim, 30)
        y_hat = sigmoid(np.dot(x_hat.reshape(-1,1), self.w) + self.b)
        self.aline = self.ax.plot(x_hat, y_hat, color=dlc["dlblue"],
                                     label="y = sigmoid(z)")
        f_wb = np.dot(x_hat.reshape(-1,1), self.w) + self.b
        self.bline = self.ax.plot(x_hat, f_wb, color=dlc["dlorange"], lw=1,
                                     label=f"z = {np.squeeze(self.w):0.2f}x+({self.b:0.2f})")
        self.alegend = self.ax.legend(loc='upper left')

    def mk_cost_lines(self, firsttime=False):
        if not firsttime:
            for artist in self.cost_items:
                artist.remove()
        self.cost_items = []
        cstr = f"cost = (1/{self.m})*("
        ctot = 0
        label = 'cost for point'
        addedbreak = False
        for p in zip(self.x_train,self.y_train):
            f_wb_p = sigmoid(self.w*p[0]+self.b)
            c_p = compute_cost_matrix(p[0].reshape(-1,1), p[1],np.array(self.w), self.b, logistic=True, lambda_=0, safe=True)
            c_p_txt = c_p
            a = self.ax.vlines(p[0], p[1],f_wb_p, lw=3, color=dlc["dlpurple"], ls='dotted', label=label)
            label=''
            cxy = [p[0], p[1] + (f_wb_p-p[1])/2]
            b = self.ax.annotate(f'{c_p_txt:0.1f}', xy=cxy, xycoords='data',color=dlc["dlpurple"],
                        xytext=(5, 0), textcoords='offset points')
            cstr += f"{c_p_txt:0.1f} +"
            if len(cstr) > 38 and addedbreak is False:
                cstr += "\n"
                addedbreak = True
            ctot += c_p
            self.cost_items.extend((a,b))
        ctot = ctot/(len(self.x_train))
        cstr = cstr[:-1] + f") = {ctot:0.2f}"
        c = self.ax.text(0.05,0.02,cstr, transform=self.ax.transAxes, color=dlc["dlpurple"])
        self.cost_items.append(c)
class cost_plot:
    def __init__(self,ax):
        self.ax = ax
        self.ax.set_ylabel("Cost score")
        self.ax.set_xlabel("Adjustments")
        self.costs = []
        self.cline = self.ax.plot(0,0, color=dlc["dlblue"])

    def re_init(self):
        self.ax.clear()
        self.__init__(self.ax)

    def add_cost(self,J_hist):
        self.costs.extend(J_hist)
        self.cline[0].remove()
        self.cline = self.ax.plot(self.costs)
