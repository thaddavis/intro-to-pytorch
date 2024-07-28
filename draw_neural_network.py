import matplotlib.pyplot as plt

def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    '''
    Draw a neural network cartoon using matplotilb.
    
    :param ax: matplotlib.axes.AxesSubplot, the axes on which to plot the cartoon (get e.g. by plt.gca())
    :param left: float, the center of the leftmost node(s) will be placed here
    :param right: float, the center of the rightmost node(s) will be placed here
    :param bottom: float, the center of the bottommost node(s) will be placed here
    :param top: float, the center of the topmost node(s) will be placed here
    :param layer_sizes: list of int, list containing the number of nodes in each layer
    '''
    
    v_spacing = (top - bottom)/float(max(layer_sizes) + 1)
    h_spacing = (right - left)/float(len(layer_sizes) - 1)

    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='lightblue', ec='k', zorder=4)
            ax.add_artist(circle)
            # Annotation
            # if layer_size > 1: 
            if n == 0:
                ax.annotate(f'Input {m+1}', (n*h_spacing + left, layer_top - m*v_spacing), 
                            textcoords="offset points", xytext=(-25,25*v_spacing/4.), ha='center', fontsize=12, color='blue')
            elif n == len(layer_sizes) - 1:
                ax.annotate(f'Output {m+1}', (n*h_spacing + left, layer_top - m*v_spacing), 
                            textcoords="offset points", xytext=(25,25*v_spacing/4.), ha='center', fontsize=12, color='blue')
            else:
                ax.annotate(f'Layer {n}-{m+1}', (n*h_spacing + left, layer_top - m*v_spacing), 
                            textcoords="offset points", xytext=(0,25*v_spacing/4.), ha='center', fontsize=12, color='blue')
    
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='gray')
                ax.add_artist(line)