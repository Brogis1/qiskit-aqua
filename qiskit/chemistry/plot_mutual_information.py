# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# for plots

from __future__ import division
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def plot_graph_mutual_info(list_edges_weights, max_width=20, path=None):
    """
    Plots the orbital diagram
    :param list_edges_weights:
    :param max_width:
    :return:
    """
    G = nx.Graph()
    size = len(list_edges_weights)

    for i in range(size):
        G.add_edge(list_edges_weights[i][0], list_edges_weights[i][1], weight=list_edges_weights[i][2])

    weights = []
    for i in range(size):
        weights.append(list_edges_weights[i][2])

    max_weight = max(weights)

    # widths of edges for all the points
    widths = []
    for i in range(size):
        widths.append(weights[i] / max_weight * max_width)

    edges = []
    for i in range(size):
        edges.append([(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] == weights[i]])

    # elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 1]
    # esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 1]

    pos = nx.circular_layout(G)  # positions for all nodes
    # pos = nx.shell_layout(G)

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)

    for i in range(size):
        nx.draw_networkx_edges(G, pos, edgelist=edges[i],
                               width=widths[i])

    # labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')

    plt.axis('off')
    plt.show()
    if path is not None:
        plt.savefig(path)
    return 0

def plot_graph_mutual_and_sinle_orb_info(list_edges_weights, list_orb_entr,  max_width=19, max_node_size=2000, shift_width = 1,
                                         shift_node_size = 200, coloring= True, threshold= None, path=None):
    """
    Plots the orbital diagram with node size proportional to the
    :param list_edges_weights:
    :param max_width:
    :return:
    """
    G = nx.Graph()
    size = len(list_edges_weights)

    for i in range(size):
        G.add_edge(list_edges_weights[i][0], list_edges_weights[i][1], weight=list_edges_weights[i][2])

    # nodes
    # widths of edges for all the points
    entropies = []
    size_entr = len(list_orb_entr)
    for i in range(size_entr):
        entropies.append(list_orb_entr[i])
    max_size = max(entropies)
    node_sizes = []
    for i in range(size_entr):
        node_sizes.append(entropies[i] / max_size * max_node_size + shift_node_size)

    # edges
    # widths of edges for all the points
    weights = []
    for i in range(size):
        weights.append(list_edges_weights[i][2])
    max_weight = max(weights)

    if threshold == None:
        # average of weights is the threshold
        sum = 0
        n= 0
        for i in weights:
            sum += i
            n+=1
        threshold = sum / n

    widths = []
    for i in range(size):
        widths.append(weights[i] / max_weight * max_width + shift_width)
    edges = []
    edges_colors = []
    for i in range(size):
        edges.append([(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] == weights[i]])
        if weights[i] > threshold:
            edges_colors.append('black')
        else:
            edges_colors.append('lime')




    pos = nx.circular_layout(G)  # positions for all nodes
    # pos = nx.spring_layout(G)  # positions for all nodes

    # nodes
    nodes=list(G)
    print(nodes)

    for i in range(size):
        nx.draw_networkx_edges(G, pos, edgelist=edges[i],
                               width=widths[i], edge_color=edges_colors[i])

    for i in range(len(nodes)):
        print(i)
        nx.draw_networkx_nodes(G, pos, nodelist=[nodes[i]],  node_size=node_sizes[i], node_color='red')

    # labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')

    plt.axis('off')
    plt.show()
    if path is not None:
        plt.savefig(path)

    return 0


def plot_orb_entropy(list_orb_entr, path):
    # plot orbital energy
    bars = []
    for i in range(len(list_orb_entr)):
        bars.append(i)
    bars = tuple(bars)
    # bars = ('A', 'B', 'C', 'D')
    y_pos = np.arange(len(bars))

    # Create bars
    plt.bar(y_pos, list_orb_entr)

    # Create names on the x-axis
    plt.xticks(y_pos, bars)

    # Show graphic
    plt.show()
    if path is not None:
        plt.savefig(path)
    # print(total_entr, list_orb_entr)
    return plt


plot_graph =True
if plot_graph:

    # Author: Rodrigo Dorantes-Gilardi (rodgdor@gmail.com)

    # G = nx.generators.directed.random_k_out_graph(4, 3, 0.5)
    # G = G.to_undirected()
    # pos = nx.layout.spring_layout(G)
    #
    # node_sizes = [3 + 10 * i for i in range(len(G))]
    # M = G.number_of_edges()
    # edge_colors = range(5, M + 5)
    #
    # edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
    #
    # nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='red')
    # edges = nx.draw_networkx_edges(G, pos, node_size=node_sizes, edge_color=edge_colors,
    #                                edge_cmap=plt.cm.Blues, width=2)
    # # # set alpha value for each edge
    # # for i in range(M):
    # #     edges[i].set_alpha(edge_alphas[i])
    # #
    # # pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Blues)
    # # pc.set_array(edge_colors)
    # # plt.colorbar(pc)
    #
    # ax = plt.gca()
    # ax.set_axis_off()
    # plt.show()

    #################################################

    # Author: Aric Hagberg (hagberg@lanl.gov)

    # import matplotlib.pyplot as plt
    # import networkx as nx
    #
    # G = nx.Graph()
    #
    # G.add_edge('a', 'b', weight=1)
    # G.add_edge('a', 'c', weight=1)
    # G.add_edge('c', 'b', weight=10)
    #
    # elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 1]
    # esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 1]
    #
    # pos = nx.spring_layout(G)  # positions for all nodes
    #
    # # nodes
    # nx.draw_networkx_nodes(G, pos, node_size=700)
    # # edges
    # nx.draw_networkx_edges(G, pos, edgelist=elarge,
    #                        width=10)
    # nx.draw_networkx_edges(G, pos, edgelist=esmall,
    #                        width=6, alpha=0.5, edge_color='b', style='dashed')
    #
    # # labels
    # nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')
    #
    # plt.axis('off')
    # plt.show()


    # plot_graph_mutual_info([[0,1,2],[1,2,5],[2, 1, 6], [0,2,20], [2,3,2]])
    plot_graph_mutual_and_sinle_orb_info([[0,1,2,2],[1,2,5,1],[2, 1, 15,7], [0,2,20,7], [2,3,2,10], [2,4,4,12]], [11,2,5,4,6])