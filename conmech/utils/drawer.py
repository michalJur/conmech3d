"""
Created at 21.08.2019
@author: Michał Jureczka
@author: Piotr Bartman
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class Drawer:
    def __init__(self, state):
        self.state = state
        self.mesh = state.mesh
        self.node_size = 20 + (3000 / len(self.mesh.initial_nodes))

    def draw(self, temp_max=None, temp_min=None):
        f, ax = plt.subplots()

        if hasattr(self.state, "temperature"):
            temperature = np.concatenate(
                (self.state.temperature[:], np.zeros(self.mesh.dirichlet_count))
            )
            self.draw_field(temperature, temp_min, temp_max, ax, f)

        self.draw_mesh(
            self.mesh.initial_nodes,
            ax,
            label="Original",
            node_color="0.6",
            edge_color="0.8",
        )

        self.draw_mesh(
            self.state.displaced_points, ax, label="Deformed", node_color="k"
        )
        for contact_boundary in self.mesh.boundaries.contact:
            self.draw_boundary(
                self.state.displaced_points[contact_boundary], ax, edge_color="b"
            )
        for dirichlet_boundary in self.mesh.boundaries.dirichlet:
            self.draw_boundary(
                self.state.displaced_points[dirichlet_boundary], ax, edge_color="r"
            )
        for dirichlet_boundary in self.mesh.boundaries.neumann:
            self.draw_boundary(
                self.state.displaced_points[dirichlet_boundary], ax, edge_color="g"
            )

        # turns on axis, since networkx turn them off
        plt.axis("on")
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

        f.set_size_inches(self.mesh.scale_x * 12, self.mesh.scale_y * 10)

        plt.show()

    def draw_mesh(self, nodes, ax, label="", node_color="k", edge_color="k"):
        graph = nx.Graph()
        for i, j, k in self.mesh.cells:
            graph.add_edge(i, j)
            graph.add_edge(i, k)
            graph.add_edge(j, k)

        nx.draw(
            graph,
            pos=nodes,
            label=label,
            node_color=node_color,
            edge_color=edge_color,
            node_size=self.node_size,
            ax=ax,
        )

    def draw_boundary(self, nodes, ax, label="", node_color="k", edge_color="k"):
        graph = nx.Graph()
        for i in range(1, len(nodes)):
            graph.add_edge(i - 1, i)

        nx.draw(
            graph,
            pos=nodes,
            label=label,
            node_color=node_color,
            edge_color=edge_color,
            node_size=self.node_size,
            ax=ax,
            width=6,
        )

    def draw_field(self, field, v_min, v_max, ax, f):
        x = self.state.displaced_points[:, 0]
        y = self.state.displaced_points[:, 1]

        n_layers = 100
        ax.tricontourf(
            x,
            y,
            self.mesh.cells,
            field,
            n_layers,
            cmap=plt.cm.magma,
            vmin=v_min,
            vmax=v_max,
        )

        # cbar_ax = f.add_axes([0.875, 0.15, 0.025, 0.6])
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.magma, norm=plt.Normalize(vmin=v_min, vmax=v_max)
        )
        sm.set_array([])
        f.colorbar(sm)

