from ctypes import ArgumentError
from typing import List, Optional

import torch
from torch import nn
from torch.nn import Parameter
from torch_geometric.data.batch import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter_sum

from deep_conmech.data.dataset_statistics import DatasetStatistics, FeaturesStatistics
from deep_conmech.helpers import thh
from deep_conmech.scene.scene_input import SceneInput
from deep_conmech.training_config import TrainingData


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias, activation, dropout_rate):
        super().__init__()

        layers = []
        layers.append(nn.Linear(in_channels, out_channels, bias=bias))

        # if batch_norm:  # check also after ReLU
        #    layers.append(nn.BatchNorm1d(out_channels))

        if activation:
            layers.append(activation)

        if dropout_rate:
            layers.append(nn.Dropout(dropout_rate))

        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        output = self.blocks(x)
        return output


class ResidualBlock(nn.Module):
    class InternalResidualBlock(nn.Module):
        def __init__(self, channels, activation, dropout_rate):
            super().__init__()

            layers = []
            layers.append(nn.Linear(channels, channels))
            # if batch_norm:  # check also after ReLU
            #    layers.append(nn.BatchNorm1d(channels))

            layers.append(activation)

            if dropout_rate:
                layers.append(nn.Dropout(dropout_rate))

            self.blocks = nn.Sequential(*layers)

        def forward(self, x):
            output = self.blocks(x)
            return output

    def __init__(self, channels, activation, dropout_rate, skip):
        super().__init__()
        self.skip = skip

        self.blocks = nn.Sequential(
            self.InternalResidualBlock(
                channels=channels,
                # batch_norm=batch_norm,
                activation=activation,
                dropout_rate=dropout_rate,
            ),
            self.InternalResidualBlock(
                channels=channels,
                # batch_norm=batch_norm,
                activation=activation,
                dropout_rate=False,
            ),
        )

    def forward(self, x):
        output = self.blocks(x)
        if self.skip:
            output = x + output  # += not working on newer torch versions
        return output


class DataNorm(nn.Module):
    def __init__(self, in_channels, statistics: FeaturesStatistics):
        super().__init__()
        self.in_channels = in_channels
        self.register_buffer("x_mean", statistics.data_mean)
        self.register_buffer("x_std", statistics.data_std)
        self.register_buffer("mask", statistics.data_std == 0)

    def forward(self, x):
        output = (x - self.x_mean) / self.x_std
        output = torch.nan_to_num(output)
        return output


class ForwardNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        layers_count: int,
        output_linear_dim: int,
        statistics: Optional[FeaturesStatistics],
        batch_norm: bool,
        layer_norm: bool,
        td: TrainingData,
    ):
        super().__init__()
        layers = []

        self.statistics = statistics
        if batch_norm:
            layers.append(nn.BatchNorm1d(input_dim))
            if statistics is not None:
                raise ArgumentError()
        else:
            if statistics is not None:
                layers.append(DataNorm(in_channels=input_dim, statistics=statistics))

        layers.append(
            BasicBlock(
                in_channels=input_dim,
                out_channels=td.latent_dimension,
                bias=True,
                # batch_norm=config.BATCH_NORM,
                activation=td.activation,
                dropout_rate=False,
            )
        )

        for _ in range(layers_count):
            layers.append(
                ResidualBlock(
                    td.latent_dimension,
                    # batch_norm=config.BATCH_NORM,
                    activation=td.activation,
                    dropout_rate=td.dropout_rate,
                    skip=td.skip_connections,
                )
            )

        layers.append(
            BasicBlock(
                in_channels=td.latent_dimension,
                out_channels=output_linear_dim,
                bias=True,
                # batch_norm=False,
                activation=False,
                dropout_rate=False,
            )
        )

        if layer_norm:
            layers.append(nn.LayerNorm(output_linear_dim))

        self.net = thh.set_precision(nn.Sequential(*layers))

    def forward(self, x):
        result = self.net(x)
        return result


class Attention(nn.Module):
    def __init__(self, td: TrainingData):
        super().__init__()

        if td.attention_heads_count is None:
            self.blocks = None
            return

        attention_heads = BasicBlock(
            in_channels=td.latent_dimension,  # * 3,
            out_channels=td.attention_heads_count,
            bias=True,
            activation=td.activation,
            dropout_rate=False,
        )

        self.blocks = (
            attention_heads
            if td.attention_heads_count == 1
            else nn.Sequential(attention_heads, nn.Linear(td.attention_heads_count, 1, bias=False))
        )

    def forward(self, edge_inputs, index):
        if self.blocks is None:
            return 1.0

        alpha_score = self.blocks(edge_inputs)
        alpha = softmax(alpha_score, index)
        # torch.sum(alpha * (index == 5).reshape(-1,1)) == 1
        return alpha


# pylint: disable=W0223, W0221
class ProcessorLayer(MessagePassing):
    def __init__(self, td: TrainingData):
        super().__init__()

        self.edge_processor = ForwardNet(
            input_dim=td.latent_dimension * 3,
            layers_count=td.processor_layers_count,
            output_linear_dim=td.latent_dimension,
            statistics=None,
            batch_norm=td.internal_batch_norm,
            layer_norm=td.layer_norm,
            td=td,
        )
        self.node_processor = ForwardNet(
            input_dim=td.latent_dimension * 2,
            layers_count=td.processor_layers_count,
            output_linear_dim=td.latent_dimension,
            statistics=None,
            batch_norm=td.internal_batch_norm,
            layer_norm=td.layer_norm,
            td=td,
        )
        self.attention = Attention(td=td)

        self.epsilon = Parameter(torch.Tensor(1))
        self.new_edge_latents = None

    def forward(self, edge_index, node_latents, edge_latents):
        new_node_latents = self.propagate(
            edge_index=edge_index, node_latents=node_latents, edge_latents=edge_latents
        )
        new_edge_latents = self.new_edge_latents
        self.new_edge_latents = None
        return new_node_latents, new_edge_latents

    def message(self, node_latents_i, node_latents_j, edge_latents):  # index
        edge_inputs = torch.hstack((node_latents_i, node_latents_j, edge_latents))
        self.new_edge_latents = edge_latents + self.edge_processor(edge_inputs)
        return self.new_edge_latents

    def aggregate(self, new_edge_latents, index):  # weighted_edge_latents
        alpha = self.attention(new_edge_latents, index)
        aggregated_edge_latents = scatter_sum(alpha * self.new_edge_latents, index, dim=0)
        return aggregated_edge_latents

    def update(self, aggregated_edge_latents, node_latents):
        to_node_latents = node_latents[-1] if isinstance(node_latents, tuple) else node_latents
        node_inputs = torch.hstack((to_node_latents, aggregated_edge_latents))
        new_node_latents = to_node_latents + self.node_processor(node_inputs)
        return new_node_latents


class CustomGraphNet(nn.Module):
    def __init__(
        self,
        statistics: Optional[DatasetStatistics],
        td: TrainingData,
    ):
        super().__init__()
        self.td = td

        self.node_encoder = ForwardNet(
            input_dim=SceneInput.get_nodes_data_dim(td.dimension),
            layers_count=td.encoder_layers_count,
            output_linear_dim=td.latent_dimension,
            statistics=None if statistics is None else statistics.nodes_statistics,
            batch_norm=td.input_batch_norm,
            layer_norm=td.layer_norm,
            td=td,
        )

        self.edge_encoder = ForwardNet(
            input_dim=SceneInput.get_edges_data_dim(td.dimension),
            layers_count=td.encoder_layers_count,
            output_linear_dim=td.latent_dimension,
            statistics=None if statistics is None else statistics.edges_statistics,
            batch_norm=td.input_batch_norm,
            layer_norm=td.layer_norm,
            td=td,
        )

        self.processor_layers = nn.ModuleList(
            [
                ProcessorLayer(td=td)
                for _ in range(td.message_passes * (td.mesh_layers_count * 2 - 1))
            ]
        )
        self.upward_processor_layer = ProcessorLayer(td=td)
        self.downward_processor_layer = ProcessorLayer(td=td)

        self.decoder = ForwardNet(
            input_dim=td.latent_dimension,
            layers_count=td.decoder_layers_count,
            output_linear_dim=td.dimension,
            statistics=None,
            batch_norm=td.internal_batch_norm,
            layer_norm=False,  # TODO #65
            td=td,
        )

        test_layers = []
        test_layers.append(nn.BatchNorm1d(10))
        test_layers.append(nn.Linear(10, 128))
        test_layers.append(nn.ReLU())
        test_layers.append(nn.Linear(128, 2))
        self.test_linear = nn.Sequential(*test_layers)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def statistics(self):
        return self.statistics

    @property
    def node_statistics(self):
        return self.node_encoder.statistics

    @property
    def edge_statistics(self):
        return self.edge_encoder.statistics

    def move_from_down(self, node_latents, edge_latents, layer):
        node_latents_to = self.node_encoder(layer.x)
        new_node_latents, _ = self.upward_processor_layer(
            edge_index=layer.edge_index_from_down,
            node_latents=(node_latents, node_latents_to),
            edge_latents=edge_latents,
        )
        return new_node_latents

    def move_to_down(self, node_latents_up, node_latents, edge_latents, layer):
        new_node_latents, _ = self.downward_processor_layer(
            edge_index=layer.edge_index_to_down,
            node_latents=(node_latents_up, node_latents),
            edge_latents=edge_latents,
        )
        # residual connection (included in processor)
        # new_node_latents = node_latents + node_latents_from_up
        return new_node_latents

    def propagate_messages(
        self, layer: Data, node_latents: torch.Tensor, edge_latents: torch.Tensor
    ):
        for _ in range(self.td.message_passes):
            node_latents, edge_latents = self.processor_layers[self.processor_number](
                layer.edge_index, node_latents, edge_latents
            )
            self.processor_number += 1
        return node_latents, edge_latents

    def process_by_layer(
        self, layer_list: List[Data], layer_number: int, node_latents: torch.Tensor
    ):
        layer = layer_list[layer_number]
        edge_latents = self.edge_encoder(layer.edge_attr)

        node_latents, edge_latents = self.propagate_messages(
            layer=layer, node_latents=node_latents, edge_latents=edge_latents
        )

        if layer_number == len(layer_list) - 1:
            return node_latents

        layer_up = layer_list[layer_number + 1]

        node_latents_up = self.move_from_down(
            node_latents=node_latents,
            edge_latents=self.edge_encoder(layer_up.edge_attr_from_down),
            layer=layer_up,
        )

        node_latents_up = self.process_by_layer(
            layer_list=layer_list,
            layer_number=layer_number + 1,
            node_latents=node_latents_up,
        )

        node_latents = self.move_to_down(
            node_latents_up=node_latents_up,
            edge_latents=self.edge_encoder(layer_up.edge_attr_to_down),
            layer=layer_up,
            node_latents=node_latents,
        )

        node_latents, edge_latents = self.propagate_messages(
            layer=layer, node_latents=node_latents, edge_latents=edge_latents
        )
        return node_latents

    def forward(self, layer_list: List[Data]):
        main_layer = layer_list[0]
        self.processor_number = 0

        # nodes = main_layer.pos
        # nodes_up = self.move_from_down(node_latents=nodes, layer=layer_list[1])
        # nodes_new = self.move_to_down(node_latents=nodes_up, layer=layer_list[1])
        # assert torch.allclose(nodes, nodes_new)

        # node_latents = self.node_encoder(main_layer.x)
        # net_output = self.decoder(node_latents)
        # net_output = self.test_linear(main_layer.x)
        # return net_output

        node_latents = self.node_encoder(main_layer.x)
        # position "pos" will not generalize
        processed_node_latents = self.process_by_layer(
            layer_list=layer_list,
            layer_number=0,
            node_latents=node_latents,
        )
        net_output = self.decoder(node_latents + processed_node_latents)  # processed_node_latents

        # TODO: #65 Include mass_density
        # main_layer.x[:,:2]
        return net_output  # main_layer.forces + net_output

    def solve_all(self, scene: SceneInput):
        self.eval()
        layers_count = len(scene.all_layers)
        layers_list = [
            scene.get_features_data(layer_number=layer_number).to(self.device)
            for layer_number in range(layers_count)
        ]
        normalized_a_cuda = self(layer_list=layers_list)

        normalized_a = thh.to_np_double(normalized_a_cuda)
        a = scene.denormalize_rotate(normalized_a)
        return a, normalized_a

    def solve(self, scene: SceneInput, initial_a):
        _ = initial_a
        a, _ = self.solve_all(scene)
        return a
