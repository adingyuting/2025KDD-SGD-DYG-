from typing import Iterable, List, Tuple

import torch
import torch.nn as nn

from layers import TensorGraphConvolution, PredictionLayer, FFTLayer, GraphConvolution, FMLPLayer


class SGDDyG(nn.Module):
    def __init__(self, time_slices, N, hidden_features, num_feature, out_features, bandwidth, tgc_dropout=0.6, fft_dropout=0.6,
                 fft=True, tensor_con=True, fft_mlp=True):
        super(SGDDyG, self).__init__()
        self.M = None
        self.N = N
        self.layers = len(hidden_features)
        self.F = [num_feature] + hidden_features
        self.time_slices = time_slices
        self.X = nn.Parameter(torch.empty(time_slices, N, num_feature))
        self.fft = fft

        if fft:
            if fft_mlp:
                self.fft_layer = FFTLayer(time_slices, num_feature, num_feature, fft_dropout)
            else:
                self.fft_layer = FMLPLayer(time_slices, num_feature, num_feature, fft_dropout)

        self.tgcs = nn.ModuleList()
        for layer in range(self.layers):
            if tensor_con:
                self.tgcs.append(TensorGraphConvolution(time_slices, self.F[layer], self.F[layer + 1], bandwidth, tgc_dropout))
            else:
                self.tgcs.append(GraphConvolution(time_slices, self.F[layer], self.F[layer + 1], tgc_dropout))
        self.activation = nn.ReLU()

        self.predict = PredictionLayer(self.F[-1], out_features)

        self.init_weight()

    def forward(self, A, edges_nodes, M, cl=True):
        X = self.X

        if self.fft:
            H = self.fft_layer(X)
        else:
            H = X

        if cl:
            H = X

        if cl and not self.fft:
            H = self.corruption(H)

        for layer, tgc in enumerate(self.tgcs):
            H = tgc(A, H, M)
            if layer is not self.layers - 1:
                H = self.activation(H)

        edge_src_nodes, edge_trg_nodes = edges_nodes
        output = self.predict(H, edge_src_nodes, edge_trg_nodes)

        return torch.squeeze(output), H

    def corruption(self, X):
        neg_X = X.clone()
        for t in range(X.shape[0]):
            perm = torch.randperm(self.N)
            neg_X[t] = X[t, perm]
        return neg_X

    def init_weight(self):
        nn.init.xavier_normal_(self.X)


class ScaleSelector(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, num_scales: int = 3):
        super().__init__()
        self.selector = nn.Sequential(
            nn.Linear(4 * embed_dim + 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_scales)
        )

    def forward(self, embeddings: torch.Tensor, edges_nodes: Tuple[torch.Tensor, torch.Tensor],
                observability_stats: torch.Tensor) -> torch.Tensor:
        edge_src_nodes, edge_trg_nodes = edges_nodes
        flattened_embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        src_nodes_features = flattened_embeddings[edge_src_nodes]
        trg_nodes_features = flattened_embeddings[edge_trg_nodes]

        pair_embedding = torch.cat(
            (
                src_nodes_features,
                trg_nodes_features,
                src_nodes_features * trg_nodes_features,
                torch.abs(src_nodes_features - trg_nodes_features),
            ),
            dim=1
        )
        context = torch.cat((pair_embedding, observability_stats), dim=1)
        return self.selector(context)


class MultiScaleSGDDyG(nn.Module):
    def __init__(self, encoder: SGDDyG, selector_hidden: int, time_slices: int, num_scales: int = 3,
                 prior_beta: float = 1.0):
        super().__init__()
        self.encoder = encoder
        self.num_scales = num_scales
        self.prior_beta = prior_beta
        self.scale_selector = ScaleSelector(embed_dim=self.encoder.F[-1], hidden_dim=selector_hidden,
                                            num_scales=num_scales)
        self.timeslot_prior = nn.Parameter(torch.zeros(time_slices, num_scales))

    def forward(self, multi_scale_adj: Iterable[List[torch.Tensor]], edges_nodes, edge_times, M, observability_stats,
                cl=True):
        scale_outputs = []
        short_embeddings = None
        for adj in multi_scale_adj:
            output, embeddings = self.encoder(adj, edges_nodes, M, cl)
            scale_outputs.append(output)
            if short_embeddings is None:
                short_embeddings = embeddings
        scale_outputs = torch.stack(scale_outputs, dim=1)

        logits = self.scale_selector(short_embeddings, edges_nodes, observability_stats)
        timeslot_prior = torch.softmax(self.timeslot_prior, dim=1)
        edge_prior = timeslot_prior[edge_times]
        alpha = torch.softmax(logits + self.prior_beta * torch.log(edge_prior + 1e-8), dim=1)
        combined_output = torch.sum(alpha * scale_outputs, dim=1)

        return combined_output, short_embeddings, alpha
