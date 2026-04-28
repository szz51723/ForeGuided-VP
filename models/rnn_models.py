import torch
import torch.nn as nn

from .common import Sin, SpatioTemporalEmbedding


class BaseRNNRegressor(nn.Module):
    def __init__(
        self,
        rnn_type: str,
        spatial_vocab_size: int,
        input_size: int = 1,
        embed_output_dim: int = 32,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        graphemb: int = 1,
        fc_dropout: float = 0.3,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.use_graph_emb = bool(graphemb)
        self.bidirectional = bidirectional

        self.st_embedding = (
            SpatioTemporalEmbedding(spatial_vocab_size, output_dim=embed_output_dim)
            if self.use_graph_emb
            else None
        )

        rnn_input = input_size + embed_output_dim if self.use_graph_emb else input_size

        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                rnn_input,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=0.3 if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                rnn_input,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=0.3 if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")

        feat_dim = hidden_size * 2 if bidirectional else hidden_size
        self.fc1 = nn.Linear(feat_dim, 32)
        self.ac1 = Sin()
        self.dropout = nn.Dropout(fc_dropout)
        self.fc2 = nn.Linear(32, 32)
        self.ac2 = Sin()
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x_bat, x_grid, x_time, x_kin):
        x = (
            torch.cat([x_bat, self.st_embedding(x_grid, x_time, x_kin)], dim=-1)
            if self.use_graph_emb
            else x_bat
        )
        out, _ = self.rnn(x)
        features = out[:, -1, :]
        kd_feat = self.ac1(self.fc1(features))
        x = self.dropout(kd_feat)
        x = self.ac2(self.fc2(x))
        pred = self.fc3(x)
        return pred, kd_feat


class LSTMRegressor(BaseRNNRegressor):
    def __init__(self, *args, **kwargs):
        super().__init__("LSTM", *args, **kwargs, bidirectional=False)


class GRURegressor(BaseRNNRegressor):
    def __init__(self, *args, **kwargs):
        super().__init__("GRU", *args, **kwargs, bidirectional=False)


class BiGRURegressor(BaseRNNRegressor):
    def __init__(self, *args, **kwargs):
        super().__init__("GRU", *args, **kwargs, bidirectional=True)
