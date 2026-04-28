import torch
import torch.nn as nn


class Sin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


class SpatioTemporalEmbedding(nn.Module):
    def __init__(
        self,
        spatial_vocab_size: int,
        spatial_dim: int = 16,
        time_dim: int = 16,
        kinematic_dim: int = 16,
        output_dim: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.spatial_embed = nn.Sequential(
            nn.Embedding(spatial_vocab_size, spatial_dim),
            nn.Dropout(dropout),
        )
        self.time_embed = nn.Sequential(
            nn.Linear(4, time_dim),
            nn.ReLU(),
        )
        self.kinematic_embed = nn.Sequential(
            nn.Linear(2, kinematic_dim),
            nn.ReLU(),
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(spatial_dim + time_dim + kinematic_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, grid_ids, time_features, kinematic_features):
        return self.fusion_layer(
            torch.cat(
                [
                    self.spatial_embed(grid_ids),
                    self.time_embed(time_features),
                    self.kinematic_embed(kinematic_features),
                ],
                dim=-1,
            )
        )


class EarlyStopper:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, current_loss: float, model: nn.Module) -> bool:
        if current_loss + self.min_delta < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
            self.best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
