import torch
import torch.nn as nn

from .common import SpatioTemporalEmbedding
# from .compare_backbones import SegRNN, FreTS, ModernTCN, MultiPatchFormer, ConvTimeNet
from .compare_backbones import SegRNN, FreTS, ModernTCN,ConvTimeNet


class BaseWrappedCompareModel(nn.Module):
    def __init__(
        self,
        spatial_vocab_size: int,
        lookback: int,
        embed_output_dim: int = 32,
        graphemb: int = 1,
        kd_dim: int = 32,
        chunk_size: int = 512,
    ) -> None:
        super().__init__()
        self.use_graph_emb = bool(graphemb)
        self.lookback = lookback
        self.embed_output_dim = embed_output_dim
        self.chunk_size = chunk_size

        self.st_embedding = (
            SpatioTemporalEmbedding(spatial_vocab_size, output_dim=embed_output_dim)
            if self.use_graph_emb
            else None
        )

        total_dim = 1 + (embed_output_dim if self.use_graph_emb else 0)
        self.kd_head = nn.Sequential(
            nn.Linear(total_dim, kd_dim),
            nn.Tanh(),
        )

    def pack_inputs(self, x_bat, x_grid, x_time, x_kin):
        if self.use_graph_emb:
            emb = self.st_embedding(x_grid, x_time, x_kin)   # [B, L, D]
            seq_all = torch.cat([x_bat, emb], dim=-1)        # [B, L, 1+D]
            extra_feat = emb.unsqueeze(1)                    # [B, 1, L, D]
        else:
            seq_all = x_bat                                  # [B, L, 1]
            extra_feat = None

        feat = x_bat.squeeze(-1).unsqueeze(1)                # [B, 1, L]
        return feat, extra_feat, seq_all

    def build_kd_feat(self, seq_all):
        pooled = seq_all.mean(dim=1)
        kd_feat = self.kd_head(pooled)
        return kd_feat


class FreTSRegressor(BaseWrappedCompareModel):
    def __init__(self, spatial_vocab_size, lookback, embed_output_dim=32, graphemb=1):
        super().__init__(spatial_vocab_size, lookback, embed_output_dim, graphemb)
        n_fea = 1 + (embed_output_dim if self.use_graph_emb else 0)
        self.backbone = FreTS(
            seq_len=lookback,
            pred_len=1,
            n_fea=n_fea,
            embed_size=128,
            hidden_size=256,
        )
        # self.backbone.chunk_size = chunk_size

    def forward(self, x_bat, x_grid, x_time, x_kin):
        feat, extra_feat, seq_all = self.pack_inputs(x_bat, x_grid, x_time, x_kin)
        out = self.backbone(feat, extra_feat)
        pred = out[:, -1].unsqueeze(-1)
        kd_feat = self.build_kd_feat(seq_all)
        return pred, kd_feat



class SegRNNRegressor(BaseWrappedCompareModel):
    def __init__(self, spatial_vocab_size, lookback, embed_output_dim=32, graphemb=1):
        super().__init__(spatial_vocab_size, lookback, embed_output_dim, graphemb)
        n_fea = 1 + (embed_output_dim if self.use_graph_emb else 0)
        self.backbone = SegRNN(seq_len=lookback, pred_len=1, n_fea=n_fea, seg_len=1, d_model=256, dropout=0.1)

    def forward(self, x_bat, x_grid, x_time, x_kin):
        feat, extra_feat, seq_all = self.pack_inputs(x_bat, x_grid, x_time, x_kin)
        out = self.backbone(feat, extra_feat)
        pred = out[:, -1].unsqueeze(-1)
        kd_feat = self.build_kd_feat(seq_all)
        return pred, kd_feat


class ModernTCNRegressor(BaseWrappedCompareModel):
    def __init__(self, spatial_vocab_size, lookback, embed_output_dim=32, graphemb=1):
        super().__init__(spatial_vocab_size, lookback, embed_output_dim, graphemb)
        n_fea = 1 + (embed_output_dim if self.use_graph_emb else 0)
        self.backbone = ModernTCN(seq_len=lookback, n_fea=n_fea, pred_len=1)

    def forward(self, x_bat, x_grid, x_time, x_kin):
        feat, extra_feat, seq_all = self.pack_inputs(x_bat, x_grid, x_time, x_kin)
        out = self.backbone(feat, extra_feat)
        pred = out[:, -1].unsqueeze(-1)
        kd_feat = self.build_kd_feat(seq_all)
        return pred, kd_feat


class ConvTimeNetRegressor(BaseWrappedCompareModel):
    def __init__(self, spatial_vocab_size, lookback, embed_output_dim=32, graphemb=1):
        super().__init__(spatial_vocab_size, lookback, embed_output_dim, graphemb)
        n_fea = 1 + (embed_output_dim if self.use_graph_emb else 0)
        self.backbone = ConvTimeNet(c_in=n_fea, c_out=1, seq_len=lookback)

    def forward(self, x_bat, x_grid, x_time, x_kin):
        feat, extra_feat, seq_all = self.pack_inputs(x_bat, x_grid, x_time, x_kin)
        out = self.backbone(feat, extra_feat)
        pred = out[:, -1].unsqueeze(-1)
        kd_feat = self.build_kd_feat(seq_all)
        return pred, kd_feat

#
# class MultiPatchFormerRegressor(BaseWrappedCompareModel):
#     """
#     MultiPatchFormer 原版在 pred_len=1 时，内部很多 pred_len // 8 会变成 0，
#     因此这里采用内部预测长度最少为 8，然后只取最后一步，适配你的单步电压预测任务。
#     """
#     def __init__(self, spatial_vocab_size, lookback, embed_output_dim=32, graphemb=1):
#         super().__init__(spatial_vocab_size, lookback, embed_output_dim, graphemb)
#         n_fea = 1 + (embed_output_dim if self.use_graph_emb else 0)
#         self.internal_pred_len = 8
#         self.backbone = MultiPatchFormer(
#             seq_len=lookback,
#             pred_len=self.internal_pred_len,
#             n_fea=n_fea,
#             e_layers=2,
#             d_model=256,
#             d_ff=1024,
#             n_heads=8,
#             dropout=0.1,
#         )
#
#     def forward(self, x_bat, x_grid, x_time, x_kin):
#         feat, extra_feat, seq_all = self.pack_inputs(x_bat, x_grid, x_time, x_kin)
#         out = self.backbone(feat, extra_feat)     # [B, 1]
#         pred = out[:, -1].unsqueeze(-1)
#         kd_feat = self.build_kd_feat(seq_all)
#         return pred, kd_feat
