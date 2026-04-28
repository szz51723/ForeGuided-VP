from .rnn_models import LSTMRegressor, GRURegressor, BiGRURegressor
from .wrapped_models import (
    FreTSRegressor,
    SegRNNRegressor,
    ModernTCNRegressor,
    ConvTimeNetRegressor,
    # MultiPatchFormerRegressor,
)


class PredictionModel:
    """
    电池电压预测版本的模型工厂。
    统一返回一个具有 forward(x_bat, x_grid, x_time, x_kin) -> (pred, kd_feat) 的模型。
    同时统一在工厂层设置 self.model.chunk_size，和你找到的原项目风格保持一致。
    """
    def __init__(
        self,
        spatial_vocab_size: int,
        model_name: str,
        conf: dict,
        seq_l: int,
        graphemb: int,
        chunk_size: int = 512,
    ) -> None:
        self.model_name = model_name.lower()
        common_kwargs = dict(
            spatial_vocab_size=spatial_vocab_size,
            embed_output_dim=conf["embed_output_dim"],
            graphemb=graphemb,
        )

        if self.model_name == 'lstm':
            self.model = LSTMRegressor(
                hidden_size=conf['hidden_size'],
                num_layers=conf['num_layers'],
                fc_dropout=conf['fc_dropout'],
                **common_kwargs,
            )
        elif self.model_name == 'gru':
            self.model = GRURegressor(
                hidden_size=conf['hidden_size'],
                num_layers=conf['num_layers'],
                fc_dropout=conf['fc_dropout'],
                **common_kwargs,
            )
        elif self.model_name == 'bigru':
            self.model = BiGRURegressor(
                hidden_size=conf['hidden_size'],
                num_layers=conf['num_layers'],
                fc_dropout=conf['fc_dropout'],
                **common_kwargs,
            )
        elif self.model_name == 'segrnn':
            self.model = SegRNNRegressor(lookback=seq_l, **common_kwargs)
        elif self.model_name == 'frets':
            self.model = FreTSRegressor(lookback=seq_l, **common_kwargs)
        elif self.model_name == 'moderntcn':
            self.model = ModernTCNRegressor(lookback=seq_l, **common_kwargs)
        elif self.model_name == 'convtimenet':
            self.model = ConvTimeNetRegressor(lookback=seq_l, **common_kwargs)
        # elif self.model_name in ('multipatchformer', 'mpf'):
        #     self.model = MultiPatchFormerRegressor(lookback=seq_l, **common_kwargs)
        else:
            raise ValueError(f'Unsupported model_name: {model_name}')

        self.model.chunk_size = chunk_size

    def update_chunksize(self, chunk_size: int) -> None:
        self.model.chunk_size = chunk_size


# def build_model(model_type, spatial_vocab_size, conf, lookback, use_graph, chunk_size=512):
#     return PredictionModel(
#         spatial_vocab_size=spatial_vocab_size,
#         model_name=model_type,
#         conf=conf,
#         seq_l=lookback,
#         graphemb=use_graph,
#         chunk_size=chunk_size,
#     ).model


def build_model(model_type, spatial_vocab_size, conf, lookback, use_graph, chunk_size=512):
    model_type = model_type.upper()

    common_kwargs = dict(
        spatial_vocab_size=spatial_vocab_size,
        embed_output_dim=conf["embed_output_dim"],
        graphemb=use_graph,
    )

    if model_type == "LSTM":
        model = LSTMRegressor(
            hidden_size=conf["hidden_size"],
            num_layers=conf["num_layers"],
            fc_dropout=conf["fc_dropout"],
            **common_kwargs,
        )

    elif model_type == "GRU":
        model = GRURegressor(
            hidden_size=conf["hidden_size"],
            num_layers=conf["num_layers"],
            fc_dropout=conf["fc_dropout"],
            **common_kwargs,
        )

    elif model_type == "BIGRU":
        model = BiGRURegressor(
            hidden_size=conf["hidden_size"],
            num_layers=conf["num_layers"],
            fc_dropout=conf["fc_dropout"],
            **common_kwargs,
        )

    elif model_type == "FRETS":
        model = FreTSRegressor(
            lookback=lookback,
            **common_kwargs,
        )

    elif model_type == "SEGRNN":
        model = SegRNNRegressor(
            lookback=lookback,
            **common_kwargs,
        )

    elif model_type == "MODERNTCN":
        model = ModernTCNRegressor(
            lookback=lookback,
            **common_kwargs,
        )

    elif model_type == "CONVTIMENET":
        model = ConvTimeNetRegressor(
            lookback=lookback,
            **common_kwargs,
        )

    # elif model_type == "MULTIPATCHFORMER":
    #     model = MultiPatchFormerRegressor(
    #         lookback=lookback,
    #         **common_kwargs,
    #     )

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # 关键：同时给外层和backbone赋 chunk_size
    model.chunk_size = chunk_size
    if hasattr(model, "backbone"):
        model.backbone.chunk_size = chunk_size

    return model