import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import savemat

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def get_filtered_metrics(df, models_to_show, focus_submodel="student", focus_split="Test"):
    """
    严格筛选：
      - 指定模型
      - 指定子模型（teacher / student / baseline）
      - 指定数据划分（Train / Validation / Test）
    """
    metric_cols = ["MAE", "MAPE", "MSE", "RMSE", "R2"]

    df_plot = df[
        (df["SubModel"].astype(str).str.lower() == str(focus_submodel).lower()) &
        (df["Split"].astype(str).str.lower() == str(focus_split).lower()) &
        (df["ModelFamily"].isin(models_to_show))
    ][["ModelFamily"] + metric_cols].copy()

    df_plot["ModelFamily"] = pd.Categorical(
        df_plot["ModelFamily"],
        categories=models_to_show,
        ordered=True
    )
    df_plot = df_plot.sort_values("ModelFamily").reset_index(drop=True)

    return df_plot


def load_model_curves(root_dir, models_to_show, submodel="student", split="Test"):
    """
    读取多个模型的 trues / preds npy
    返回:
        true_curve: 统一真实值（取第一个存在的）
        pred_dict: {model_name: pred_curve}
        trues_dict: {model_name: true_curve}
    """
    root_dir = Path(root_dir)
    pred_dict = {}
    trues_dict = {}
    true_curve = None

    for model_name in models_to_show:
        base_dir = root_dir / f"plots_{model_name}" / submodel / "voltage"
        trues_path = base_dir / f"{split.lower()}_trues.npy"
        preds_path = base_dir / f"{split.lower()}_preds.npy"

        if trues_path.exists() and preds_path.exists():
            trues = np.load(trues_path).reshape(-1)
            preds = np.load(preds_path).reshape(-1)

            if true_curve is None:
                true_curve = trues.copy()

            min_len = min(len(trues), len(preds))
            trues = trues[:min_len]
            preds = preds[:min_len]

            pred_dict[model_name] = preds
            trues_dict[model_name] = trues

            if true_curve is not None:
                true_curve = true_curve[:min_len]

    if true_curve is None or len(pred_dict) == 0:
        raise FileNotFoundError("没有找到可用于绘图的 trues/preds npy 文件。")

    global_min_len = min([len(true_curve)] + [len(v) for v in pred_dict.values()])
    true_curve = true_curve[:global_min_len]
    for k in list(pred_dict.keys()):
        pred_dict[k] = pred_dict[k][:global_min_len]
        trues_dict[k] = trues_dict[k][:global_min_len]

    return true_curve, pred_dict, trues_dict


def export_model_mats(root_dir, models_to_show, trues_dict, pred_dict, df_plot,
                      focus_submodel="student", focus_split="Test"):
    """
    每个模型一个 .mat，里面包含：
      - trues / preds / error
      - 该模型对应的原始指标（MAE/MAPE/MSE/RMSE/R2）
    """
    root_dir = Path(root_dir)
    mat_dir = root_dir / "mat_exports"
    mat_dir.mkdir(parents=True, exist_ok=True)

    metric_cols = ["MAE", "MAPE", "MSE", "RMSE", "R2"]

    for model_name in models_to_show:
        if model_name not in pred_dict or model_name not in trues_dict:
            continue

        row = df_plot[df_plot["ModelFamily"] == model_name]
        metric_data = {}
        if len(row) > 0:
            for col in metric_cols:
                metric_data[col] = float(row.iloc[0][col])

        mat_data = {
            "model_name": np.array([model_name], dtype=object),
            "submodel": np.array([focus_submodel], dtype=object),
            "split": np.array([focus_split], dtype=object),
            "trues": trues_dict[model_name].reshape(-1, 1),
            "preds": pred_dict[model_name].reshape(-1, 1),
            "error": (pred_dict[model_name] - trues_dict[model_name]).reshape(-1, 1),
            "metric_names": np.array(metric_cols, dtype=object).reshape(-1, 1),
        }

        for k, v in metric_data.items():
            mat_data[k] = np.array([[v]], dtype=float)

        save_path = mat_dir / f"{model_name}_{focus_submodel}_{focus_split}.mat"
        savemat(str(save_path), mat_data)

    print(f"各模型 .mat 文件已保存到: {mat_dir}")


def export_combined_mat(root_dir, models_to_show, true_curve, pred_dict, trues_dict, df_plot,
                        focus_submodel="student", focus_split="Test"):
    """
    当前画图对应的一个总 .mat：
      - shared_true_curve
      - 所有模型的 trues/preds/error
      - 当前蜘蛛图原始指标表
    """
    root_dir = Path(root_dir)
    mat_dir = root_dir / "mat_exports"
    mat_dir.mkdir(parents=True, exist_ok=True)

    metric_cols = ["MAE", "MAPE", "MSE", "RMSE", "R2"]

    mat_data = {
        "shared_true_curve": true_curve.reshape(-1, 1),
        "submodel": np.array([focus_submodel], dtype=object),
        "split": np.array([focus_split], dtype=object),
        "model_names": np.array(models_to_show, dtype=object).reshape(-1, 1),
        "metric_names": np.array(metric_cols, dtype=object).reshape(-1, 1),
        "metrics_table": df_plot[metric_cols].to_numpy(dtype=float),
        "metrics_model_order": np.array(df_plot["ModelFamily"].astype(str).tolist(), dtype=object).reshape(-1, 1),
    }

    for model_name in models_to_show:
        safe_name = model_name.replace("-", "_")
        if model_name in pred_dict:
            mat_data[f"{safe_name}_preds"] = pred_dict[model_name].reshape(-1, 1)
        if model_name in trues_dict:
            mat_data[f"{safe_name}_trues"] = trues_dict[model_name].reshape(-1, 1)
        if model_name in pred_dict and model_name in trues_dict:
            mat_data[f"{safe_name}_error"] = (pred_dict[model_name] - trues_dict[model_name]).reshape(-1, 1)

    save_path = mat_dir / f"all_models_{focus_submodel}_{focus_split}.mat"
    savemat(str(save_path), mat_data)
    print(f"总 .mat 文件已保存到: {save_path}")


def build_axis_ranges(df_plot, metric_cols, padding_ratio=0.08):
    """
    为每个指标构建独立坐标范围。
    返回:
        axis_ranges = {
            'MAE': (min_val, max_val, larger_better_bool),
            ...
        }
    """
    axis_ranges = {}
    for col in metric_cols:
        vals = df_plot[col].astype(float).values
        vmin = np.min(vals)
        vmax = np.max(vals)

        if np.isclose(vmin, vmax):
            pad = 0.1 * (abs(vmin) + 1.0)
            vmin -= pad
            vmax += pad
        else:
            pad = (vmax - vmin) * padding_ratio
            vmin -= pad
            vmax += pad

        # R2 越大越好，其他误差指标越小越好
        larger_better = True if col == "R2" else False
        axis_ranges[col] = (vmin, vmax, larger_better)

    return axis_ranges


def value_to_radius(value, vmin, vmax, larger_better):
    """
    将原始值映射到 [0,1] 的半径，但图上的刻度仍显示原始值。
    """
    if np.isclose(vmax, vmin):
        return 0.5

    if larger_better:
        r = (value - vmin) / (vmax - vmin)
    else:
        r = (vmax - value) / (vmax - vmin)

    return float(np.clip(r, 0.0, 1.0))


def draw_raw_spider(ax, df_plot, metric_cols, color_map, highlight_model=None, title="Metrics Radar (Raw Scale)"):
    """
    自定义蜘蛛图：
      - 每个指标用自己的原始刻度
      - 内部按各自范围映射到半径
      - 不是统一全局归一化
    """
    ax.set_aspect('equal')
    ax.axis('off')

    n_axes = len(metric_cols)
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, n_axes, endpoint=False)

    # 每个指标独立范围
    axis_ranges = build_axis_ranges(df_plot, metric_cols)

    # 画多层网格
    grid_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
    for lvl in grid_levels:
        xs = [lvl * np.cos(a) for a in angles] + [lvl * np.cos(angles[0])]
        ys = [lvl * np.sin(a) for a in angles] + [lvl * np.sin(angles[0])]
        ax.plot(xs, ys, color="#cccccc", lw=0.8, alpha=0.8, zorder=0)

    # 画轴线
    for i, metric in enumerate(metric_cols):
        a = angles[i]
        ax.plot([0, np.cos(a)], [0, np.sin(a)], color="#bbbbbb", lw=1.0, zorder=0)

        # 轴标题
        ax.text(
            1.18 * np.cos(a),
            1.18 * np.sin(a),
            metric,
            ha='center',
            va='center',
            fontsize=12,
            fontweight='bold'
        )

        # 该轴原始刻度
        vmin, vmax, larger_better = axis_ranges[metric]
        tick_vals = np.linspace(vmin, vmax, 5)

        for tv in tick_vals:
            rr = value_to_radius(tv, vmin, vmax, larger_better)
            tx = rr * np.cos(a)
            ty = rr * np.sin(a)

            # 刻度标签略微偏移
            dx = 0.04 * np.cos(a)
            dy = 0.04 * np.sin(a)

            ax.text(
                tx + dx,
                ty + dy,
                f"{tv:.3g}",
                fontsize=9,
                color="red",
                ha='center',
                va='center'
            )

    # 画模型曲线
    for _, row in df_plot.iterrows():
        model_name = row["ModelFamily"]
        radii = []
        for metric in metric_cols:
            vmin, vmax, larger_better = axis_ranges[metric]
            rr = value_to_radius(float(row[metric]), vmin, vmax, larger_better)
            radii.append(rr)

        xs = [r * np.cos(a) for r, a in zip(radii, angles)]
        ys = [r * np.sin(a) for r, a in zip(radii, angles)]
        xs.append(xs[0])
        ys.append(ys[0])

        lw = 2.4 if model_name == highlight_model else 1.6
        alpha = 0.98 if model_name == highlight_model else 0.9

        ax.plot(
            xs, ys,
            linewidth=lw,
            color=color_map.get(model_name, None),
            alpha=alpha,
            label=model_name
        )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)


def draw_dashboard(
    root_dir="Lookback_10",
    models_to_show=None,
    focus_submodel="student",
    focus_split="Test",
    save_name=None,
    max_points=400,
    highlight_model="ModernTCN",
):
    root_dir = Path(root_dir)
    csv_path = root_dir / "results_summary.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"没找到结果表: {csv_path}")

    if models_to_show is None:
        models_to_show = ["LSTM", "GRU", "BiGRU", "FreTS", "SegRNN", "ModernTCN", "ConvTimeNet"]

    df = pd.read_csv(csv_path)

    # ---------------- 右边雷达图数据：严格筛选 student + Test ----------------
    metric_cols = ["MAE", "MAPE", "MSE", "RMSE", "R2"]

    df_plot = get_filtered_metrics(
        df=df,
        models_to_show=models_to_show,
        focus_submodel=focus_submodel,
        focus_split=focus_split
    )

    if df_plot.empty:
        raise ValueError(
            f"results_summary.csv 中没有找到 SubModel={focus_submodel}, Split={focus_split} 的记录"
        )

    print("\n筛选后的原始指标表：")
    print(df_plot.to_string(index=False))

    filtered_csv_path = root_dir / f"filtered_metrics_{focus_submodel}_{focus_split}.csv"
    df_plot.to_csv(filtered_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n已保存筛选后的原始指标表: {filtered_csv_path}")

    # ---------------- 左边：读取多模型 npy 曲线 ----------------
    true_curve, pred_dict, trues_dict = load_model_curves(
        root_dir=root_dir,
        models_to_show=models_to_show,
        submodel=focus_submodel,
        split=focus_split
    )

    # 新增：导出 mat
    export_model_mats(
        root_dir=root_dir,
        models_to_show=models_to_show,
        trues_dict=trues_dict,
        pred_dict=pred_dict,
        df_plot=df_plot,
        focus_submodel=focus_submodel,
        focus_split=focus_split
    )

    export_combined_mat(
        root_dir=root_dir,
        models_to_show=models_to_show,
        true_curve=true_curve,
        pred_dict=pred_dict,
        trues_dict=trues_dict,
        df_plot=df_plot,
        focus_submodel=focus_submodel,
        focus_split=focus_split
    )

    if max_points is not None and len(true_curve) > max_points:
        true_curve = true_curve[-max_points:]
        for k in pred_dict:
            pred_dict[k] = pred_dict[k][-max_points:]
            trues_dict[k] = trues_dict[k][-max_points:]

    x = np.arange(len(true_curve))

    # 配色
    color_map = {
        "LSTM": "#5b9bd5",
        "GRU": "#ed7d31",
        "BiGRU": "#70ad47",
        "FreTS": "#c00000",
        "SegRNN": "#7030a0",
        "ModernTCN": "#00b0f0",
        "ConvTimeNet": "#ffc000",
    }

    # ========== 创建总图 ==========
    fig = plt.figure(figsize=(18, 10), facecolor="white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], wspace=0.22, hspace=0.18)

    # ---------------- 左上：预测结果对比 ----------------
    ax_pred = fig.add_subplot(gs[0, 0])

    ax_pred.plot(
        x, true_curve,
        linestyle="--",
        color="black",
        linewidth=1.8,
        label="Target"
    )

    for model_name in models_to_show:
        if model_name in pred_dict:
            ax_pred.plot(
                x,
                pred_dict[model_name],
                linewidth=1.4 if model_name != highlight_model else 2.2,
                color=color_map.get(model_name, None),
                alpha=0.95,
                label=model_name
            )

    ax_pred.set_title("(a) Prediction Curve Comparison", fontsize=16, fontweight="bold", pad=10)
    ax_pred.set_xlabel("Sample Index", fontsize=13)
    ax_pred.set_ylabel("Voltage", fontsize=13)
    ax_pred.grid(True, linestyle="--", alpha=0.3)
    ax_pred.legend(loc="best", fontsize=11, frameon=True)

    # ---------------- 左下：误差对比 ----------------
    ax_err = fig.add_subplot(gs[1, 0])

    for model_name in models_to_show:
        if model_name in pred_dict:
            err = pred_dict[model_name] - true_curve
            ax_err.scatter(
                x,
                err,
                s=8 if model_name != highlight_model else 10,
                color=color_map.get(model_name, None),
                alpha=0.70,
                label=model_name
            )
            ax_err.plot(
                x,
                err,
                linewidth=1.0 if model_name != highlight_model else 1.6,
                color=color_map.get(model_name, None),
                alpha=0.9
            )

    ax_err.axhline(0.0, color="gray", linestyle=":", linewidth=1.2)
    ax_err.set_title("(b) Prediction Error Comparison", fontsize=16, fontweight="bold", pad=10)
    ax_err.set_xlabel("Sample Index", fontsize=13)
    ax_err.set_ylabel("Prediction Error", fontsize=13)
    ax_err.grid(True, linestyle="--", alpha=0.3)

    # ---------------- 右边：自定义原始刻度蜘蛛图 ----------------
    ax_spider = fig.add_subplot(gs[:, 1])
    draw_raw_spider(
        ax=ax_spider,
        df_plot=df_plot,
        metric_cols=metric_cols,
        color_map=color_map,
        highlight_model=highlight_model,
        title=f"{focus_split} / {focus_submodel} Metrics Spider (Raw Scale)"
    )

    # 图例
    handles, labels = ax_spider.get_legend_handles_labels()
    if handles:
        legend = ax_spider.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=2,
            fontsize=11,
            frameon=True
        )
        legend.get_frame().set_alpha(1.0)

    fig.suptitle(
        f"Multi-model Visualization Comparison ({focus_submodel}, {focus_split})",
        fontsize=22,
        fontweight="bold",
        y=0.98
    )

    if save_name is None:
        save_name = f"dashboard_overlay_{focus_submodel}_{focus_split}.png"

    save_path = root_dir / save_name
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"图已保存到: {save_path}")


if __name__ == "__main__":
    draw_dashboard(
        root_dir="Lookback_30",
        # root_dir="Lookback_10_h6",
        # root_dir="Lookback_10_h9",
        # root_dir="Lookback_10",
        models_to_show=["LSTM", "GRU", "BiGRU", "FreTS", "ConvTimeNet"],
        # models_to_show=["LSTM", "GRU", "BiGRU", "FreTS", "SegRNN", "ModernTCN", "ConvTimeNet"],
        focus_submodel="student",   # teacher / student / baseline
        focus_split="Test",         # Train / Validation / Test
        max_points=400,
        highlight_model="BiGRU"
    )