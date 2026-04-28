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


def get_filtered_metrics(df, methods_to_show, focus_submodel="student", focus_split="Test"):
    """
    严格筛选：
      - 指定方法（Method = ModelFamily + ExpVariant）
      - 指定子模型（teacher / student / baseline）
      - 指定数据划分（Train / Validation / Test）
    """
    metric_cols = ["MAE", "MAPE", "MSE", "RMSE", "R2"]

    df = df.copy()
    df["Method"] = df["ModelFamily"].astype(str) + "-" + df["ExpVariant"].astype(str)

    df_plot = df[
        (df["SubModel"].astype(str).str.lower() == str(focus_submodel).lower()) &
        (df["Split"].astype(str).str.lower() == str(focus_split).lower()) &
        (df["Method"].isin(methods_to_show))
    ][["Method"] + metric_cols].copy()

    df_plot["Method"] = pd.Categorical(
        df_plot["Method"],
        categories=methods_to_show,
        ordered=True
    )
    df_plot = df_plot.sort_values("Method").reset_index(drop=True)

    return df_plot


def load_method_curves(root_dir, methods_to_show, submodel="student", split="Test"):
    """
    读取多个方法的 trues / preds npy
    method 格式:
        ModelFamily-ExpVariant
    对应目录:
        plots_LSTM_full
        plots_LSTM_wo_ig
        plots_BiGRU_wo_guidance
    """
    root_dir = Path(root_dir)
    pred_dict = {}
    true_curve = None
    trues_dict = {}

    for method_name in methods_to_show:
        if "-" not in method_name:
            continue

        model_family, exp_variant = method_name.split("-", 1)
        base_dir = root_dir / f"plots_{model_family}_{exp_variant}" / submodel / "voltage"

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

            pred_dict[method_name] = preds
            trues_dict[method_name] = trues

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


def export_method_mats(root_dir, methods_to_show, trues_dict, pred_dict, focus_submodel="student", focus_split="Test"):
    """
    导出每个方法各自对应的 .mat 文件
    """
    root_dir = Path(root_dir)
    mat_dir = root_dir / "mat_exports"
    mat_dir.mkdir(parents=True, exist_ok=True)

    for method_name in methods_to_show:
        if method_name not in pred_dict or method_name not in trues_dict:
            continue

        model_family, exp_variant = method_name.split("-", 1)

        mat_data = {
            "method_name": method_name,
            "model_family": model_family,
            "exp_variant": exp_variant,
            "submodel": focus_submodel,
            "split": focus_split,
            "trues": trues_dict[method_name].reshape(-1, 1),
            "preds": pred_dict[method_name].reshape(-1, 1),
            "error": (pred_dict[method_name] - trues_dict[method_name]).reshape(-1, 1),
        }

        save_path = mat_dir / f"{model_family}_{exp_variant}_{focus_submodel}_{focus_split}.mat"
        savemat(str(save_path), mat_data)

    print(f"各方法 .mat 文件已保存到: {mat_dir}")


def export_combined_mat(root_dir, methods_to_show, true_curve, pred_dict, trues_dict, focus_submodel="student", focus_split="Test"):
    """
    导出总的 .mat 文件，把所有方法放到一个文件里
    """
    root_dir = Path(root_dir)
    mat_dir = root_dir / "mat_exports"
    mat_dir.mkdir(parents=True, exist_ok=True)

    mat_data = {
        "shared_true_curve": true_curve.reshape(-1, 1),
        "submodel": focus_submodel,
        "split": focus_split,
    }

    for method_name in methods_to_show:
        safe_name = method_name.replace("-", "_")
        if method_name in pred_dict:
            mat_data[f"{safe_name}_preds"] = pred_dict[method_name].reshape(-1, 1)
        if method_name in trues_dict:
            mat_data[f"{safe_name}_trues"] = trues_dict[method_name].reshape(-1, 1)
        if method_name in pred_dict and method_name in trues_dict:
            mat_data[f"{safe_name}_error"] = (pred_dict[method_name] - trues_dict[method_name]).reshape(-1, 1)

    save_path = mat_dir / f"all_methods_{focus_submodel}_{focus_split}.mat"
    savemat(str(save_path), mat_data)
    print(f"总 .mat 文件已保存到: {save_path}")


def build_axis_ranges(df_plot, metric_cols, padding_ratio=0.08):
    """
    为每个指标构建独立坐标范围。
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


def draw_raw_spider(ax, df_plot, metric_cols, color_map, highlight_method=None,
                    title="Metrics Spider (Raw Scale)"):
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

    axis_ranges = build_axis_ranges(df_plot, metric_cols)

    grid_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
    for lvl in grid_levels:
        xs = [lvl * np.cos(a) for a in angles] + [lvl * np.cos(angles[0])]
        ys = [lvl * np.sin(a) for a in angles] + [lvl * np.sin(angles[0])]
        ax.plot(xs, ys, color="#cccccc", lw=0.8, alpha=0.8, zorder=0)

    for i, metric in enumerate(metric_cols):
        a = angles[i]
        ax.plot([0, np.cos(a)], [0, np.sin(a)], color="#bbbbbb", lw=1.0, zorder=0)

        ax.text(
            1.20 * np.cos(a),
            1.20 * np.sin(a),
            metric,
            ha='center',
            va='center',
            fontsize=12,
            fontweight='bold'
        )

        vmin, vmax, larger_better = axis_ranges[metric]
        tick_vals = np.linspace(vmin, vmax, 5)

        for tv in tick_vals:
            rr = value_to_radius(tv, vmin, vmax, larger_better)
            tx = rr * np.cos(a)
            ty = rr * np.sin(a)
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

    for _, row in df_plot.iterrows():
        method_name = row["Method"]
        radii = []
        for metric in metric_cols:
            vmin, vmax, larger_better = axis_ranges[metric]
            rr = value_to_radius(float(row[metric]), vmin, vmax, larger_better)
            radii.append(rr)

        xs = [r * np.cos(a) for r, a in zip(radii, angles)]
        ys = [r * np.sin(a) for r, a in zip(radii, angles)]
        xs.append(xs[0])
        ys.append(ys[0])

        lw = 2.6 if method_name == highlight_method else 1.6
        alpha = 0.98 if method_name == highlight_method else 0.9

        ax.plot(
            xs, ys,
            linewidth=lw,
            color=color_map.get(method_name, None),
            alpha=alpha,
            label=method_name
        )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)


def draw_dashboard(
    root_dir="Ablation_study_Lookback_10",
    methods_to_show=None,
    focus_submodel="student",
    focus_split="Test",
    save_name=None,
    max_points=400,
    highlight_method="BiGRU-full",
):
    root_dir = Path(root_dir)
    csv_path = root_dir / "results_summary.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"没找到结果表: {csv_path}")

    if methods_to_show is None:
        methods_to_show = [
            "LSTM-full",
            "LSTM-wo_ig",
            "LSTM-wo_guidance",
            "BiGRU-full",
            "BiGRU-wo_ig",
            "BiGRU-wo_guidance",
        ]

    df = pd.read_csv(csv_path)

    metric_cols = ["MAE", "MAPE", "MSE", "RMSE", "R2"]

    df_plot = get_filtered_metrics(
        df=df,
        methods_to_show=methods_to_show,
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

    true_curve, pred_dict, trues_dict = load_method_curves(
        root_dir=root_dir,
        methods_to_show=methods_to_show,
        submodel=focus_submodel,
        split=focus_split
    )

    # 新增：导出 mat
    export_method_mats(
        root_dir=root_dir,
        methods_to_show=methods_to_show,
        trues_dict=trues_dict,
        pred_dict=pred_dict,
        focus_submodel=focus_submodel,
        focus_split=focus_split
    )

    export_combined_mat(
        root_dir=root_dir,
        methods_to_show=methods_to_show,
        true_curve=true_curve,
        pred_dict=pred_dict,
        trues_dict=trues_dict,
        focus_submodel=focus_submodel,
        focus_split=focus_split
    )

    if max_points is not None and len(true_curve) > max_points:
        true_curve = true_curve[-max_points:]
        for k in pred_dict:
            pred_dict[k] = pred_dict[k][-max_points:]
            trues_dict[k] = trues_dict[k][-max_points:]

    x = np.arange(len(true_curve))

    color_map = {
        "LSTM-full": "#1f77b4",
        "LSTM-wo_ig": "#6baed6",
        "LSTM-wo_guidance": "#9ecae1",
        "BiGRU-full": "#d62728",
        "BiGRU-wo_ig": "#ff9896",
        "BiGRU-wo_guidance": "#fcbba1",
    }

    linestyle_map = {
        "LSTM-full": "-",
        "LSTM-wo_ig": "--",
        "LSTM-wo_guidance": ":",
        "BiGRU-full": "-",
        "BiGRU-wo_ig": "--",
        "BiGRU-wo_guidance": ":",
    }

    fig = plt.figure(figsize=(18, 10), facecolor="white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], wspace=0.22, hspace=0.18)

    # 左上：预测曲线
    ax_pred = fig.add_subplot(gs[0, 0])

    ax_pred.plot(
        x, true_curve,
        linestyle="--",
        color="black",
        linewidth=1.8,
        label="Target"
    )

    for method_name in methods_to_show:
        if method_name in pred_dict:
            ax_pred.plot(
                x,
                pred_dict[method_name],
                linewidth=2.2 if method_name == highlight_method else 1.5,
                linestyle=linestyle_map.get(method_name, "-"),
                color=color_map.get(method_name, None),
                alpha=0.95,
                label=method_name
            )

    ax_pred.set_title("(a) Prediction Curve Comparison", fontsize=16, fontweight="bold", pad=10)
    ax_pred.set_xlabel("Sample Index", fontsize=13)
    ax_pred.set_ylabel("Voltage", fontsize=13)
    ax_pred.grid(True, linestyle="--", alpha=0.3)
    ax_pred.legend(loc="best", fontsize=10, frameon=True, ncol=2)

    # 左下：误差曲线
    ax_err = fig.add_subplot(gs[1, 0])

    for method_name in methods_to_show:
        if method_name in pred_dict:
            err = pred_dict[method_name] - true_curve
            ax_err.plot(
                x,
                err,
                linewidth=1.8 if method_name == highlight_method else 1.2,
                linestyle=linestyle_map.get(method_name, "-"),
                color=color_map.get(method_name, None),
                alpha=0.9,
                label=method_name
            )

    ax_err.axhline(0.0, color="gray", linestyle=":", linewidth=1.2)
    ax_err.set_title("(b) Prediction Error Comparison", fontsize=16, fontweight="bold", pad=10)
    ax_err.set_xlabel("Sample Index", fontsize=13)
    ax_err.set_ylabel("Prediction Error", fontsize=13)
    ax_err.grid(True, linestyle="--", alpha=0.3)

    # 右边：蜘蛛图
    ax_spider = fig.add_subplot(gs[:, 1])
    draw_raw_spider(
        ax=ax_spider,
        df_plot=df_plot,
        metric_cols=metric_cols,
        color_map=color_map,
        highlight_method=highlight_method,
        title=f"{focus_split} / {focus_submodel} Metrics Spider (Raw Scale)"
    )

    handles, labels = ax_spider.get_legend_handles_labels()
    if handles:
        legend = ax_spider.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=2,
            fontsize=10,
            frameon=True
        )
        legend.get_frame().set_alpha(1.0)

    fig.suptitle(
        f"Ablation Study Comparison ({focus_submodel}, {focus_split})",
        fontsize=22,
        fontweight="bold",
        y=0.98
    )

    if save_name is None:
        save_name = f"ablation_dashboard_{focus_submodel}_{focus_split}.png"

    save_path = root_dir / save_name
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"图已保存到: {save_path}")


if __name__ == "__main__":
    draw_dashboard(
        root_dir="Ablation_study_Lookback_10",
        methods_to_show=[
            "LSTM-full",
            "LSTM-wo_ig",
            "LSTM-wo_guidance",
            "BiGRU-full",
            "BiGRU-wo_ig",
            "BiGRU-wo_guidance",
        ],
        focus_submodel="student",   # teacher / student / baseline
        focus_split="Test",         # Train / Validation / Test
        max_points=400,
        highlight_method="BiGRU-full"
    )