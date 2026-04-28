import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置中文字体（如果系统支持）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 原始数据 ====================
models = ["LSTM", "GRU", "BiGRU", "FreTS", "SegRNN", "ConvTimeNet"]
metrics = ["MAE", "MAPE", "MSE", "RMSE", "R2"]

data = np.array([
    [0.202620, 0.034860, 0.071589, 0.267560, 0.978076],  # LSTM
    [0.205949, 0.035440, 0.075590, 0.274937, 0.976851],  # GRU
    [0.204775, 0.035227, 0.071720, 0.267805, 0.978036],  # BiGRU
    [0.382528, 0.065780, 0.217303, 0.466158, 0.933452],  # FreTS
    [5.466856, 0.939781, 30.91839, 5.560431, -8.468626], # SegRNN (异常)
    [0.411965, 0.070855, 0.274091, 0.523537, 0.916061],  # ConvTimeNet
])

# ==================== 建议：剔除异常模型 ====================
# 若不剔除，可将下一行注释掉，并启用后面的对数缩放方案
data = np.delete(data, 4, axis=0)          # 移除 SegRNN (索引4)
models = [m for i, m in enumerate(models) if i != 4]

# ==================== 指标正向化与归一化 ====================
# 方案：对前4个指标取倒数，R²不变。然后对每个指标进行 min-max 归一化到 [0,1]。
forward_data = data.copy()
forward_data[:, :4] = 1.0 / (forward_data[:, :4] + 1e-8)  # 加小量防止除零

norm_data = np.zeros_like(forward_data)
for j in range(len(metrics)):
    col = forward_data[:, j]
    min_val, max_val = col.min(), col.max()
    norm_data[:, j] = (col - min_val) / (max_val - min_val + 1e-8)

# ==================== 雷达图绘制 ====================
N = len(metrics)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # 闭合

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# 设置刻度标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')

# 设置 y 轴范围
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)

# 绘制每个模型
colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
for i, model in enumerate(models):
    values = norm_data[i].tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=model)
    ax.fill(angles, values, alpha=0.15, color=colors[i])

# 图例与标题
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
ax.set_title("Model Performance Comparison (Normalized, higher is better)",
             fontsize=15, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig("model_radar.png", dpi=200, bbox_inches='tight')
plt.show()