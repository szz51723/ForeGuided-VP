import argparse
import os
from pathlib import Path
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models import build_model
from models.common import EarlyStopper


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_MODEL_HPARAMS = {
    'teacher':  {'embed_output_dim': 32, 'hidden_size': 64,  'num_layers': 2, 'fc_dropout': 0.3, 'lr': 5e-4, 'weight_decay': 0.0},
    'student':  {'embed_output_dim': 32, 'hidden_size': 128, 'num_layers': 1, 'fc_dropout': 0.3, 'lr': 8e-4, 'weight_decay': 3e-4},
    'baseline': {'embed_output_dim': 32, 'hidden_size': 128, 'num_layers': 1, 'fc_dropout': 0.3, 'lr': 5e-4, 'weight_decay': 3e-4},
}


def load_and_split_data(filepath, lookback, horizon, batch_size, start_row=2):
    print(f'Loading data from {filepath}...')
    df = pd.read_excel(filepath, header=None, skiprows=max(start_row - 1, 0))
    target_cols = {'time': 0, 'speed': 2, 'status': 4, 'lon': 48, 'lat': 49, 'angle': 76, 'v': 15}

    df_ext = df.iloc[:, list(target_cols.values())].copy()
    df_ext.columns = list(target_cols.keys())
    df_ext['orig_idx'] = np.arange(len(df_ext))

    numeric_cols = ['lon', 'lat', 'speed', 'angle', 'v']
    df_ext[numeric_cols] = df_ext[numeric_cols].apply(pd.to_numeric, errors='coerce').interpolate(method='linear', limit_direction='both')

    for col in ['speed', 'angle', 'v']:
        col_data = df_ext[col].values
        mean_val, std_val = np.nanmean(col_data), np.nanstd(col_data)
        if std_val > 1e-6:
            df_ext[col] = np.clip(col_data, mean_val - 3.0 * std_val, mean_val + 3.0 * std_val)

    df_filtered = df_ext[df_ext['status'] == '未充电状态'].dropna().reset_index(drop=True)
    orig_idx_arr = df_filtered['orig_idx'].values
    print(f'Valid samples after preprocessing: {len(df_filtered)}')

    time_dt = pd.to_datetime(df_filtered['time'])
    time_feats = np.column_stack([
        np.sin(2 * np.pi * time_dt.dt.hour / 24),
        np.cos(2 * np.pi * time_dt.dt.hour / 24),
        np.sin(2 * np.pi * time_dt.dt.dayofweek / 7),
        np.cos(2 * np.pi * time_dt.dt.dayofweek / 7),
    ]).astype(np.float32)

    spatial_data = df_filtered[['lon', 'lat']].values.astype(np.float32)
    kin_data = df_filtered[['speed', 'angle']].values.astype(np.float32)
    bat_data = df_filtered[['v']].values.astype(np.float32)

    S, T, Y = [[] for _ in range(4)], [[] for _ in range(4)], []
    datas = [bat_data, spatial_data, time_feats, kin_data]

    for i in range(len(bat_data) - lookback - horizon + 1):
        target_idx = i + lookback + horizon - 1
        if orig_idx_arr[target_idx] - orig_idx_arr[i] != (lookback + horizon - 1):
            continue
        for j, d in enumerate(datas):
            S[j].append(d[i:i + lookback])
            T[j].append(d[i + horizon - 1:i + horizon - 1 + lookback])
        Y.append(bat_data[target_idx])

    S = [np.array(x) for x in S]
    T = [np.array(x) for x in T]
    Y = np.array(Y)

    n_train = (len(Y) * 15) // 20
    n_val = (len(Y) * 3) // 20

    train_spatial = np.concatenate([S[1][:n_train].reshape(-1, 2), T[1][:n_train].reshape(-1, 2)], axis=0)
    lon_min, lat_min = train_spatial.min(axis=0)
    lon_range, lat_range = np.maximum(train_spatial.ptp(axis=0), 1e-6)

    def to_grid(spatial):
        norm = np.clip((spatial - [lon_min, lat_min]) / [lon_range, lat_range], 0.0, 1.0)
        return (np.rint(norm[..., 0] * 19) * 20 + np.rint(norm[..., 1] * 19)).astype(np.int64)

    S[1], T[1] = to_grid(S[1]), to_grid(T[1])

    train_kin = np.concatenate([S[3][:n_train].reshape(-1, 2), T[3][:n_train].reshape(-1, 2)], axis=0)
    kin_m, kin_s = train_kin.mean(axis=0), np.maximum(train_kin.std(axis=0), 1e-6)
    S[3], T[3] = (S[3] - kin_m) / kin_s, (T[3] - kin_m) / kin_s

    train_bat = np.concatenate([S[0][:n_train].reshape(-1, 1), T[0][:n_train].reshape(-1, 1), Y[:n_train].reshape(-1, 1)])
    bat_m, bat_s = train_bat.mean(), max(train_bat.std(), 1e-6)
    S[0], T[0], Y = (S[0] - bat_m) / bat_s, (T[0] - bat_m) / bat_s, (Y - bat_m) / bat_s

    tensors = [torch.tensor(x, dtype=torch.long if i == 1 else torch.float32) for i, x in enumerate(S)] + \
              [torch.tensor(x, dtype=torch.long if i == 1 else torch.float32) for i, x in enumerate(T)] + \
              [torch.tensor(Y, dtype=torch.float32)]

    loaders = []
    for start, end in [(0, n_train), (n_train, n_train + n_val), (n_train + n_val, len(Y))]:
        ds = TensorDataset(*[t[start:end] for t in tensors])
        loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=(start == 0)))

    return *loaders, bat_m, bat_s, 400


def get_feature_kd_alpha(epoch, alpha_start, warmup, anneal):
    if epoch < warmup:
        return alpha_start
    if anneal == 0:
        return 0.0
    progress = (epoch - warmup + 1) / float(anneal)
    return alpha_start * 0.5 * (1.0 + np.cos(np.pi * progress)) if progress <= 1.0 else 0.0


def collect_predictions(model, loader, v_mean, v_std, is_teacher):
    model.eval()
    trues, preds = [], []
    with torch.no_grad():
        for batch in loader:
            batch = [b.to(device) for b in batch]
            inputs = batch[4:8] if is_teacher else batch[:4]
            out, _ = model(*inputs)
            trues.append(batch[-1].cpu().numpy() * v_std + v_mean)
            preds.append(out.cpu().numpy() * v_std + v_mean)
    return np.vstack(trues).astype(np.float32), np.vstack(preds).astype(np.float32)


def compute_metrics(trues, preds, eps=1e-8):
    trues = np.asarray(trues).reshape(-1)
    preds = np.asarray(preds).reshape(-1)
    error = preds - trues
    abs_error = np.abs(error)
    sq_error = error ** 2
    mae = np.mean(abs_error)
    mse = np.mean(sq_error)
    rmse = np.sqrt(mse)
    # 防止真实值接近0导致MAPE爆炸
    mape = np.mean(abs_error / np.maximum(np.abs(trues), eps)) * 100.0

    ss_res = np.sum((trues - preds) ** 2)
    ss_tot = np.sum((trues - np.mean(trues)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + eps)

    return {'MAE': float(mae),
        'MAPE': float(mape),
        'MSE': float(mse),
        'RMSE': float(rmse),
        'R2': float(r2),}

def fit_model(name, model, train_loader, val_loader, args, v_mean, v_std, is_teacher=False, teacher_model=None):
    train_losses = []
    val_losses = []

    print(f'\n--- Training {name.capitalize()} ---')
    opt = torch.optim.Adam(model.parameters(), lr=DEFAULT_MODEL_HPARAMS[name]['lr'], weight_decay=DEFAULT_MODEL_HPARAMS[name]['weight_decay'])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=max(1, args.patience // 2), min_lr=1e-6)
    stop = EarlyStopper(patience=args.patience)
    mse_loss = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        train_mse_sum, count = 0.0, 0
        alpha_f = get_feature_kd_alpha(epoch, args.alpha, args.feature_kd_warmup_epochs, args.feature_kd_anneal_epochs) if teacher_model else 0.0

        for batch in train_loader:
            batch = [b.to(device) for b in batch]
            y = batch[-1]
            inputs = batch[4:8] if is_teacher else batch[:4]
            outputs, hidden = model(*inputs)
            loss = mse_loss(outputs, y)

            if teacher_model:
                with torch.no_grad():
                    t_out, t_hid = teacher_model(*batch[4:8])
                loss += alpha_f * mse_loss(hidden, t_hid) + mse_loss(outputs, t_out)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_mse_sum += mse_loss(outputs.detach() * v_std + v_mean, y * v_std + v_mean).sum().item()
            count += y.numel()

        val_loss, val_mse_sum, v_count = 0.0, 0.0, 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = [b.to(device) for b in batch]
                y = batch[-1]
                inputs = batch[4:8] if is_teacher else batch[:4]
                outputs, _ = model(*inputs)
                val_loss += mse_loss(outputs, y).item()
                val_mse_sum += mse_loss(outputs * v_std + v_mean, y * v_std + v_mean).sum().item()
                v_count += y.numel()

        val_loss /= len(val_loader)
        sched.step(val_loss)
        train_losses.append(train_mse_sum / count)
        val_losses.append(val_mse_sum / v_count)
        print(f'[{name.capitalize()}] Epoch {epoch+1}/{args.epochs}, Train MSE: {train_mse_sum / count:.6f}, Val MSE: {val_mse_sum / v_count:.6f}')

        if stop.step(val_loss, model):
            print(f'[{name.capitalize()}] Early stopping at epoch {epoch+1}')
            break

    stop.restore(model)
    return model, train_losses, val_losses


def save_loss_curve(train_loss, val_loss, save_path, title):
    plt.figure()
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(save_path)
    plt.close()


def save_plot(trues, preds, title, filename, color, is_error=False):
    plt.figure(figsize=(12, 6))
    if is_error:
        plt.plot(preds - trues, color=color, alpha=0.9, linewidth=1.5)
        plt.axhline(0.0, color='black', linestyle=':', linewidth=1.0)
        plt.ylabel('Prediction Error')
    else:
        plt.plot(trues, label='True Values', color='black', alpha=0.8)
        plt.plot(preds, label='Predictions', color=color, alpha=0.8, linestyle='--')
        plt.legend()
        plt.ylabel('Real Value')
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

#
# def train_model(args):
#     torch.manual_seed(42)
#     filepath = str(Path(__file__).resolve().parent / args.data_path)
#     model_tag = getattr(args, 'model_type', 'GRU')
#     plot_dir = f'plots_{model_tag}'
#     for m in ['teacher', 'student', 'baseline']:
#         os.makedirs(f'{plot_dir}/{m}/voltage', exist_ok=True)
#
#     train_ld, val_ld, test_ld, v_mean, v_std, vocab_size = load_and_split_data(filepath, args.lookback, args.horizon, args.batch_size)
#
#     def build_one_model(name, use_graph):
#         conf = DEFAULT_MODEL_HPARAMS[name]
#         return build_model(
#             model_type=args.model_type,
#             spatial_vocab_size=vocab_size,
#             conf=conf,
#             lookback=args.lookback,
#             use_graph=use_graph,
#             chunk_size=args.chunk_size,
#         ).to(device)
#
#     teacher, t_train_loss, t_val_loss = fit_model(
#         'teacher', build_one_model('teacher', args.graphemb),
#         train_ld, val_ld, args, v_mean, v_std, is_teacher=True
#     )
#
#     student, s_train_loss, s_val_loss = fit_model(
#         'student', build_one_model('student', args.graphemb),
#         train_ld, val_ld, args, v_mean, v_std, teacher_model=teacher
#     )
#
#     baseline, b_train_loss, b_val_loss = fit_model(
#         'baseline', build_one_model('baseline', 0),
#         train_ld, val_ld, args, v_mean, v_std
#     )
#
#     loss_dict = {
#         'teacher': (t_train_loss, t_val_loss),
#         'student': (s_train_loss, s_val_loss),
#         'baseline': (b_train_loss, b_val_loss),
#     }
#     for name, (tr_l, val_l) in loss_dict.items():
#         save_path = f'{plot_dir}/{name}/voltage/loss_curve.png'
#         save_loss_curve(tr_l, val_l, save_path, f'{name.capitalize()} Loss Curve')
#
#     for model, name, color, is_t in [
#         (teacher, 'teacher', 'blue', True),
#         (student, 'student', 'green', False),
#         (baseline, 'baseline', 'red', False),
#     ]:
#         print(f'\n--- {name.capitalize()} Real MSE ---')
#         base_dir = f'{plot_dir}/{name}/voltage'
#
#         for split_name, loader in [('Train', train_ld), ('Validation', val_ld), ('Test', test_ld)]:
#             trues, preds = collect_predictions(model, loader, v_mean, v_std, is_t)
#             np.save(f'{base_dir}/{split_name.lower()}_trues.npy', trues)
#             np.save(f'{base_dir}/{split_name.lower()}_preds.npy', preds)
#             print(f'{split_name} Set Voltage MSE: {((trues - preds) ** 2).mean():.4f}')
#
#             p_trues, p_preds = (trues[-200:], preds[-200:]) if split_name == 'Test' else (trues, preds)
#             base_path = f'{base_dir}/{split_name.lower()}_voltage'
#             save_plot(p_trues, p_preds, f'{name.capitalize()} {split_name}', f'{base_path}_pred_vs_true.png', color)
#             save_plot(p_trues, p_preds, f'{name.capitalize()} {split_name} Error', f'{base_path}_error.png', color, is_error=True)

def train_model(args):
    torch.manual_seed(42)
    filepath = str(Path(__file__).resolve().parent / args.data_path)
    model_tag = getattr(args, 'model_type', 'GRU')
    plot_dir = f'plots_{model_tag}'
    for m in ['teacher', 'student', 'baseline']:
        os.makedirs(f'{plot_dir}/{m}/voltage', exist_ok=True)
    train_ld, val_ld, test_ld, v_mean, v_std, vocab_size = load_and_split_data(filepath, args.lookback, args.horizon, args.batch_size)

    def build_one_model(name, use_graph):
        conf = DEFAULT_MODEL_HPARAMS[name]
        return build_model(
            model_type=args.model_type,
            spatial_vocab_size=vocab_size,
            conf=conf,
            lookback=args.lookback,
            use_graph=use_graph,
            chunk_size=args.chunk_size,
        ).to(device)

    teacher, t_train_loss, t_val_loss = fit_model(
        'teacher',
        build_one_model('teacher', args.graphemb),
        train_ld, val_ld, args, v_mean, v_std,
        is_teacher=True
    )

    student, s_train_loss, s_val_loss = fit_model(
        'student',
        build_one_model('student', args.graphemb),
        train_ld, val_ld, args, v_mean, v_std,
        teacher_model=teacher
    )

    baseline, b_train_loss, b_val_loss = fit_model(
        'baseline',
        build_one_model('baseline', 0),
        train_ld, val_ld, args, v_mean, v_std
    )

    loss_dict = {
        'teacher': (t_train_loss, t_val_loss),
        'student': (s_train_loss, s_val_loss),
        'baseline': (b_train_loss, b_val_loss),
    }

    for name, (tr_l, val_l) in loss_dict.items():
        save_path = f'{plot_dir}/{name}/voltage/loss_curve.png'
        save_loss_curve(tr_l, val_l, save_path, f'{name.capitalize()} Loss Curve')

    summary_rows = []
    for model, name, color, is_t in [
        (teacher, 'teacher', 'blue', True),
        (student, 'student', 'green', False),
        (baseline, 'baseline', 'red', False),
    ]:
        print(f'\n--- {name.capitalize()} Real Metrics ---')
        base_dir = f'{plot_dir}/{name}/voltage'

        for split_name, loader in [
            ('Train', train_ld),
            ('Validation', val_ld),
            ('Test', test_ld)
        ]:
            trues, preds = collect_predictions(model, loader, v_mean, v_std, is_t)
            np.save(f'{base_dir}/{split_name.lower()}_trues.npy', trues)
            np.save(f'{base_dir}/{split_name.lower()}_preds.npy', preds)
            metrics = compute_metrics(trues, preds)

            print(
                f"{split_name} | "
                f"MAE: {metrics['MAE']:.6f}, "
                f"MAPE: {metrics['MAPE']:.6f}, "
                f"MSE: {metrics['MSE']:.6f}, "
                f"RMSE: {metrics['RMSE']:.6f}, "
                f"R2: {metrics['R2']:.6f}"
            )

            summary_rows.append({
                'ModelFamily': args.model_type,
                'SubModel': name,              # teacher / student / baseline
                'Split': split_name,           # Train / Validation / Test
                'MAE': metrics['MAE'],
                'MAPE': metrics['MAPE'],
                'MSE': metrics['MSE'],
                'RMSE': metrics['RMSE'],
                'R2': metrics['R2'],
            })

            p_trues, p_preds = ((trues[-200:], preds[-200:]) if split_name == 'Test' else (trues, preds))
            base_path = f'{base_dir}/{split_name.lower()}_voltage'
            save_plot(p_trues,p_preds,f'{name.capitalize()} {split_name}',f'{base_path}_pred_vs_true.png',color)
            save_plot(p_trues,p_preds,f'{name.capitalize()} {split_name} Error',f'{base_path}_error.png',color,is_error=True)
    return summary_rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--horizon', type=int, default=2)
    parser.add_argument('--horizon', type=int, default=9)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lookback', type=int, default=10)  #应该相对较差
    # parser.add_argument('--lookback', type=int, default=20) #应该相对中等
    # parser.add_argument('--lookback', type=int, default=30) #应该相对最好
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_path', type=str, default='dataezpro.xlsx')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--graphemb', type=int, choices=[0, 1], default=1)
    parser.add_argument('--feature_kd_warmup_epochs', type=int, default=20)
    parser.add_argument('--feature_kd_anneal_epochs', type=int, default=30)
    parser.add_argument('--chunk_size', type=int, default=512)
    parser.add_argument('--model_type',type=str,default='GRU',
        choices=['LSTM', 'GRU', 'BiGRU', 'FreTS', 'SegRNN', 'ModernTCN', 'ConvTimeNet'],)

    args = parser.parse_args()
    all_models = ['LSTM', 'GRU', 'BiGRU', 'FreTS', 'SegRNN', 'ModernTCN', 'ConvTimeNet']
    failed_models = []
    all_summary_rows = []
    for model_name in all_models:
        args.model_type = model_name
        print(f'\n{"=" * 20} Running {model_name} {"=" * 20}\n')
        start_time = time.time()
        try:
            summary_rows = train_model(args)
            elapsed = time.time() - start_time
            print(f'[OK] {model_name} finished in {elapsed:.2f} s')
            # 给当前模型所有记录补充运行时间
            for row in summary_rows:
                row['ElapsedTimeSec'] = elapsed
            all_summary_rows.extend(summary_rows)
        except Exception as e:
            elapsed = time.time() - start_time
            print(f'[ERROR] {model_name} failed after {elapsed:.2f} s: {e}')
            failed_models.append(model_name)

    # 保存总表
    if len(all_summary_rows) > 0:
        summary_df = pd.DataFrame(all_summary_rows)

        split_order = {'Train': 0, 'Validation': 1, 'Test': 2}
        submodel_order = {'teacher': 0, 'student': 1, 'baseline': 2}
        summary_df['SplitOrder'] = summary_df['Split'].map(split_order)
        summary_df['SubModelOrder'] = summary_df['SubModel'].map(submodel_order)
        summary_df = summary_df.sort_values(
            by=['ModelFamily', 'SubModelOrder', 'SplitOrder']
        ).drop(columns=['SplitOrder', 'SubModelOrder'])
        summary_csv_path = 'results_summary.csv'
        summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
        print(f'\n[Saved] Summary CSV -> {summary_csv_path}')

    print('\n================ All models finished ================\n')
    if failed_models:
        print('Failed models:', failed_models)
    else:
        print('All models completed successfully.')