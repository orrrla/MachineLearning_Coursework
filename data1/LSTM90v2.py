import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader

# ========= 配置参数 ========= #
input_hours = 90 * 24       # 输入序列长度（90天）
output_hours = 90 * 24      # 预测序列长度（90天）
use_cols = [
    'Global_active_power','Global_reactive_power',
    'Global_intensity','Sub_metering_1',
    'Sub_metering_2','Sub_metering_3',
    'sub_metering_remainder'
]

# ========= 读取 & 补充字段 ========= #
def load_hourly(path):
    df = pd.read_csv(path, parse_dates=['DateTime'])
    df.sort_values('DateTime', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 计算 sub_metering_remainder
    df["sub_metering_remainder"] = (
        df["Global_active_power"] * 1000 / 60
        - df["Sub_metering_1"]
        - df["Sub_metering_2"]
        - df["Sub_metering_3"]
    ).clip(lower=0)

    df = df[['DateTime'] + use_cols].copy()
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

train_df = load_hourly("train_hourly.csv")
test_df  = load_hourly("test_hourly.csv")  # ✅ 直接加载，无需 repeat

# 合并数据（确保按时间拼接）
full_df = pd.concat([train_df, test_df], ignore_index=True)

# ========= 标准化 ========= #
scaler = MinMaxScaler()
scaled = scaler.fit_transform(full_df[use_cols])
target = scaled[:, 0]  # 目标值：Global_active_power

# ========= 滑动窗口构造函数 ========= #
def build_samples(data, target, input_len, output_len):
    X, y = [], []
    for i in range(input_len, len(data) - output_len):
        X.append(data[i - input_len:i])
        y.append(target[i:i + output_len])
    return np.array(X), np.array(y)

# ========= 构建样本 ========= #
X_all, y_all = build_samples(scaled, target, input_hours, output_hours)

# ========= 拆分 train / test ========= #
# 通过时间戳判断 test 起点，也可根据 test_df 的开始索引来切分
split_index = len(train_df) - input_hours  # 保证 test 的输入从 train 末尾接上
X_train = X_all[:split_index]
y_train = y_all[:split_index]
X_test  = X_all[split_index:]
y_test  = y_all[split_index:]

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test  shape: {X_test.shape}")
print(f"y_test  shape: {y_test.shape}")

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_dim=7, hidden_size=128, num_layers=2, dropout=0.3, output_len=2160):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_len)  # 双向 => 2 × hidden_size

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        out, _ = self.bilstm(x)  # out: (batch_size, seq_len, hidden_size*2)
        out = self.dropout(out)
        out = out[:, -1, :]  # 取最后时间步
        out = self.fc(out)  # 输出 (batch_size, 2160)
        return out


import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from torch.utils.data import TensorDataset

train_ds = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32)
)

n_features = 7
patience = 10
no_improve_count = 0

model = LSTMModel(input_dim=n_features, output_len=2160)
# GPU 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 数据集切割：80% 训练, 20% 验证
total = len(train_ds)
train_size = int(total * 0.8)
val_size = total - train_size
train_subset, val_subset = random_split(train_ds, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

# 损失和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train_losses, val_losses = [], []
best_val = float('inf')
best_model = None
for epoch in range(1, 101):
    # —— 训练阶段 —— #
    model.train()
    total_train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * xb.size(0)
    avg_train = total_train_loss / len(train_subset)
    train_losses.append(avg_train)

    # —— 验证阶段 —— #
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            total_val_loss += loss.item() * xb.size(0)
    avg_val = total_val_loss / len(val_subset)
    val_losses.append(avg_val)

    print(f"Epoch {epoch:02d}: Train Loss = {avg_train:.4f}, Val Loss = {avg_val:.4f}")
    if avg_val < best_val:
        best_val = avg_val
        best_model = model.state_dict()
        no_improve_count = 0
    else:
        no_improve_count += 1

    if no_improve_count >= patience:
        print(f"Early stopped at epoch {epoch}")
        break
# 🔍 训练 & 验证 Loss 曲线
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label='Train MSE')
plt.plot(val_losses, label='Val   MSE')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig("Loss.png", dpi=300)
plt.show()


import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载模型
model.load_state_dict(best_model)
model.eval()

# 准备测试集
test_ds = torch.utils.data.TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.float32)
)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

# 预测
preds, gts = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        out = model(xb).cpu().numpy()  # shape: (batch_size, 2160)
        preds.append(out)
        gts.append(yb.numpy())

preds = np.concatenate(preds, axis=0)  # (N, 2160)
gts   = np.concatenate(gts,   axis=0)

# ========== 反归一化 ==========
zeros = np.zeros((preds.shape[0], preds.shape[1], 6))  # (N, 2160, 6)
preds_padded = np.concatenate([preds[..., np.newaxis], zeros], axis=-1)
gts_padded   = np.concatenate([gts[..., np.newaxis], zeros], axis=-1)

preds_real, gts_real = [], []
for i in range(preds.shape[0]):
    pred_inv = scaler.inverse_transform(preds_padded[i])[:, 0]
    gt_inv   = scaler.inverse_transform(gts_padded[i])[:, 0]
    preds_real.append(pred_inv)
    gts_real.append(gt_inv)

preds_real = np.array(preds_real)  # shape: (N, 2160)
gts_real   = np.array(gts_real)

# ========== 聚合为每天 ==========
# 每24小时为一组 -> reshape (N, 90, 24) -> axis=2 mean -> (N, 90)
preds_day = preds_real.reshape(preds_real.shape[0], 90, 24).mean(axis=2)
gts_day   = gts_real.reshape(gts_real.shape[0], 90, 24).mean(axis=2)

# ========== 评估指标（小时级）==========
mae_hour  = mean_absolute_error(gts_real, preds_real)
rmse_hour = np.sqrt(mean_squared_error(gts_real, preds_real))
print(f"反归一化后（小时级） MAE: {mae_hour:.4f} kW")
print(f"反归一化后（小时级） RMSE: {rmse_hour:.4f} kW")

# ========== 可视化一个样本（按天）==========
plt.figure(figsize=(10, 5))
plt.plot(gts_day[0], label="真实值", linewidth=1.2)
plt.plot(preds_day[0], label="预测值", linewidth=1.2)
plt.xlabel("未来天数")
plt.ylabel("每日平均总有功功率（kW）")
plt.title("未来90天预测 vs 真实值（每日聚合）")
plt.xticks(ticks=np.arange(0, 91, 5))  # 每5天一个刻度
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("prediction_vs_truth.png", dpi=300)
plt.show()
