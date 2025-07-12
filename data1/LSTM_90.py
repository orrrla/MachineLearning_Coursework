import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -- 1. 加载数据 --
use_cols = [
    'Global_active_power', 'Global_reactive_power',
    'Global_intensity', 'Sub_metering_1',
    'Sub_metering_2', 'Sub_metering_3'
]
train = pd.read_csv("train_hourly.csv", parse_dates=["DateTime"], index_col="DateTime")[use_cols]
test = pd.read_csv("test_hourly.csv", parse_dates=["DateTime"], index_col="DateTime")[use_cols]

# -- 2. 归一化，仅用 train 拟合 scaler --
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train.values)
test_scaled = scaler.transform(test.values)


# -- 3. 构造滑窗样本函数（预测未来24小时 Global_active_power） --
def create_sliding_dataset(data_scaled, n_in=2160, n_out=24, target_col=0):
    """
    data_scaled: shape [T, n_features]
    return: X [N, n_in, n_feat], y [N, n_out]
    """
    X, y = [], []
    for i in range(len(data_scaled) - n_in - n_out + 1):
        x_win = data_scaled[i: i + n_in]
        y_win = data_scaled[i + n_in: i + n_in + n_out, target_col]  # 只取 var1 的24步
        X.append(x_win)
        y.append(y_win)
    return np.array(X), np.array(y)


# -- 4. 构造滑窗训练集和测试集 --
n_in, n_out = 2160, 24  # 输入90天 → 预测1天
X_train, y_train = create_sliding_dataset(train_scaled, n_in, n_out)
X_test, y_test = create_sliding_dataset(test_scaled, n_in, n_out)

print("Shapes:")
print("X_train:", X_train.shape)  # [N, 2160, 6]
print("y_train:", y_train.shape)  # [N, 24]
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


# -- 5. PyTorch Dataset 封装 --
class SlidingWindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_ds = SlidingWindowDataset(X_train, y_train)
test_ds = SlidingWindowDataset(X_test, y_test)

# -- 6. DataLoader --
batch_size = 64
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# -- 7. 验证形状 --
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
xb, yb = next(iter(train_loader))
print("Batch shapes:", xb.shape, yb.shape, "; device =", device)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=200, num_layers=2, dropout1=0.2, dropout2=0.3, output_len=24):
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False
        )
        self.drop1 = nn.Dropout(dropout1)

        self.lstm2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False
        )
        self.drop2 = nn.Dropout(dropout2)

        # 输出为 [batch, output_len]
        self.fc = nn.Linear(hidden_size, output_len)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(out)

        out, _ = self.lstm2(out)
        out = out[:, -1, :]  # 只取最后时间步
        out = self.drop2(out)

        out = self.fc(out)  # shape: [batch, output_len]
        return out


n_vars = X_train.shape[2]  # 等价于 6

model = LSTMModel(input_size=n_vars)
# GPU 设备
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 数据集切割：80% 训练, 20% 验证
total = len(train_ds)
train_size = int(total * 0.8)
val_size = total - train_size
train_subset, val_subset = random_split(train_ds, [train_size, val_size])  # :contentReference[oaicite:6]{index=6}

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

# 损失和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_losses, val_losses = [], []

best_val_loss = float('inf')
for epoch in range(1, 101):  # 训练 100 个 epoch
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
    avg_train = total_train_loss / len(train_loader)
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
    avg_val = total_val_loss / len(val_loader)
    val_losses.append(avg_val)

    print(f"Epoch {epoch:02d}: Train Loss = {avg_train:.4f}, Val Loss = {avg_val:.4f}")
    # 保存最佳模型
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), 'best_model.pth')

    # 每10个epoch保存检查点
    if epoch % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, f'checkpoint_epoch_{epoch}.pth')



plt.figure(figsize=(8, 4))
plt.plot(train_losses, label='Train MSE')
plt.plot(val_losses, label='Val   MSE')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_validation_loss.png', dpi=300, bbox_inches='tight')
plt.show()

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
model.eval()

# 设定滑动窗口参数
window_hours = 2160  # 90 天输入
predict_hours = 24  # 每次预测 1 天
predict_days = 90  # 共预测 90 天
n_steps = predict_days

# 初始化滑动窗口输入
history = test_vals[:window_hours]  # shape: [2160, 6]
predictions = []

with torch.no_grad():
    for step in range(n_steps):
        input_window = history[-window_hours:]  # shape: [2160, 6]
        x_input = torch.tensor(input_window, dtype=torch.float32).unsqueeze(0).to(device)  # [1, 2160, 6]
        y_pred = model(x_input).cpu().numpy().flatten()  # shape: [24]

        predictions.append(y_pred)

        # 将预测结果拼到历史（只填 var1，其他为0）
        pad = np.zeros((predict_hours, 6))
        pad[:, 0] = y_pred
        history = np.concatenate([history, pad], axis=0)

# 拼接所有预测值（缩放前）
preds_scaled = np.concatenate(predictions, axis=0).reshape(-1, 1)  # [2160, 1]

# 构建和预测值形状一致的 Ground Truth（也是 2160 条）
gt_scaled = test_vals[window_hours: window_hours + predict_days * 24, 0].reshape(-1, 1)  # 取 var1


# 拼出完整6维进行反归一化
def inverse_var1(scaled_arr):
    full = np.zeros((len(scaled_arr), 6))
    full[:, 0] = scaled_arr[:, 0]
    return scaler.inverse_transform(full)[:, 0]


preds_inv = inverse_var1(preds_scaled)
gt_inv = inverse_var1(gt_scaled)

# —— 日均值 —— #
pred_daily = preds_inv.reshape(predict_days, 24).mean(axis=1)
gt_daily = gt_inv.reshape(predict_days, 24).mean(axis=1)

# —— 评价指标 —— #
mae = mean_absolute_error(gt_inv, preds_inv)
mse = mean_squared_error(gt_inv, preds_inv)
print(f"[反归一化] 滑动预测 MAE = {mae:.4f}, MSE = {mse:.4f}")

# —— 可视化 —— #
plt.figure(figsize=(12, 5))
plt.plot(gt_daily, label='Ground Truth (每日均值)', color='orange')
plt.plot(pred_daily, label='Predicted (每日均值)', color='blue')
plt.xlabel('未来天数')
plt.ylabel('Global_active_power (kW)')
plt.title('未来90天每日总有功功率预测 VS 真实值')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('daily_power_prediction_vs_ground_truth.png',
            dpi=300,
            bbox_inches='tight',
            facecolor='white')  # 可选：设置白色背景
plt.show()
