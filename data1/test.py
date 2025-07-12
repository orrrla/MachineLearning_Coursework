import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- 1. 加载数据 ---
use_cols = [
    'Global_active_power', 'Global_reactive_power',
    'Global_intensity', 'Sub_metering_1',
    'Sub_metering_2', 'Sub_metering_3'
]
test = pd.read_csv("test_hourly.csv", parse_dates=["DateTime"], index_col="DateTime")[use_cols]

# --- 2. 归一化（需与训练时相同的scaler）---
scaler = MinMaxScaler()
scaler.fit(test.values)  # 假设scaler已在训练时拟合，此处仅用于演示
test_scaled = scaler.transform(test.values)


# --- 3. 定义模型 ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=200, num_layers=2, dropout1=0.2, dropout2=0.3, output_len=24):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.drop1 = nn.Dropout(dropout1)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.drop2 = nn.Dropout(dropout2)
        self.fc = nn.Linear(hidden_size, output_len)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]
        out = self.drop2(out)
        out = self.fc(out)
        return out


# --- 4. 加载模型 ---
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model = LSTMModel(input_size=6).to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# --- 5. 预测逻辑 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

window_hours = 2160  # 输入窗口
predict_hours = 24  # 每次预测24小时
predict_days = 90  # 预测90天
n_steps = predict_days

history = test_scaled[:window_hours]  # 初始化历史数据
predictions = []

with torch.no_grad():
    for step in range(n_steps):
        input_window = history[-window_hours:]
        x_input = torch.tensor(input_window, dtype=torch.float32).unsqueeze(0).to(device)
        y_pred = model(x_input).cpu().numpy().flatten()
        predictions.append(y_pred)

        # 更新历史数据（仅填充目标列）
        new_steps = np.zeros((predict_hours, 6))
        new_steps[:, 0] = y_pred
        history = np.concatenate([history, new_steps], axis=0)

# --- 6. 后处理与评估 ---
preds_scaled = np.concatenate(predictions, axis=0).reshape(-1, 1)
gt_scaled = test_scaled[window_hours: window_hours + predict_days * 24, 0].reshape(-1, 1)


def inverse_var1(scaled_arr):
    full = np.zeros((len(scaled_arr), 6))
    full[:, 0] = scaled_arr[:, 0]
    return scaler.inverse_transform(full)[:, 0]


preds_inv = inverse_var1(preds_scaled)
gt_inv = inverse_var1(gt_scaled)

# 计算日均值
pred_daily = preds_inv.reshape(predict_days, 24).mean(axis=1)
gt_daily = gt_inv.reshape(predict_days, 24).mean(axis=1)

# 评估指标
mae = mean_absolute_error(gt_inv, preds_inv)
mse = mean_squared_error(gt_inv, preds_inv)
print(f"MAE: {mae:.4f}, MSE: {mse:.4f}")

# --- 7. 可视化 ---
plt.figure(figsize=(12, 5))
plt.plot(gt_daily, label='真实值', color='orange')
plt.plot(pred_daily, label='预测值', color='blue')
plt.xlabel('未来天数')
plt.ylabel('总有功功率 (kW)')
plt.title('90天每日功率预测 vs 真实值')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('prediction_result.png', dpi=300, bbox_inches='tight')
plt.show()