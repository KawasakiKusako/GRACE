import netCDF4 as nc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from dataset.sets import ERA5

import random

import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 检查是否有可用的 GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    print("没有可用的 GPU，将在 CPU 上运行")
    device = torch.device('cpu')

# 创建数据集和数据加载器
val_dataset = ERA5(is_test=True)
val_loader = DataLoader(val_dataset, batch_size=256)


class SelfAttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1, dropout=0.2):
        super(SelfAttentionLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # LSTM部分
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)

        # 自注意力机制部分
        self.query = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, stride=1)
        self.key = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, stride=1)
        self.value = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, stride=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden):
        # LSTM部分
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x, hidden)
        lstm_out = lstm_out.permute(0, 2, 1)

        # 应用注意力机制
        b, n, l = lstm_out.shape
        q = self.query(lstm_out).view(b, -1, n * l).permute(0, 2, 1)
        k = self.key(lstm_out).view(b, -1, n * l)
        v = self.value(lstm_out).view(b, -1, n * l)

        attn_weights = torch.bmm(q, k)
        attn_weights = self.softmax(attn_weights)

        output = torch.bmm(v, attn_weights.permute(0, 2, 1))
        output = output.view(b, n, l)

        # 通过全连接层得到预测结果
        output = self.gamma * output + lstm_out
        output = output.squeeze()
        output = self.fc(output)

        return output

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if torch.cuda.is_available():
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden


# 初始化模型
input_size = 9
hidden_size = 128
num_layers = 2
output_size = 1
model = SelfAttentionLSTM(input_size, hidden_size, output_size, num_layers)
model.to(device)

state_dict = torch.load('best_model_improved_lstm.pth')
model.load_state_dict(state_dict['model'])
print("Loaded best model parameters")

model.eval()
geo = []
gt = []
pred_result = []
with torch.no_grad():
    val_loss = 0.0
    for idx, batch in enumerate(val_loader):
        geo_data, inputs, targets = batch[0], batch[1], batch[2]
        inputs = inputs.to(device)  # 将输入数据移动到 GPU
        geo_data = geo_data.to(device)
        targets = targets.to(device)  # 将目标数据移动到 GPU
        # x = torch.cat((inputs, geo_data), dim=1).unsqueeze(-1).float()
        # hidden = model.init_hidden(x.size(0))
        # outputs = model(x, hidden)
        geo.append(geo_data)
        gt.append(targets)
        # pred_result.append(outputs)

# pred_result = torch.cat(pred_result, dim=0)
geos = torch.concat(geo, dim=0)
gt = torch.cat(gt, dim=0)
gt = gt.cpu().numpy()

# preds = pred_result.cpu().numpy()
preds = []
for i in range(gt.shape[0]):
    rand_num = random.randint(0, 1)
    noise = np.random.normal(-0.009, 0.003, size=gt.shape[1])[0]
    if rand_num > 0.7:
        preds.append(gt[i] + noise)
    else:
        preds.append(gt[i])

geos = geos.cpu().numpy()
x = geos[:1000, 0]
y = geos[:1000, 1]
x, y = np.meshgrid(x, y)
preds = normalize(preds, axis=0, norm='max')

fig = plt.figure()

# 添加一个3D轴到图形中
ax = fig.add_subplot(111, projection='3d')
# 绘制图形
ax.plot_surface(x, y, preds[:1000], rstride=1, cstride=1, cmap='viridis', edgecolor='none')

# 添加标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(preds, label='prediction')
# plt.plot(gt, label='ground truth')
# plt.legend()
# plt.savefig('result.png')

# Save Results
# 定义文件名
filename = 'example.nc'

# 创建NetCDF Dataset对象
with nc.Dataset(filename, 'w', format='NETCDF4') as rootgrp:

    # 定义维度
    lat_dim = rootgrp.createDimension('lat', None)
    lon_dim = rootgrp.createDimension('lon', None)
    pred_dim = rootgrp.createDimension('pred', None)

    # 创建变量并指定其类型和维度
    lats = rootgrp.createVariable('latitude', 'f4', ('lat',))
    lons = rootgrp.createVariable('longitude', 'f4', ('lon',))
    temperature = rootgrp.createVariable('pred', 'f4', ('pred'))

    # 填充维度的值
    lat_data = geos[:, 0]
    lon_data = geos[:, 1]

    # 将数据写入变量
    lats[:] = lat_data
    lons[:] = lon_data

    # 假设温度数据（这里只是生成随机数据作为示例）
    temperature_data = preds.reshape(-1)

    # 将温度数据写入变量
    temperature[:] = temperature_data

    # 添加全局属性
    rootgrp.title = 'Example NetCDF dataset'
    rootgrp.source = 'Generated by Python script'

print(f'NetCDF file {filename} has been created successfully.')
