import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from dataset.sets import ERA5
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

# 检查是否有可用的 GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    print("没有可用的 GPU，将在 CPU 上运行")
    device = torch.device('cpu')

# 创建数据集和数据加载器
train_dataset = ERA5(is_test=False)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

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
        q = self.query(lstm_out).view(b, -1, n*l).permute(0, 2, 1)
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

# state_dict = torch.load('best_model_improved_lstm_0320.pth')
# model.load_state_dict(state_dict['model'])
# print("Loaded best model parameters")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.0001)  # 调整学习率和正则化项

# 学习率调度器
T_max = 200
scheduler = CosineAnnealingLR(optimizer, T_max, eta_min=0.000001, last_epoch=-1)

# 训练模型
num_epochs = 500
best_val_loss = float('inf')
best_model = None

# 存储训练和验证损失
train_losses = []
val_losses = []
epochs = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for idx, batch in enumerate(train_loader):
        geo_data, inputs, targets = batch[0], batch[1], batch[2]
        optimizer.zero_grad()
        inputs = inputs.to(device)  # 将输入数据移动到 GPU
        geo_data = geo_data.to(device)
        targets = targets.to(device)  # 将目标数据移动到 GPU
        x = torch.cat((inputs, geo_data), dim=1).unsqueeze(-1).float()
        hidden = model.init_hidden(x.size(0))
        outputs = model(x, hidden)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()  # 累加每个batch的损失

        if (idx + 1) % 1000 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Batch {idx + 1}/{len(train_loader)}, Loss: {train_loss}')

    # 在验证集上计算损失
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for idx, batch in enumerate(val_loader):
            geo_data, inputs, targets = batch[0], batch[1], batch[2]
            inputs = inputs.to(device)  # 将输入数据移动到 GPU
            geo_data = geo_data.to(device)
            targets = targets.to(device)  # 将目标数据移动到 GPU
            x = torch.cat((inputs, geo_data), dim=1).unsqueeze(-1).float()
            hidden = model.init_hidden(x.size(0))
            outputs = model(x, hidden)
            val_loss += criterion(outputs, targets).item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    print(f'Validation: Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss}, Validation MSE: {avg_val_loss}')
    epochs.append(epoch + 1)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    # 调整学习率
    scheduler.step(avg_val_loss)

    # 保存在验证集上表现最好的模型
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model = model.state_dict()
        # 保存最佳模型参数
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
                   'best_model_improved_lstm.pth')

df = pd.DataFrame({'epoch': epochs, 'train_loss': train_losses, 'val_loss': val_losses})
df.to_csv('loss_improved_lstm.csv', index=False)
