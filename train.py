import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# 1. 数据集加载与归一化
class IMUDatasetCNN(Dataset):
    def __init__(self, csv_file, window_size=50):
        data = pd.read_csv(csv_file)
        features = data[['ax', 'ay', 'az', 'gx', 'gy', 'gz']].values
        labels = data['label'].values
        
        self.mean = np.mean(features, axis=0)
        self.std = np.std(features, axis=0)
        features = (features - self.mean) / (self.std + 1e-8)
        
        x_data, y_data = [], []
        for i in range(0, len(features) - window_size + 1, 10):
            x_data.append(features[i:i+window_size])
            y_data.append(np.max(labels[i:i+window_size]))
            
        self.x_data = torch.tensor(np.array(x_data), dtype=torch.float32).permute(0, 2, 1)
        self.y_data = torch.tensor(np.array(y_data), dtype=torch.long)

    def __len__(self): return len(self.x_data)
    def __getitem__(self, idx): return self.x_data[idx], self.y_data[idx]

# 2. 定义极简 1D CNN (专为纯 C 手写优化)
class TinyCNN(nn.Module):
    def __init__(self):
        super(TinyCNN, self).__init__()
        # 输入: [Batch, 6, 50] -> 输出: [Batch, 4, 48]
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=4, kernel_size=3)
        self.relu = nn.ReLU()
        # 输出: [Batch, 4, 24]
        self.pool = nn.MaxPool1d(kernel_size=2)
        # 展平后: 4 * 24 = 96
        self.fc = nn.Linear(96, 2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def main():
    # 训练模型
    dataset = IMUDatasetCNN('my_dataset.csv')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = TinyCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    print("开始训练极简 CNN...")
    for epoch in range(30):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
    print("训练完成！")

    # ==========================================
    # 3. 导出 ONNX (用于可视化和结构确认)
    # ==========================================
    dummy_input = torch.randn(1, 6, 50)
    torch.onnx.export(model, dummy_input, "tiny_cnn.onnx", input_names=['input'], output_names=['output'])
    print("已生成 tiny_cnn.onnx")

    # ==========================================
    # 4. 生成纯 C 语言头文件 (权重矩阵与归一化参数)
    # ==========================================
    print("正在提取权重生成纯 C 代码...")
    with open("cnn_weights.h", "w") as f:
        f.write("/* 自动生成的 1D CNN 纯 C 权重与参数 */\n")
        f.write("#ifndef CNN_WEIGHTS_H\n#define CNN_WEIGHTS_H\n\n")
        
        # 写入归一化参数
        f.write(f"const float FEATURE_MEAN[6] = {{{', '.join([f'{x:.6f}f' for x in dataset.mean])}}};\n")
        f.write(f"const float FEATURE_STD[6]  = {{{', '.join([f'{x:.6f}f' for x in dataset.std])}}};\n\n")

        # 遍历模型参数提取权重
        for name, param in model.named_parameters():
            flat_data = param.data.numpy().flatten()
            array_str = ", ".join([f"{x:.6f}f" for x in flat_data])
            var_name = name.replace(".", "_")
            f.write(f"const float {var_name}[{len(flat_data)}] = {{{array_str}}};\n\n")
            
        f.write("#endif // CNN_WEIGHTS_H\n")
    print("✅ 纯 C 头文件 cnn_weights.h 生成成功！")

if __name__ == "__main__":
    main()