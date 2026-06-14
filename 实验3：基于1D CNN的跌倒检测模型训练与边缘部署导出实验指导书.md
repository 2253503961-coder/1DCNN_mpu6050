# 基于1D CNN的跌倒检测模型训练与边缘部署导出实验指导书

# 一、实验目的

1. **掌握时间序列数据处理**：学习如何使用滑动窗口（Sliding Window）技术将连续的 IMU 传感器流数据切分为可供神经网络训练的离散样本。

2. **理解 1D 卷积神经网络（1D CNN）**：掌握 1D 卷积在处理多通道传感器时间序列（六轴数据）中的优势，并能手算特征图的维度变化。

3. **掌握 PyTorch 训练流水线**：熟悉自定义 `Dataset`、模型构建、损失函数、优化器及完整的前向与反向传播训练流程。

4. **掌握边缘计算部署前置技能**：学习如何将 PyTorch 模型导出为 ONNX 标准格式，为后续在 STM32（通过 X-CUBE-AI）部署打下基础。

# 二、实验环境准备

- **编程语言**：Python 3.8+

- **核心依赖库**：`torch`, `pandas`, `numpy`

- **数据集**：需要准备上一阶段采集的 `dataset.csv` 文件（包含 ax, ay, az, gx, gy, gz 和 label 列）。

# 三、实验步骤与代码原理解析

## 步骤 1：数据读取与标准化 (Z-score Normalization)

### 【原理解析】

六轴传感器中，加速度计和陀螺仪的量纲和数值范围差异巨大（如加速度在 $\pm 2000$ 左右，角速度可能上万）。神经网络对输入特征的绝对大小非常敏感，如果不统一量纲，模型极难收敛。在此采用 Z-score 归一化公式：

 $X_{norm} = \frac{X - \mu}{\sigma}$ 

### 【代码映射】

在 `IMUDatasetCNN` 类的初始化中，我们通过 Pandas 读取 CSV，并利用 NumPy 计算均值和标准差，对数据进行归一化。

**⚠️ 部署关键点**：代码中打印出的 `mean`（均值）和 `std`（标准差）必须记录下来！后续在 STM32 上运行 C 代码推理前，必须对实时读取到的传感器数据进行一模一样的运算。

```python
# 读取数据
        data = pd.read_csv(csv_file)
        features = data[['ax', 'ay', 'az', 'gx', 'gy', 'gz']].values
        
        # [非常重要] Z-score 归一化
        self.mean = np.mean(features, axis=0)
        self.std = np.std(features, axis=0)
        
        # 加上 1e-8 防止除以 0 的情况发生
        features = (features - self.mean) / (self.std + 1e-8)
```

## 步骤 2：滑动窗口切片与张量转换

### 【原理解析】

跌倒是一个“过程”而非单一“瞬间”。我们使用滑动窗口截取长度为 50 帧（假设采样率为 50Hz，即 1 秒的数据）作为一个样本片段。此外，PyTorch 的 Conv1d 层对输入数据的形状有严格要求：(Batch_Size, Channels, Sequence_Length)。我们提取出的原始形状是 (N, 50, 6)，必须进行维度互换。

### 【代码映射】

使用 permute(0, 2, 1) 将序列长度（50）和通道数（6）互换，变为 (N, 6, 50)。

```python
# 滑动窗口切片 (窗口大小50，步长10)
        for i in range(0, len(features) - window_size + 1, step_size):
            window = features[i:i+window_size]
            window_label = np.max(labels[i:i+window_size]) # 窗口内只要包含跌倒帧，即视为跌倒样本
            x_data.append(window)
            
        # 转换为 PyTorch 张量并翻转维度
        self.x_data = torch.tensor(np.array(x_data), dtype=torch.float32).permute(0, 2, 1)
```

## 步骤 3：构建专供微控制器的轻量级 1D CNN

### 【原理解析】

由于 STM32F103 的 RAM 和 Flash 资源极其有限，模型必须足够轻量。我们使用一层 1D 卷积提取时间序列特征，一层最大池化降维，最后连接全连接层进行二分类。

维度计算公式： $L_{out} = L_{in} - kernel\_size + 1$  (在 stride=1 且 padding=0 的情况下)。

- Conv1d：输入长度 50，卷积核 3。输出长度 =  $50 - 3 + 1 = 48$ 。

- MaxPool1d：核为 2，长度减半。输出长度 =  $48 / 2 = 24$ 。

- 展平 (Flatten)：输出通道数 8 × 长度 24 = 192。

### 【代码映射】

```python
class Lightweight1DCNN(nn.Module):
    def __init__(self, window_size=50, num_classes=2):
        # 卷积层: 序列长度由 50 变为 48
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=8, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2) # 池化后长度变为 24
        
        # 计算展平后的节点数: 8 * 24 = 192
        flattened_size = 8 * 24 
        
        self.fc1 = nn.Linear(flattened_size, 16)
        self.fc2 = nn.Linear(16, num_classes) # 输出 2 个神经元 (日常 vs 跌倒)
```

## 步骤 4：定义训练流水线

### 【原理解析】

训练过程是一个不断试错并调整内部权重的过程。通过前向传播得到预测结果，通过 CrossEntropyLoss 计算预测结果与真实标签的误差（Loss），然后通过 Adam 优化器进行反向传播，更新模型权重。

### 【代码映射】

我们在 train_and_export 函数中定义了上述过程，并通过循环遍历 30 个 Epoch：

```python
# 实例化模型、损失函数和优化器 (学习率 0.005)
    model = Lightweight1DCNN(window_size=window_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # 核心训练循环
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()            # 1. 清空梯度缓存
            outputs = model(inputs)          # 2. 前向传播预测
            loss = criterion(outputs, labels)# 3. 计算误差
            loss.backward()                  # 4. 反向传播求导
            optimizer.step()                 # 5. 更新权重
```

## 步骤 5：模型导出为 ONNX 格式

### 【原理解析】

PyTorch 训练出来的模型（.pth 或 .pt）无法直接放到 STM32 上运行。ONNX (Open Neural Network Exchange) 是一种通用的深度学习模型格式。我们需要先将模型导出为 ONNX 格式，随后 STMicroelectronics 官方的 X-CUBE-AI 工具就能读取该格式，并自动将其翻译为 STM32 能懂的 C 语言数组和库调用。

### 【代码映射】

要导出 ONNX，必须提供一个与实际输入形状完全一致的“假张量（Dummy Input）”，以便工具追踪数据流转图。

```python
model.eval() # 切换为评估模式 (关闭 Dropout 等)
    
    # 构造 Dummy 张量: (Batch=1, Channels=6, Seq_Len=50)
    dummy_input = torch.randn(1, 6, window_size)
    onnx_filename = "fall_detection_f103.onnx"
    
    # 导出模型
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_filename, 
        input_names=['imu_input'], 
        output_names=['fall_prob'],
        opset_version=11 # 指定算子集版本，保证兼容性
    )
```

# 四、课后操作任务

在运行此代码前，请确保上一节课中通过 --mode collect 收集的数据集已被命名为 dataset.csv 并放置在同级目录下。

运行脚本：`python train.py`。观察终端中 Loss 的下降趋势和 Accuracy（准确率）的上升趋势。

在目录中找到生成的 fall_detection_f103.onnx 文件，将其备份，这将是下一节单片机部署课的核心文件。