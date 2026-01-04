from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
import cv2
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========全局设置==========
# 配置阿里云MNIST镜像源，优先级最高，彻底替换默认国外源
os.environ['TORCHVISION_DATASETS_MNIST_URL'] = 'https://mirrors.aliyun.com/pytorch-vision/mnist/'
np.random.seed(42)
sns.set_style(style="white")
sns.set_palette(palette='bright')
ChinaFonts = {"黑体": "simhei", "宋体": "simsun", "华文楷体": "STKAITI"}
plt.rcParams["font.sans-serif"] = ChinaFonts["华文楷体"]  # 解决中文乱码问题
plt.rcParams["axes.unicode_minus"] = False # 解决负号无法正常显示的问题
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

# ==========搭建LeNet-5网络==========
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ) # 卷积层
        self.flat = nn.Flatten() # 展平层
        self.fc = nn.Sequential(
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        ) # 全连接层

    def forward(self, x):
        out = self.conv(x) # 卷积
        out = self.flat(out) # 展平
        out = self.fc(out) # 全连接

        return out

# ==========模型初始化==========
model = LeNet5().to(device) # GPU加速
criterion = nn.CrossEntropyLoss() # 交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) # Adam优化器

# ==========模型训练==========
def train(train_loader):
    model.train() # 训练模式
    total_loss = 0.0 # 累计损失
    correct = 0 # 正确预测数
    total = 0 # 总样本数
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device) # GPU加速
        outputs = model(images) # 前向传播
        loss = criterion(outputs, labels) # 计算损失
        
        optimizer.zero_grad() # 清除梯度
        loss.backward() # 反向传播
        optimizer.step() # 更新参数

        total_loss += loss.item() * images.size(0) # 累计损失
        _, predicted = torch.max(outputs, 1) # 获取预测类别
        total += labels.size(0)
        correct += (predicted == labels).sum().item() # 累计正确预测数

        # 打印训练信息
        if (batch_idx + 1) % 100 == 0:
            print(f"Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%")

    # 计算本轮训练的平均损失和准确率
    avg_loss = total_loss / len(train_loader.dataset)
    avg_acc = correct / total

    return avg_loss, avg_acc

# ==========模型评估==========
def evaluate(test_loader):
    model.eval() # 评估模式
    total_loss = 0.0 
    correct = 0 
    total = 0 
    with torch.no_grad(): 
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device) 
            outputs = model(images) 
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0) 
            _, predicted = torch.max(outputs, 1) 
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # 计算测试集的平均损失和准确率
    avg_loss = total_loss / len(test_loader.dataset)
    avg_acc = correct / total

    return avg_loss, avg_acc

# ==========可视化==========
def visualize(train_loss, test_loss, train_acc, test_acc):
    plt.figure(figsize=(12, 6))
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="Train Loss", linewidth=2, alpha=0.8)
    plt.plot(test_loss, label="Test Loss", linewidth=2, alpha=0.8)
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper left")
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label="Train Accuracy", linewidth=2, alpha=0.8)
    plt.plot(test_acc, label="Test Accuracy", linewidth=2, alpha=0.8)
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper left")

    plt.tight_layout()
    plt.show()

# ==========主函数==========
def main():
    # 数据预处理,将图像转换为张量
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # 数据标准化
    ])
    # 加载数据集
    train_dataset = datasets.MNIST(
        root="D:\medical_image_process\data", train=True, transform=transform, download=False
    )
    test_dataset = datasets.MNIST(
        root="D:\medical_image_process\data", train=False, transform=transform, download=False
    )
    # 数据加载器
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True # 训练集随机打乱
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False # 测试集不随机打乱
    )

    # 初始化损失、准确率列表
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(EPOCHS):
        # 训练模型
        train_loss, train_acc = train(train_loader)
        # 评估模型
        test_loss, test_acc = evaluate(test_loader)
        # 记录损失和准确率  
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        # 每10轮打印一次训练信息
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}]")
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {100 * train_acc:.2f}%")
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {100 * test_acc:.2f}%")
    
    # 可视化损失和准确率
    visualize(train_losses, test_losses, train_accs, test_accs)

# ==========主程序==========
if __name__ == "__main__":
    main()