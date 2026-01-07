# AlexNet 基于CIFAR-10数据集

## 数据来源

- CIFAR-10数据集
	- 训练集 50000张(3,32,32)
	- 测试集 10000张(3,32,32)

---

## AlexNet模型架构

```python
class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        ) # 卷积层
        self.flat = nn.Flatten() # 展平层
        self.fc = nn.Sequential(
            nn.Linear(6 * 6 * 256, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)
        ) # 全连接层

    def forward(self, x):
        out = self.conv(x) # 卷积
        out = self.flat(out) # 展平
        out = self.fc(out) # 全连接

        return out
```

---

## Improved AlexNet模型架构

```python
class Improved_AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) # 卷积层
        self.flat = nn.Flatten() # 展平层
        self.fc = nn.Sequential(
            nn.Linear(4 * 4 * 64, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        ) # 全连接层

    def forward(self, x):
        out = self.conv(x) # 卷积
        out = self.flat(out) # 展平
        out = self.fc(out) # 全连接

        return out
```

---

## 实验结果

### AlexNet模型 

*EPOCHS=10*

| 指标 | 数值 |
|:---:|:---:|
| Train Loss | 0.6747 |
| Test Loss | 0.9185 |
| Train Accuracy | 76.11% |
| Test Accuracy | 68.80% |

### Improved AlexNet模型 

*EPOCHS=10*

| 指标 | 数值 |
|:---:|:---:|
| Train Loss | 0.6798 |
| Test Loss | 0.6034 |
| Train Accuracy | 77.34% |
| Test Accuracy | 79.46% |

*EPOCHS=50*

| 指标 | 数值 |
|:---:|:---:|
| Train Loss | 0.2627 |
| Test Loss | 0.4938 |
| Train Accuracy | 91.31% |
| Test Accuracy | 85.87% |

---

## 实验评价
>由于AlexNet设计之初是为了ImageNet数据集的分类任务，而CIFAR-10数据集是一个较小的10分类数据集，因此在CIFAR-10数据集上训练AlexNet模型的效果并不理想，10个EPOCHS内的训练准确率只有76.11%，测试准确率只有68.80%。
Improved AlexNet模型在CIFAR-10数据集上要略好于AlexNet模型，得益于增添了部分图像增强模块以及LeakyReLu的改进，10个EPOCHS内的训练准确率达到77.34%，测试准确率达到79.48%，随着EPOCHS的增加，预计在50~60个EPOCHS将会达到两个模型的最佳性能，但提升能力有限迭代后期存在过拟合问题。