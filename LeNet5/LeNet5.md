# LeNet-5模型——MNIST数据集

## 数据来源

- MNIST数据集
	- 训练集 60000张
	- 测试集 10000张

---

## LeNet-5模型架构
```python
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) # 卷积层
        self.flat = nn.Flatten() # 展平层
        self.fc = nn.Sequential(
            nn.Linear(400, 120),
            nn.ReLU(), 
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        ) # 全连接层

    def forward(self, x):
        out = self.conv(x) # 卷积
        out = self.flat(out) # 展平
        out = self.fc(out) # 全连接

        return out
```

---

## 项目结果

| 指标 | 数值 |
|:---:|:---:|
| Train Loss | 0.0187 |
| Test Loss | 0.0367 |
| Train Accuracy | 99.41% |
| Test Accuracy | 98.80% |

![LeNet-5结果图](LeNet5/figure/LeNet5.png)