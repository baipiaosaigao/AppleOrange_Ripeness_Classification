import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# ===========================
# 1. 全局配置参数
# ===========================
DATA_DIR = './dataset/train'
MODEL_PTH_PATH = './models/cnn_model.pth'
MODEL_ONNX_PATH = './models/cnn_model.onnx'

BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
INPUT_SIZE = 224

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"当前使用的训练设备: {device}")


# ===========================
# 2. 定义 CNN 网络结构
# ===========================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ===========================
# 3. 训练主函数
# ===========================
def train_model():
    # A. 数据预处理 (已去掉 Normalize，保持 0-1)
    data_transforms = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ])

    if not os.path.exists(DATA_DIR):
        print(f"错误: 找不到数据集目录 {DATA_DIR}")
        return

    full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
    print(f"检测到的类别索引: {full_dataset.class_to_idx}")
    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = SimpleCNN(num_classes=len(full_dataset.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if not os.path.exists('./models'):
        os.makedirs('./models')

    best_acc = 0.0

    print(f"开始训练... 共 {NUM_EPOCHS} 轮")

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        corrects = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = corrects.double() / total_samples

        print(f'Epoch {epoch + 1}/{NUM_EPOCHS} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc * 100:.2f}%', end="")

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model, MODEL_PTH_PATH)
            print(f"  [★ 新纪录! 已保存]")
        else:
            print("")

    print(f"\n训练结束。历史最佳准确率: {best_acc * 100:.2f}%")

    # ---------------------------
    # E. 导出 ONNX (修复了报错点)
    # ---------------------------
    print("正在重新加载最佳模型用于导出 ONNX...")

    # === 【关键修改】 加上 weights_only=False ===
    best_model = torch.load(MODEL_PTH_PATH, weights_only=False)

    best_model.to(device)
    best_model.eval()

    dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(device)

    torch.onnx.export(best_model,
                      dummy_input,
                      MODEL_ONNX_PATH,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'])

    print(f"最佳模型已导出至: {MODEL_ONNX_PATH}")


if __name__ == '__main__':
    train_model()