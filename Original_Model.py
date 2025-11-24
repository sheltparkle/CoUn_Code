import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.optim.lr_scheduler import CosineAnnealingLR
from CoUn_Utils import *


# ===================== 数据预处理 =====================
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),  # 增强数据，提升泛化性
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 加载全量CIFAR10数据集（训练原始模型需要全量数据）
train_dataset = datasets.CIFAR10(
    root=DATASET_ROOT, train=True, download=True, transform=transform_train
)
test_dataset = datasets.CIFAR10(
    root=DATASET_ROOT, train=False, download=True, transform=transform_test
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True
)


# ===================== 适配CIFAR10的ResNet18 =====================
def build_original_resnet18(num_classes=10):
    """构建适配CIFAR10的ResNet18（调整stem层+预训练权重初始化）"""
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # 1. 调整stem层，适配32×32小尺寸
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    # 2. 重新初始化调整后的层
    nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')

    # 3. 调整全连接层，适配CIFAR10的10类
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    nn.init.normal_(model.fc.weight, mean=0.0, std=0.01)
    nn.init.zeros_(model.fc.bias)

    return model


# ===================== 训练与验证函数 =====================
def train_epoch(model, train_loader, optimizer, loss_fn, epoch):
    model.train()
    total_loss = 0.0
    for data, labels in train_loader:
        data, labels = data.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch{epoch + 1:2d} | Train Loss: {avg_loss:.4f}", end=" | ")
    return avg_loss


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"Test Acc: {acc:.2f}%")
    return acc


# ===================== 主训练流程 =====================
if __name__ == "__main__":
    print("=" * 50)
    print("开始训练本地原始模型（ResNet18-CIFAR10）")
    print(f"设备：{DEVICE} | 训练轮数：{EPOCHS} | 批次大小：{BATCH_SIZE}")
    print("=" * 50)

    # 1. 初始化模型
    original_model = build_original_resnet18(num_classes=10).to(DEVICE)

    # 2. 优化器与损失函数
    optimizer = optim.AdamW(
        original_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑，提升泛化性

    # 3. 训练过程
    best_acc = 0.0
    for epoch in range(EPOCHS):
        train_epoch(original_model, train_loader, optimizer, loss_fn, epoch)
        current_acc = evaluate_model(original_model, test_loader)

        # 保存最优模型
        if current_acc > best_acc:
            best_acc = current_acc
            torch.save(original_model.state_dict(), Original_PATH)
            print(f"✅ 保存最优模型（准确率：{best_acc:.2f}%）")
        else:
            print()

        scheduler.step()

    # 最终结果
    print("=" * 50)
    print(f"训练完成！最优原始模型准确率：{best_acc:.2f}%")
    print(f"原始模型已保存到：{Original_PATH}")