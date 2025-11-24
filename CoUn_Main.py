import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
import numpy as np
from tqdm import tqdm
import warnings
import random 
from CoUn_Utils import *
warnings.filterwarnings('ignore')





# ========================== 数据预处理 ==========================
def get_cifar10_datasets():
    """
    加载CIFAR10数据集并划分保留数据Dr：
    - Dr：训练集中剔除待遗忘类别（0），且每个非遗忘类可自定义采样样本数
    - 数据增强策略与原始模型训练完全一致，保证数据分布匹配
    """
    # 固定随机种子（采样可复现）
    random.seed(SAMPLING_SEED)
    np.random.seed(SAMPLING_SEED)

    # 复用原始模型训练的transform（保证数据增强策略一致）
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  # 原始模型训练的增强策略
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 加载完整CIFAR10训练集（先不应用transform，后续在Subset中生效）
    full_train_set = torchvision.datasets.CIFAR10(
        root=DATASET_ROOT, train=True, download=True, transform=train_transform
    )
    train_labels = np.array(full_train_set.targets)

    # 步骤1：筛选所有非遗忘类的样本索引（类别1-9）
    non_forget_classes = [c for c in range(NUM_CLASSES) if c != FORGET_CLASS]
    class_indices = {}  # 键：类别，值：该类所有样本索引
    for c in non_forget_classes:
        class_indices[c] = np.where(train_labels == c)[0].tolist()

    # 步骤2：对每个非遗忘类采样指定数量样本
    dr_indices = []
    for c in non_forget_classes:
        indices = class_indices[c]
        # 若指定样本数超过该类总数，取全部；否则随机采样
        sample_num = len(indices) if PER_CLASS_SAMPLES is None else min(PER_CLASS_SAMPLES, len(indices))
        sampled_indices = random.sample(indices, sample_num)
        dr_indices.extend(sampled_indices)
        print(
            f"📊 类别{c}（{['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'][c]}）采样样本数：{sample_num}")

    # 步骤3：构建Dr数据集（自定义采样后的保留数据）
    dr_set = Subset(full_train_set, dr_indices)
    print(f"\n📝 Dr数据集总样本数：{len(dr_set)}（非遗忘类共{len(non_forget_classes)}个）")

    return {"dr": dr_set}


def get_dataloaders(datasets):
    """构建数据加载器（与原始模型训练的加载器配置一致）"""
    dataloaders = {
        "dr": DataLoader(
            datasets["dr"],
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True  # 加速GPU数据传输（与原始模型训练一致）
        )
    }
    return dataloaders


# ========================== 模型定义（与原始模型结构对齐 + CoUn封装） ==========================
def build_resnet18_coun(num_classes=NUM_CLASSES):
    """
    构建与原始模型完全一致的ResNet18结构：
    - 调整stem层（conv1 + maxpool）适配CIFAR10
    - 移除classifier别名，确保与原始模型权重key完全匹配
    """
    # 初始化ResNet18（无预训练权重）
    model = torchvision.models.resnet18(pretrained=False)

    # 适配CIFAR10的stem层（与原始模型结构一致）
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    # 调整分类头（与原始模型一致）
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


class CoUnWrapper(nn.Module):
    """
    CoUn模型封装（对齐论文双模块共享参数设计）：
    - 骨干网络：与原始模型一致的ResNet18
    - 前向输出：logits（用于CE损失） + 归一化特征（用于CL损失）
    """

    def __init__(self, backbone):
        super(CoUnWrapper, self).__init__()
        self.backbone = backbone  # 复用原始模型的ResNet18骨干

    def forward(self, x):
        """
        前向传播逻辑（论文双模块共享参数）：
        Args:
            x: 输入图像 (B, 3, 32, 32)
        Returns:
            logits: 分类输出 (B, 10) —— 用于CE损失
            norm_feat: 归一化特征 (B, 512) —— 用于CL损失
        """
        # ResNet18原生前向流程（与原始模型一致）
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # 特征提取（avgpool + 展平）
        feat = self.backbone.avgpool(x)
        feat = torch.flatten(feat, 1)

        # 分类输出（使用原始模型的fc层，无classifier别名）
        logits = self.backbone.fc(feat)

        # 特征L2归一化（论文CL模块强制要求）
        norm_feat = F.normalize(feat, p=2, dim=1)

        return logits, norm_feat


# ========================== 损失函数（严格复刻CoUn论文公式） ==========================
class InfoNCE_Loss(nn.Module):
    """
    InfoNCE损失实现（严格对齐CoUn论文公式）：
    L_CL = -1/(2N) * Σ [log(exp(sim(f1,f2)/τ)/Σexp(sim(f1,fj)/τ)) + 对称项]
    """

    def __init__(self, tau=TAU):
        super(InfoNCE_Loss, self).__init__()
        self.tau = tau  # 温度系数

    def forward(self, norm_feats):
        """
        Args:
            norm_feats: 归一化特征 (2N, 512) —— 每个样本的两个正视图特征拼接
        Returns:
            cl_loss: InfoNCE损失值（标量）
        """
        batch_size = norm_feats.shape[0] // 2  # 原始批次大小N（输入为2N）
        # 拆分两个正视图特征
        feat1, feat2 = norm_feats[:batch_size], norm_feats[batch_size:]

        # 计算余弦相似度矩阵（sim(fi, fj) = fi·fj）
        sim_matrix = torch.mm(norm_feats, norm_feats.t()) / self.tau
        # 掩码排除自身相似度（对角线置为极小值）
        mask = torch.eye(2 * batch_size, device=DEVICE).bool()
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)

        # 正样本对相似度（feat1与feat2对应位置）
        pos_sim = torch.sum(feat1 * feat2, dim=1) / self.tau

        # 计算InfoNCE损失（论文公式）
        loss1 = -pos_sim + torch.logsumexp(sim_matrix[:batch_size], dim=1)
        loss2 = -pos_sim + torch.logsumexp(sim_matrix[batch_size:], dim=1)
        cl_loss = (loss1 + loss2).mean() / 2

        return cl_loss


# ========================== CoUn核心训练流程（对齐论文 + 适配原始模型） ==========================
def train_coun():
    """
    CoUn核心训练流程（无遗忘数据Du访问，仅用保留数据Dr）：
    1. 加载原始模型权重，初始化CoUn模型
    2. 双损失协同训练：CE损失（保效） + CL损失（遗忘）
    3. 保存最优CoUn模型（基于总损失）
    """
    # 步骤1：加载数据集和数据加载器（支持自定义Dr样本数）
    datasets = get_cifar10_datasets()
    dataloaders = get_dataloaders(datasets)

    # 步骤2：初始化模型并加载原始模型权重
    print(f"\n🔍 加载预训练原始模型：{Original_PATH}")
    backbone = build_resnet18_coun()
    # 加载原始模型权重（key完全匹配：conv1、maxpool、fc等）
    backbone.load_state_dict(torch.load(Original_PATH, map_location=DEVICE))
    # 封装为CoUn模型（添加CL分支）
    coun_model = CoUnWrapper(backbone).to(DEVICE)
    coun_model.train()

    # 步骤3：定义损失函数和优化器
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 与原始模型训练一致
    cl_criterion = InfoNCE_Loss(tau=TAU)  # 论文CL损失
    # 优化器（AdamW，与原始模型训练优化器一致）
    optimizer = optim.AdamW(
        coun_model.parameters(),
        lr=LR_COUN,
        weight_decay=WEIGHT_DECAY
    )
    # 学习率调度器（余弦退火，与原始模型训练一致）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_COUN, eta_min=1e-6)

    # 步骤4：CoUn训练主循环
    print("\n===== 开始训练CoUn模型（仅使用保留数据Dr） =====")
    print(f"设备：{DEVICE} | 训练轮数：{EPOCHS_COUN} | CL损失权重λ：{LAMBDA}")
    print("=" * 60)

    best_total_loss = float('inf')


    for epoch in range(EPOCHS_COUN):
        running_ce_loss = 0.0
        running_cl_loss = 0.0
        running_total_loss = 0.0

        # 批次迭代
        pbar = tqdm(dataloaders["dr"], desc=f"CoUn Epoch {epoch + 1}/{EPOCHS_COUN}")
        for inputs, labels in pbar:
            # 生成每个样本的两个正视图（论文要求，批次翻倍为2N）
            inputs = torch.cat([inputs, inputs], dim=0)
            labels = labels.to(DEVICE)
            inputs = inputs.to(DEVICE)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            logits, norm_feats = coun_model(inputs)

            # 计算CE损失（仅用前N个样本的logits，避免增强噪声）
            ce_logits = logits[:len(labels)]
            ce_loss = ce_criterion(ce_logits, labels)

            # 计算CL损失（用2N个样本的归一化特征）
            cl_loss = cl_criterion(norm_feats)

            # 总损失（论文公式：L_total = L_CE + λ*L_CL）
            total_loss = ce_loss + LAMBDA * cl_loss

            # 反向传播 + 参数更新
            total_loss.backward()
            optimizer.step()

            # 累计损失（按样本数加权）
            running_ce_loss += ce_loss.item() * len(labels)
            running_cl_loss += cl_loss.item() * len(labels)
            running_total_loss += total_loss.item() * len(labels)

            # 实时打印批次损失
            pbar.set_postfix({
                "CE Loss": f"{ce_loss.item():.4f}",
                "CL Loss": f"{cl_loss.item():.4f}",
                "Total Loss": f"{total_loss.item():.4f}"
            })

        # 计算Epoch级损失
        epoch_ce_loss = running_ce_loss / len(dataloaders["dr"].dataset)
        epoch_cl_loss = running_cl_loss / len(dataloaders["dr"].dataset)
        epoch_total_loss = running_total_loss / len(dataloaders["dr"].dataset)

        # 更新学习率
        scheduler.step()

        # 打印Epoch总结
        print(f"\nEpoch {epoch + 1} 训练总结：")
        print(f"CE损失：{epoch_ce_loss:.4f} | CL损失：{epoch_cl_loss:.4f} | 总损失：{epoch_total_loss:.4f}")
        print(f"当前学习率：{scheduler.get_last_lr()[0]:.6f}")

        # 保存最优模型（基于总损失）
        if epoch_total_loss < best_total_loss:
            best_total_loss = epoch_total_loss
            torch.save(coun_model.state_dict(), Unlearn_PATH)
            print(f"✅ 保存最优CoUn模型（总损失：{best_total_loss:.4f}）→ {Unlearn_PATH}")
        print("-" * 60)

    # 训练完成
    print("\n===== CoUn训练完成 =====")
    print(f"最优CoUn模型保存路径：{Unlearn_PATH}")
    return coun_model


# ========================== 主流程（一键运行） ==========================
if __name__ == "__main__":
    # 检查原始模型权重是否存在
    if not os.path.exists(Original_PATH):
        raise FileNotFoundError(f"❌ 预训练原始模型不存在：{Original_PATH}")

    # 打印自定义采样配置
    if PER_CLASS_SAMPLES is None:
        print(f"🔧 采样配置：Dr中每个非遗忘类取全部样本")
    else:
        print(f"🔧 采样配置：Dr中每个非遗忘类取{PER_CLASS_SAMPLES}个样本（种子：{SAMPLING_SEED}）")

    # 启动CoUn训练
    train_coun()