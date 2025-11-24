import torch
import torch.nn as nn
import torchvision
import numpy as np
import os
import pandas as pd
from tabulate import tabulate
from torch.utils.data import DataLoader, Subset
from CoUn_Utils import *
# ===================== 全局配置 =====================
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]



# ===================== 1. 模型定义（与训练时完全一致） =====================
def build_resnet18_base(num_classes=10):
    """构建与训练时一致的ResNet18（仅调整stem层+fc层，无多余dropout）"""
    model = torchvision.models.resnet18(weights=None)
    # 适配CIFAR10的stem层
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    # 调整分类头
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ===================== 2. 权重加载（处理backbone前缀） =====================
def load_model(weight_path, is_coun_model=False):
    """
    加载模型并处理权重前缀：
    - is_coun_model=True：移除backbone.前缀
    - is_coun_model=False：直接加载原始模型权重
    """
    # 构建基础模型
    model = build_resnet18_base(num_classes=NUM_CLASSES)
    # 加载权重
    state_dict = torch.load(weight_path, map_location=DEVICE)

    # 处理CoUn模型的backbone前缀
    if is_coun_model:
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("backbone."):
                new_k = k.replace("backbone.", "")
                new_state_dict[new_k] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict

    # 加载权重（strict=False兼容num_batches_tracked等无关key）
    model.load_state_dict(state_dict, strict=False)
    model = model.to(DEVICE).eval()
    return model


# ===================== 3. 数据加载（准确率测试用） =====================
def get_cifar10_test_loader():
    """加载CIFAR10测试集（与训练时一致的预处理）"""
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_dataset = torchvision.datasets.CIFAR10(
        root=DATASET_ROOT, train=False, download=True, transform=transform_test
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )
    return test_loader


# ===================== 4. 类别准确率计算与对比 =====================
def calculate_class_accuracy(model, test_loader):
    """计算模型的类别级准确率"""
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)

            # 统计每个类别的预测结果
            for label, pred in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1

    # 计算类别准确率
    class_acc = [100 * (correct / total) if total > 0 else 0.0
                 for correct, total in zip(class_correct, class_total)]
    # 计算总体准确率
    overall_acc = 100 * sum(class_correct) / sum(class_total)

    return class_acc, overall_acc, class_correct, class_total


def compare_accuracy(teacher_acc, coun_acc):
    """对比两个模型的类别准确率差异"""
    acc_diff = [coun - teacher for coun, teacher in zip(coun_acc, teacher_acc)]
    return acc_diff


# ===================== 5. 权重差异计算与对比 =====================
def calculate_weight_diff(teacher_param, coun_param):
    """计算单一层的权重差异（L2距离、余弦相似度、相对误差）"""
    # 展平为一维向量
    t_flat = teacher_param.detach().cpu().flatten().numpy()
    c_flat = coun_param.detach().cpu().flatten().numpy()

    # 1. L2距离
    l2_dist = np.linalg.norm(t_flat - c_flat)
    # 2. 余弦相似度
    norm_t, norm_c = np.linalg.norm(t_flat), np.linalg.norm(c_flat)
    cos_sim = np.dot(t_flat, c_flat) / (norm_t * norm_c + 1e-8) if (norm_t > 1e-8 and norm_c > 1e-8) else 0.0
    # 3. 平均相对误差
    rel_error = np.mean(np.abs(t_flat - c_flat) / (np.abs(t_flat) + 1e-8))

    return {
        "l2_distance": l2_dist,
        "cosine_similarity": cos_sim,
        "relative_error": rel_error,
        "is_high_diff": rel_error > DIFF_THRESHOLD
    }


def compare_layer_weights(teacher_model, coun_model):
    """逐层级对比两个模型的权重差异"""
    teacher_params = dict(teacher_model.named_parameters())
    coun_params = dict(coun_model.named_parameters())

    diff_results = []
    summary = {
        "total_layers": 0,
        "high_diff_layers": 0,
        "avg_l2": 0.0,
        "avg_cos_sim": 0.0,
        "avg_rel_error": 0.0,
        "high_diff_layer_names": []
    }

    # 遍历所有层
    for layer_name in sorted(teacher_params.keys()):
        if layer_name not in coun_params:
            print(f"⚠️  跳过不存在的层：{layer_name}")
            continue

        t_param = teacher_params[layer_name]
        c_param = coun_params[layer_name]

        # 检查参数形状是否一致
        if t_param.shape != c_param.shape:
            print(f"⚠️  跳过形状不匹配的层：{layer_name}（教师：{t_param.shape}，CoUn：{c_param.shape}）")
            continue

        # 计算差异
        diff = calculate_weight_diff(t_param, c_param)
        diff_results.append({
            "layer_name": layer_name,
            "shape": str(t_param.shape),
            "param_num": np.prod(t_param.shape),
            "l2": diff["l2_distance"],
            "cos_sim": diff["cosine_similarity"],
            "rel_error": diff["relative_error"],
            "high_diff": diff["is_high_diff"]
        })

        # 更新汇总统计
        summary["total_layers"] += 1
        summary["avg_l2"] += diff["l2_distance"]
        summary["avg_cos_sim"] += diff["cosine_similarity"]
        summary["avg_rel_error"] += diff["relative_error"]
        if diff["is_high_diff"]:
            summary["high_diff_layers"] += 1
            summary["high_diff_layer_names"].append(layer_name)

    # 计算平均值
    if summary["total_layers"] > 0:
        summary["avg_l2"] /= summary["total_layers"]
        summary["avg_cos_sim"] /= summary["total_layers"]
        summary["avg_rel_error"] /= summary["total_layers"]

    return diff_results, summary


# ===================== 6. 报告输出 =====================
def print_accuracy_report(teacher_acc, coun_acc, acc_diff, overall_teacher, overall_coun):
    """打印准确率对比报告"""
    print("\n" + "=" * 120)
    print("【第一步：类别准确率对比报告（原始模型 vs CoUn遗忘模型）】")
    print("=" * 120)

    # 构建准确率表格
    table_data = []
    for i in range(NUM_CLASSES):
        is_forget = "✅" if i == FORGET_CLASS else ""
        table_data.append([
            i, CIFAR10_CLASSES[i], is_forget,
            f"{teacher_acc[i]:.2f}%", f"{coun_acc[i]:.2f}%", f"{acc_diff[i]:+.2f}%"
        ])

    # 添加总体行
    table_data.append([
        "总计", "-", "-",
        f"{overall_teacher:.2f}%", f"{overall_coun:.2f}%", f"{overall_coun - overall_teacher:+.2f}%"
    ])

    # 打印表格
    headers = ["类别ID", "类别名称", "是否遗忘", "原始模型准确率", "CoUn模型准确率", "准确率差值"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def print_weight_diff_report(diff_results, summary):
    """打印权重差异报告"""
    print("\n" + "=" * 120)
    print("【第二步：逐层级权重差异报告（原始模型 vs CoUn遗忘模型）】")
    print("=" * 120)

    # 汇总统计
    print("\n【权重差异汇总】")
    print(f"1. 总对比层数：{summary['total_layers']}")
    print(f"2. 高差异层数（相对误差> {DIFF_THRESHOLD}）：{summary['high_diff_layers']}")
    print(f"3. 平均L2距离：{summary['avg_l2']:.4f}（越大差异越大）")
    print(f"4. 平均余弦相似度：{summary['avg_cos_sim']:.4f}（越接近1越相似）")
    print(f"5. 平均相对误差：{summary['avg_rel_error']:.4f}（越大差异越大）")
    if summary["high_diff_layer_names"]:
        print(f"6. 高差异层列表：{', '.join(summary['high_diff_layer_names'])}")

    # 逐层级详情
    print("\n【逐层级权重差异详情】")
    table_data = []
    for res in diff_results:
        table_data.append([
            res["layer_name"],
            res["shape"],
            res["param_num"],
            f"{res['l2']:.4f}",
            f"{res['cos_sim']:.4f}",
            f"{res['rel_error']:.4f}",
            "✅" if res["high_diff"] else "❌"
        ])
    headers = ["层名", "参数形状", "参数总数", "L2距离", "余弦相似度", "相对误差", "是否高差异"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


# ===================== 7. 主流程 =====================
if __name__ == "__main__":
    # 0. 检查文件
    if not os.path.exists(Original_PATH):
        raise FileNotFoundError(f"原始模型不存在：{Original_PATH}")
    if not os.path.exists(Unlearn_PATH):
        raise FileNotFoundError(f"CoUn模型不存在：{Unlearn_PATH}")

    # 1. 加载模型
    print("🔍 加载原始模型...")
    teacher_model = load_model(Original_PATH, is_coun_model=False)
    print("🔍 加载CoUn遗忘模型...")
    coun_model = load_model(Unlearn_PATH, is_coun_model=True)

    # 2. 加载测试集
    print("🔍 加载CIFAR10测试集...")
    test_loader = get_cifar10_test_loader()

    # 3. 计算并对比准确率
    print("\n📊 计算类别准确率...")
    teacher_class_acc, teacher_overall_acc, _, _ = calculate_class_accuracy(teacher_model, test_loader)
    coun_class_acc, coun_overall_acc, _, _ = calculate_class_accuracy(coun_model, test_loader)
    acc_diff = compare_accuracy(teacher_class_acc, coun_class_acc)

    # 打印准确率报告
    print_accuracy_report(
        teacher_class_acc, coun_class_acc, acc_diff,
        teacher_overall_acc, coun_overall_acc
    )

    # 4. 计算并对比权重差异
    print("\n📈 计算逐层级权重差异...")
    weight_diff_results, weight_summary = compare_layer_weights(teacher_model, coun_model)

    # 打印权重差异报告
    print_weight_diff_report(weight_diff_results, weight_summary)


