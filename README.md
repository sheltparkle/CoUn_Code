# CoUn: Empowering Machine Unlearning via Contrastive Learning 复现代码

本仓库为论文 **《CoUn: Empowering Machine Unlearning via Contrastive Learning》**（arXiv链接：https://arxiv.org/abs/2509.16391 ）的复现代码。

## ⚠️ 免责声明
本仓库中大部分代码由AI工具生成，作者不保证代码的准确性、完整性和正确性。用户使用本代码所产生的一切问题由自身承担。

## 📋 复现流程
### 1. 参数配置（Parameter Configuration）
根据本地环境修改 `CoUn_Utils.py` 中的全局参数：
- `DATASET_ROOT`：CIFAR-10数据集保存路径（若不存在会自动下载）
- `DEVICE`：训练设备（推荐使用 `cuda` 进行GPU加速）
- `Original_PATH`：训练好的原始ResNet18模型保存路径
-  `Unlearn_PATH`：遗忘后的ResNet18模型保存路径
- 训练超参数（批次大小、学习率、训练轮数等）
- CoUn核心参数（温度系数 `TAU`、对比损失权重 `LAMBDA`、待遗忘类别索引等）

### 2. 训练原始模型（Train Original Model）
运行 `Original_Model.py`，在完整CIFAR-10数据集上训练ResNet18模型（该模型作为机器遗忘的基础模型）：
```bash
python Original_Model.py
```
最优性能的原始模型会自动保存到 `SAVE_ORIGINAL_PATH` 指定的路径。

### 3. 执行CoUn主训练（Run CoUn Main Training）
运行 `CoUn_Main.py` 执行机器遗忘训练（仅使用保留数据 `Dr`，不访问遗忘数据 `Du`）：
```bash
python CoUn_Main.py
```
训练过程核心特点：
- 自动从训练集中剔除待遗忘类别，构建保留数据 `Dr`
- 支持自定义 `Dr` 中每个非遗忘类的样本数量（修改 `CoUn_Main.py` 中的 `PER_CLASS_SAMPLES` 参数）
- 采用CE损失（保障模型性能）与InfoNCE损失（实现遗忘功能）联合训练
- 基于总损失保存最优CoUn模型

### 4. 测试遗忘效果（Test Unlearning Effect）
运行 `Test.py` 评估遗忘性能：
```bash
python Test.py
```
测试会输出以下关键指标：
- 遗忘类别准确率（FA）：模型在待遗忘类别上的准确率（越低表示遗忘效果越好）
- 保留类别准确率（RA）：模型在非遗忘类别上的准确率（越高表示性能保留越好）
- 总体测试准确率：模型在完整测试集上的全局准确率
- 遗忘效果指标（UA）：遗忘有效性度量（1 - FA/100，越高表示遗忘效果越好）
- 原始模型与CoUn模型的逐层级权重差异（可选功能）

## 📁 文件说明（File Description）
| 文件名              | 功能描述                                                                 |
|---------------------|--------------------------------------------------------------------------|
| `CoUn_Utils.py`     | 全局参数配置文件（数据集路径、超参数、设备配置等）                       |
| `Original_Model.py` | 在完整CIFAR-10上训练ResNet18，生成机器遗忘所需的原始模型                |
| `CoUn_Main.py`      | CoUn核心训练代码（通过对比学习实现机器遗忘）                             |
| `Test.py`           | 遗忘效果评估工具（类别级准确率 + 权重差异分析）                           |

## 🛠️ 环境依赖（Environment Dependencies）
```
torch >= 1.12.0
torchvision >= 0.13.0
numpy >= 1.21.0
tqdm >= 4.64.0
pandas >= 1.4.0
tabulate >= 0.9.0
```
安装依赖命令：
```bash
pip install torch torchvision numpy tqdm pandas tabulate
```

## 📝 注意事项（Notes）
1. 代码针对CIFAR-10数据集和ResNet18网络；
2. 默认待遗忘类别为 `0（飞机）`，可通过修改 `CoUn_Utils.py` 中的 `FORGET_CLASS` 参数更换目标类别；
3. 训练需保证至少4GB GPU显存（默认批次大小为128）；
4. 随机种子可能导致训练结果存在微小差异，可调整 `CoUn_Main.py` 中的 `SAMPLING_SEED` 参数保证复现性。

