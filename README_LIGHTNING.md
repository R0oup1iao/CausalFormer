# CausalFormer PyTorch Lightning Implementation

这是CausalFormer的PyTorch Lightning重构版本，遵循PyTorch Lightning的最佳实践，提供了更标准化、更易维护的训练流程。

## 学习路径

对于这个大型项目，建议按照以下步骤逐步学习：

**第一步：环境准备和快速体验**
1. 安装依赖：`pip install -r requirements.txt`
2. 运行快速演示：`python train_lightning.py -c config/config_lorenz.json --debug`
3. 查看训练过程，了解基本工作流程

**第二步：理解数据结构和配置**
1. 查看数据目录结构：`data/` 目录下的各种数据集
2. 学习配置文件格式：`config/` 目录下的JSON配置文件
3. 理解时间序列数据的格式要求

**第三步：核心组件学习**
1. 学习数据模块：`CausalFormerDataModule` 在 `data_loader/lightning_data_module.py`
2. 学习模型模块：`CausalFormerLightningModule` 在 `model/lightning_module.py`
3. 理解PyTorch Lightning的训练流程

**第四步：训练和测试**
1. 使用不同配置进行训练
2. 运行测试脚本：`python test_lightning.py`
3. 查看模型保存和加载机制

**第五步：因果发现**
1. 使用训练好的模型进行因果发现：`python interpret_lightning.py`
2. 学习因果发现模块：`CausalInterpretLightningModule`
3. 理解RRP（回归相关性传播）分析原理

**第六步：高级功能和实验**
1. 查看实验笔记本：`experiments.ipynb`
2. 学习模型解释性功能
3. 尝试不同的数据集和配置

**第七步：深入源码**
1. 阅读原始模型代码：`model/model.py`
2. 理解RRP实现：`model/RRP.py`
3. 学习评估器：`evaluator/evaluator.py`

**第八步：自定义和扩展**
1. 修改配置文件进行实验
2. 添加新的数据集支持
3. 扩展模型功能

## 主要改进

### 1. 模块化设计
- **CausalFormerDataModule**: 统一的数据加载和管理
- **CausalFormerLightningModule**: 集成的模型、训练和验证逻辑
- **CausalInterpretLightningModule**: 因果发现和分析模块
- **标准化训练流程**: 使用PyTorch Lightning Trainer

### 2. 优势
- **自动分布式训练**: 支持多GPU训练，无需手动代码
- **标准化日志记录**: 内置TensorBoard支持
- **模型检查点**: 自动保存最佳模型
- **早停机制**: 内置早停回调
- **学习率调度**: 统一的学习率管理
- **代码可维护性**: 清晰的职责分离

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 使用PyTorch Lightning训练

```bash
# 使用默认配置训练
python train_lightning.py -c config/config_lorenz.json

# 使用自定义数据目录
python train_lightning.py -c config/config_lorenz.json -d data/basic/v/data_0.csv

# 调试模式（小批量）
python train_lightning.py -c config/config_lorenz.json --debug

# 使用多GPU训练
python train_lightning.py -c config/config_lorenz.json --gpus 2
```

### 测试实现

```bash
python test_lightning.py
```

### 因果发现

```bash
# 使用训练好的模型进行因果发现
python interpret_lightning.py -c config/config_lorenz.json -m saved/models/your_model.ckpt

# 使用真实因果图进行评估
python interpret_lightning.py -c config/config_lorenz.json -m saved/lightning_models/checkpoints/best_model-v4.ckpt -g data/lorenz96/groundtruth.csv

# 指定输出目录
python interpret_lightning.py -c config/config_lorenz.json -m saved/models/your_model.ckpt --output_dir saved/causal_results
```

## 新模型保存系统

### 问题背景

之前的模型保存系统存在以下问题：
- 所有模型都保存为 `best_model-v{数字}.ckpt`，无法区分不同配置
- 无法从文件名看出使用了哪个配置文件
- `interpret_lightning.py` 难以运行，因为不知道模型对应的配置

### 新系统解决方案

#### 1. 清晰的目录结构

**预测模型保存位置**：
```
saved/predict_model/
└── {dataset_name}/               # 数据集名称，如 "lorenz96"
    └── {run_name}/               # 运行名称，如 "lr0.01_d512_h8"
        ├── version_0/            # 自动版本号，避免覆盖
        │   ├── events.out.tfevents...
        │   ├── hparams.yaml      # Lightning 自动生成的超参数
        │   ├── config.json       # 我们额外保存的 JSON 配置副本
        │   └── best_model-{epoch}-{val_loss}.ckpt  # 模型文件
        ├── version_1/            # 第二次运行
        │   └── ...
        └── version_2/            # 第三次运行
            └── ...
```

**因果发现结果保存位置**：
```
saved/causal_discovery/
└── {dataset_name}/               # 数据集名称，如 "lorenz96"
    └── {run_name}/               # 运行名称，如 "lr0.01_d512_h8"
        ├── version_0/            # 自动版本号，避免覆盖
        │   ├── events.out.tfevents...
        │   └── causal_discovery_results.csv
        ├── version_1/            # 第二次运行
        │   └── ...
        └── version_2/            # 第三次运行
            └── ...
```

#### 2. 智能命名系统

- **实验名称** = 数据集名称（从数据目录自动提取）
- **运行名称** = 超参数组合（如 `lr0.01_d512_h8`）
- **模型文件名** = `best_model-{epoch}-{val_loss}.ckpt`

#### 3. 自动配置管理

- 训练时自动保存配置到实验目录
- `interpret_lightning.py` 自动查找模型对应的配置
- 无需手动维护多个配置文件

### 使用方法

#### 训练模型（与之前相同）

```bash
python train_lightning.py -c config/config_lorenz.json
```

**新特性**：
- 模型会保存在 `saved/predict_model/lorenz96/lr0.01_d512_h8/version_0/` 目录
- 自动生成 `config.json` 副本
- 模型文件名为 `best_model-epoch50-val0.0123.ckpt`
- 多次运行不会覆盖：`version_0`, `version_1`, `version_2`...

#### 因果发现（改进版）

**方法1：自动配置识别（推荐）**
```bash
python interpret_lightning.py -m saved/predict_model/lorenz96/lr0.01_d512_h8/version_0/best_model-epoch50-val0.0123.ckpt
```

系统会自动：
1. 从模型目录找到 `config.json`
2. 从模型文件加载保存的超参数
3. 验证配置一致性

**方法2：手动指定配置（向后兼容）**
```bash
python interpret_lightning.py -c config/config_lorenz.json -m saved/predict_model/lorenz96/lr0.01_d512_h8/version_0/best_model-epoch50-val0.0123.ckpt
```

### 技术实现

#### 主要改进

1. **`train_lightning.py`**:
   - `setup_logger()`: 智能命名实验和运行
   - `setup_callbacks()`: 模型保存到实验目录
   - `ConfigSaverCallback`: 自动保存配置副本

2. **`interpret_lightning.py`**:
   - `find_model_config()`: 自动查找配置
   - `load_trained_model()`: 智能模型加载
   - `configs_match()`: 配置一致性验证

#### 向后兼容性

- 现有训练命令无需修改
- 现有模型文件仍然可以加载
- 提供了回退机制

### 优势

1. **清晰的命名**: 从文件名就能看出模型信息
2. **自动配置管理**: 无需手动匹配模型和配置
3. **符合标准**: 使用 Lightning 的 `save_hyperparameters()`
4. **易于管理**: 删除实验时配置和模型一起删除
5. **向后兼容**: 现有代码无需大改

### 示例

查看 `demo_new_system.py` 了解完整演示：
```bash
python demo_new_system.py
```

### 故障排除

#### 如果自动配置查找失败

1. 检查模型文件是否存在
2. 检查实验目录是否有 `config.json` 或 `hparams.yaml`
3. 使用手动配置回退：`-c config_file.json`

#### 如果模型加载失败

1. 确保 PyTorch Lightning 版本兼容
2. 检查模型文件是否完整
3. 使用调试模式：`--debug`

## 文件结构

```
CausalFormer/
├── data_loader/
│   ├── data_loaders.py          # 原始数据加载器
│   └── lightning_data_module.py # PyTorch Lightning数据模块
├── model/
│   ├── model.py                 # 原始模型
│   ├── lightning_module.py      # PyTorch Lightning模型模块
│   └── interpret_lightning_module.py # 因果发现Lightning模块
├── train_lightning.py           # 新的训练入口点
├── test_lightning.py            # 测试脚本
├── interpret_lightning.py       # 因果发现入口脚本
└── README_LIGHTNING.md          # 本文档
```

## 核心组件

### CausalFormerDataModule

```python
from data_loader.lightning_data_module import CausalFormerDataModule

data_module = CausalFormerDataModule(
    data_dir="data/lorenz96/timeseries0.csv",
    batch_size=64,
    time_step=32,
    output_window=31,
    feature_dim=1,
    output_dim=1,
    validation_split=0.1,
    num_workers=4
)

# 自动处理数据划分
data_module.setup(stage="fit")
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
```

### CausalFormerLightningModule

```python
from model.lightning_module import CausalFormerLightningModule

# 从配置创建模型
model = CausalFormerLightningModule(
    config=config,
    data_module_params=data_module_params,
    learning_rate=0.001,
    weight_decay=0.0,
    lam=5e-4
)
```

### CausalInterpretLightningModule

```python
from model.interpret_lightning_module import CausalInterpretLightningModule

# 使用训练好的模型进行因果发现
interpret_model = CausalInterpretLightningModule(
    trained_model=trained_predict_model,
    config=config,
    ground_truth="data/lorenz96/groundtruth.csv",
    data_module_params=data_module_params
)

# 运行因果发现
interpret_model.run_causal_discovery(test_loader)

# 获取结果
causal_results = interpret_model.get_causal_results()
metrics = interpret_model.get_evaluation_metrics()
```

### 训练流程

```python
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# 设置回调
callbacks = [
    ModelCheckpoint(
        dirpath='checkpoints/',
        filename='causalformer-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min'
    ),
    EarlyStopping(monitor='val_loss', patience=10)
]

# 创建训练器
trainer = pl.Trainer(
    max_epochs=100,
    accelerator='auto',
    devices='auto',
    callbacks=callbacks,
    logger=True
)

# 开始训练
trainer.fit(model, datamodule=data_module)
```

## 因果发现

### 工作原理

因果发现模块使用RRP（回归相关性传播）分析来发现时间序列间的因果关系：

1. **模型训练**: 首先训练一个预测模型来学习时间序列的动态
2. **相关性传播**: 使用RRP分析模型的注意力机制和卷积权重
3. **聚类分析**: 通过K-means聚类识别重要的因果关系
4. **延迟检测**: 确定因果关系的具体时间延迟

### 使用方法

```bash
# 基本用法
python interpret_lightning.py -c config/config_lorenz.json -m saved/models/model.ckpt

# 带评估的用法
python interpret_lightning.py -c config/config_lorenz.json -m saved/models/model.ckpt -g data/lorenz96/groundtruth.csv

# 参数说明
# -c, --config: 配置文件路径
# -m, --model_path: 训练好的模型检查点路径
# -g, --ground_truth: 真实因果图文件路径（可选）
# --output_dir: 输出目录（默认：saved/lightning_interpret）
# --gpus: GPU数量
# --debug: 调试模式
```

### 输出结果

因果发现模块会输出：

- **发现的因果关系**: 格式为 (原因, 结果, 延迟)
- **评估指标**:
  - Precision': 扩展因果图的精确度
  - Recall': 扩展因果图的召回率
  - F1': 扩展因果图的F1分数
  - Precision: 直接因果图的精确度
  - Recall: 直接因果图的召回率
  - F1: 直接因果图的F1分数
  - PoD: 延迟发现百分比

### 示例输出

```
Causal Discovery Completed!
==================================================
Discovered 15 causal relationships

Evaluation Metrics Summary:
  Precision': 0.8571
  Recall': 0.7500
  F1': 0.8000
  Precision: 0.8000
  Recall: 0.6667
  F1: 0.7273
  PoD: 85.71%

Discovered Causal Relationships:
  Series_0 -> Series_1 (delay: 2)
  Series_1 -> Series_2 (delay: 1)
  Series_3 -> Series_4 (delay: 3)
```

## 配置说明

PyTorch Lightning版本使用与原始版本相同的JSON配置文件，但添加了以下增强功能：

- **动态series_num**: 自动从数据中获取时间序列数量
- **灵活的设备管理**: 自动处理CPU/GPU训练
- **改进的日志记录**: 详细的训练指标跟踪
- **因果发现配置**: 解释器参数配置

### 因果发现配置参数

在配置文件的`explainer`部分可以设置因果发现参数：

```json
"explainer": {
    "m": 2,    // 考虑的前m个重要聚类
    "n": 10    // 总聚类数量
}
```

### 示例配置

```json
{
    "name": "Causality Learning",
    "n_gpu": 1,
    "arch": {
        "type": "PredictModel",
        "args": {
            "d_model": 512,
            "n_head": 8,
            "n_layers": 1,
            "ffn_hidden": 512,
            "drop_prob": 0,
            "tau": 10
        }
    },
    "data_loader": {
        "type": "TimeseriesDataLoader",
        "args": {
            "data_dir": "data/lorenz96/timeseries0.csv",
            "batch_size": 128,
            "time_step": 32,
            "output_window": 31,
            "feature_dim": 1,
            "output_dim": 1,
            "shuffle": true,
            "validation_split": 0.1,
            "num_workers": 2
        }
    },
    "optimizer": {
        "type": "Adam",
        "args": {
            "lr": 0.01,
            "weight_decay": 0,
            "amsgrad": true
        }
    },
    "loss": "masked_mse_torch",
    "metrics": ["masked_mse_torch"],
    "lr_scheduler": {
        "type": "StepLR",
        "args": {
            "step_size": 30,
            "gamma": 0.1
        }
    },
    "trainer": {
        "epochs": 100,
        "save_dir": "saved/",
        "save_period": 1,
        "verbosity": 0,
        "monitor": "min val_loss",
        "early_stop": 10,
        "lam": 5e-4,
        "tensorboard": true
    },
    "explainer": {
        "m": 2,
        "n": 10
    }
}
```

## 模型解释性

PyTorch Lightning版本保留了原始模型的所有解释性功能：

```python
# 获取注意力权重
attention_weights = model.get_attention_weights()
causal_conv_weights = attention_weights['causal_conv']
attention_weights_attn = attention_weights['multi_variate_attention']

# 使用回归相关性传播
rel = model.relprop(output)

# 因果发现
interpret_model = CausalInterpretLightningModule(trained_model, config)
causal_results = interpret_model.run_causal_discovery(test_loader)
```

## 性能保证

PyTorch Lightning重构版本确保：

1. **模型架构一致**: 保持原始CausalFormer的所有组件
2. **训练逻辑等价**: 损失函数、正则化、优化器配置保持一致
3. **结果可复现**: 相同的输入产生相同的输出
4. **性能不下降**: 训练收敛性和模型精度与原始版本一致
5. **因果发现准确**: RRP分析和因果发现功能完整保留

## 迁移指南

### 从原始版本迁移

1. **数据加载**: 使用`CausalFormerDataModule`替代原始数据加载器
2. **模型训练**: 使用`CausalFormerLightningModule`和PyTorch Lightning Trainer
3. **因果发现**: 使用`CausalInterpretLightningModule`进行因果分析
4. **配置**: 保持相同的JSON配置文件
5. **推理**: 使用`model.predict()`方法进行批量预测

### 代码示例对比

**原始版本:**
```python
from trainer.trainer import Trainer
from data_loader.data_loaders import TimeseriesDataLoader

data_loader = TimeseriesDataLoader(...)
model = PredictModel(...)
trainer = Trainer(model, criterion, ...)
trainer.train()
```

**PyTorch Lightning版本:**
```python
from data_loader.lightning_data_module import CausalFormerDataModule
from model.lightning_module import CausalFormerLightningModule
import pytorch_lightning as pl

data_module = CausalFormerDataModule(...)
model = CausalFormerLightningModule(...)
trainer = pl.Trainer(...)
trainer.fit(model, datamodule=data_module)
```

**因果发现对比:**

**原始版本:**
```python
# 需要手动实现因果发现流程
```

**PyTorch Lightning版本:**
```python
from model.interpret_lightning_module import CausalInterpretLightningModule

interpret_model = CausalInterpretLightningModule(trained_model, config)
interpret_model.run_causal_discovery(test_loader)
results = interpret_model.get_causal_results()
```

## 故障排除

### 常见问题

1. **GPU内存不足**: 减小`batch_size`或使用梯度累积
2. **训练速度慢**: 增加`num_workers`或使用混合精度训练
3. **验证损失不下降**: 调整学习率或检查数据预处理
4. **因果发现结果不理想**: 调整`explainer`配置中的`m`和`n`参数

### 调试技巧

```bash
# 启用调试模式
python train_lightning.py --debug
python interpret_lightning.py --debug

# 查看详细日志
python train_lightning.py --config your_config.json --verbose

# 检查模型加载
python interpret_lightning.py -c config.json -m model.ckpt --debug
```

## 贡献

欢迎提交Issue和Pull Request来改进这个PyTorch Lightning实现！

## 许可证

本项目采用GPL-3.0许可证。详见LICENSE文件。
