# CausalFormer PyTorch Lightning Implementation

这是CausalFormer的PyTorch Lightning重构版本，遵循PyTorch Lightning的最佳实践，提供了更标准化、更易维护的训练流程。

## 主要改进

### 1. 模块化设计
- **CausalFormerDataModule**: 统一的数据加载和管理
- **CausalFormerLightningModule**: 集成的模型、训练和验证逻辑
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

## 文件结构

```
CausalFormer/
├── data_loader/
│   ├── data_loaders.py          # 原始数据加载器
│   └── lightning_data_module.py # PyTorch Lightning数据模块
├── model/
│   ├── model.py                 # 原始模型
│   └── lightning_module.py      # PyTorch Lightning模型模块
├── train_lightning.py           # 新的训练入口点
├── test_lightning.py            # 测试脚本
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

## 配置说明

PyTorch Lightning版本使用与原始版本相同的JSON配置文件，但添加了以下增强功能：

- **动态series_num**: 自动从数据中获取时间序列数量
- **灵活的设备管理**: 自动处理CPU/GPU训练
- **改进的日志记录**: 详细的训练指标跟踪

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
```

## 性能保证

PyTorch Lightning重构版本确保：

1. **模型架构一致**: 保持原始CausalFormer的所有组件
2. **训练逻辑等价**: 损失函数、正则化、优化器配置保持一致
3. **结果可复现**: 相同的输入产生相同的输出
4. **性能不下降**: 训练收敛性和模型精度与原始版本一致

## 迁移指南

### 从原始版本迁移

1. **数据加载**: 使用`CausalFormerDataModule`替代原始数据加载器
2. **模型训练**: 使用`CausalFormerLightningModule`和PyTorch Lightning Trainer
3. **配置**: 保持相同的JSON配置文件
4. **推理**: 使用`model.predict()`方法进行批量预测

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

## 故障排除

### 常见问题

1. **GPU内存不足**: 减小`batch_size`或使用梯度累积
2. **训练速度慢**: 增加`num_workers`或使用混合精度训练
3. **验证损失不下降**: 调整学习率或检查数据预处理

### 调试技巧

```bash
# 启用调试模式
python train_lightning.py --debug

# 查看详细日志
python train_lightning.py --config your_config.json --verbose
```

## 贡献

欢迎提交Issue和Pull Request来改进这个PyTorch Lightning实现！

## 许可证

本项目采用GPL-3.0许可证。详见LICENSE文件。
