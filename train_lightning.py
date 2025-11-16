#!/usr/bin/env python3
"""
PyTorch Lightning training script for CausalFormer
"""

import os
import json
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything

from data_loader.lightning_data_module import CausalFormerDataModule
from model.lightning_module import CausalFormerLightningModule
from utils.util import prepare_device


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='PyTorch Lightning CausalFormer Training')
    parser.add_argument('-c', '--config', default='config/config_lorenz.json', type=str,
                        help='config file path (default: config/config_lorenz.json)')
    parser.add_argument('-d', '--data_dir', type=str, default=None,
                        help='override data directory from config')
    parser.add_argument('-o', '--output_dir', type=str, default='saved/',
                        help='output directory for saved models (default: saved/)')
    parser.add_argument('--gpus', type=int, default=None,
                        help='number of gpus to use (default: from config)')
    parser.add_argument('--max_epochs', type=int, default=None,
                        help='maximum number of epochs (default: from config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true',
                        help='enable debug mode with smaller dataset')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def setup_data_module(config: dict, args: argparse.Namespace) -> CausalFormerDataModule:
    """Setup data module from configuration"""
    data_loader_config = config['data_loader']['args']
    
    # Override data directory if provided
    if args.data_dir:
        data_loader_config['data_dir'] = args.data_dir
    
    # For debug mode, use smaller batch size and fewer workers
    if args.debug:
        data_loader_config['batch_size'] = min(16, data_loader_config.get('batch_size', 64))
        data_loader_config['num_workers'] = 0
    
    return CausalFormerDataModule(
        data_dir=data_loader_config['data_dir'],
        batch_size=data_loader_config['batch_size'],
        time_step=data_loader_config['time_step'],
        output_window=data_loader_config['output_window'],
        feature_dim=data_loader_config['feature_dim'],
        output_dim=data_loader_config['output_dim'],
        validation_split=data_loader_config.get('validation_split', 0.1),
        num_workers=data_loader_config.get('num_workers', 4),
        shuffle=data_loader_config.get('shuffle', True)
    )


def setup_model(config: dict, data_module: CausalFormerDataModule) -> CausalFormerLightningModule:
    """Setup model from configuration"""
    trainer_config = config.get('trainer', {})
    optimizer_config = config.get('optimizer', {})
    
    # Get data parameters
    data_module_params = {
        'time_step': data_module.time_step,
        'output_window': data_module.output_window,
        'feature_dim': data_module.feature_dim,
        'output_dim': data_module.output_dim,
        'series_num': data_module.get_series_num()
    }
    
    # Extract learning parameters
    learning_rate = optimizer_config.get('args', {}).get('lr', 0.001)
    weight_decay = optimizer_config.get('args', {}).get('weight_decay', 0.0)
    lam = trainer_config.get('lam', 5e-4)
    
    # Learning rate scheduler config
    lr_scheduler_config = config.get('lr_scheduler', None)
    
    return CausalFormerLightningModule(
        config=config,
        data_module_params=data_module_params,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lam=lam,
        lr_scheduler_config=lr_scheduler_config
    )


def setup_callbacks(config: dict, output_dir: str, logger: TensorBoardLogger) -> list:
    """Setup training callbacks with model saved in experiment directory"""
    trainer_config = config.get('trainer', {})
    callbacks = []
    
    # Get experiment directory path
    experiment_dir = logger.log_dir
    
    # Model checkpoint callback - save to experiment directory
    checkpoint_callback = ModelCheckpoint(
        dirpath=experiment_dir,      # Save directly to experiment directory
        filename='best_model-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,  # Only save the best model
        save_last=False,  # Don't save last model
        verbose=False,  # Disable verbose output to avoid tqdm interference
        auto_insert_metric_name=False
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    early_stop = trainer_config.get('early_stop', 0)
    if early_stop > 0:
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.001,  # Small improvement threshold
            patience=early_stop,
            verbose=False,    # Disable verbose to avoid tqdm interference
            mode='min',
            check_finite=True,  # Stop if metric becomes NaN/infinite
            stopping_threshold=None,  # Optional: stop when metric reaches this value
            divergence_threshold=None,  # Optional: stop when metric becomes worse than this
            check_on_train_epoch_end=False  # Check at validation end
        )
        callbacks.append(early_stopping_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Config saver callback - save JSON config to experiment directory
    config_saver_callback = ConfigSaverCallback(config, experiment_dir)
    callbacks.append(config_saver_callback)
    
    return callbacks


class ConfigSaverCallback(Callback):
    """Callback to save configuration as JSON file in experiment directory"""
    
    def __init__(self, config: dict, save_dir: str):
        self.config = config
        self.save_dir = save_dir
    
    def on_fit_start(self, trainer, pl_module):
        """Save configuration at the start of training"""
        # Ensure the experiment directory exists
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Save configuration as JSON
        config_path = os.path.join(self.save_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        print(f"Configuration saved to: {config_path}")


def setup_logger(config: dict, output_dir: str) -> TensorBoardLogger:
    """Setup TensorBoard logger with meaningful naming and auto-versioning"""
    # Extract dataset name from data directory
    data_dir = config['data_loader']['args']['data_dir']
    dataset_name = os.path.basename(os.path.dirname(data_dir))
    
    # Create meaningful run name based on hyperparameters
    arch_config = config['arch']['args']
    optimizer_config = config['optimizer']['args']
    run_prefix = f"lr{optimizer_config['lr']}_d{arch_config['d_model']}_h{arch_config['n_head']}"
    
    # Use clear directory structure: saved/predict_model/
    predict_model_dir = os.path.join(output_dir, 'predict_model')
    
    # Use None for version to enable automatic versioning (version_0, version_1, etc.)
    # This prevents overwriting when running the same config multiple times
    return TensorBoardLogger(
        save_dir=predict_model_dir,
        name=f"{dataset_name}/{run_prefix}",  # Combine dataset and hyperparameters in name
        version=None,  # Auto-versioning to avoid overwrites,
    )


def main():
    """Main training function"""
    args = parse_args()
    
    # Set random seeds for reproducibility
    seed_everything(123, workers=True)
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup data module
    data_module = setup_data_module(config, args)
    print("Data module setup complete")
    
    # Setup model
    model = setup_model(config, data_module)
    print("Model setup complete")
    
    # Setup logger
    logger = setup_logger(config, args.output_dir)
    print("Logger setup complete")
    
    # Setup callbacks (needs logger for experiment directory)
    callbacks = setup_callbacks(config, args.output_dir, logger)
    print("Callbacks setup complete")
    
    # Setup trainer
    trainer_config = config.get('trainer', {})
    
    # Determine number of GPUs
    if args.gpus is not None:
        gpus = args.gpus
    else:
        gpus = config.get('n_gpu', 1)
    
    # Determine max epochs
    if args.max_epochs is not None:
        max_epochs = args.max_epochs
    else:
        max_epochs = trainer_config.get('epochs', 100)
    
    # Set float32 matmul precision for better performance on NVIDIA GPUs
    torch.set_float32_matmul_precision('medium')
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='auto',
        devices=gpus if gpus > 0 else 1,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=5,  # More frequent logging for small datasets
        strategy='auto',
        precision=32,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,
        benchmark=True,
        # Additional performance optimizations
        accumulate_grad_batches=1,
        gradient_clip_val=None,
        gradient_clip_algorithm='norm'
    )
    
    print("Starting training...")
    print(f"Training for {max_epochs} epochs on {gpus} GPU(s)")
    
    # Start training
    trainer.fit(model, datamodule=data_module, ckpt_path=args.resume)
    
    print("Training completed!")
    print(f"Best model saved at: {trainer.checkpoint_callback.best_model_path}")
    
    # Test the model
    print("Starting testing...")
    trainer.test(model, datamodule=data_module)
    
    print("All done!")


if __name__ == '__main__':
    main()
