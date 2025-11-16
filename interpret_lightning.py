#!/usr/bin/env python3
"""
PyTorch Lightning script for Causal Discovery in CausalFormer
This script performs causal discovery using RRP analysis on a trained prediction model.
"""

import os
import json
import argparse
import torch
import pytorch_lightning as pl
import numpy as np

from data_loader.lightning_data_module import CausalFormerDataModule
from model.lightning_module import CausalFormerLightningModule
from model.interpret_lightning_module import CausalInterpretLightningModule
from pytorch_lightning import seed_everything


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='PyTorch Lightning CausalFormer Causal Discovery')
    parser.add_argument('-c', '--config', default='config/config_lorenz.json', type=str,
                        help='config file path (default: config/config_lorenz.json)')
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help='path to trained model checkpoint')
    parser.add_argument('-d', '--data_dir', type=str, default=None,
                        help='override data directory from config')
    parser.add_argument('-g', '--ground_truth', type=str, default=None,
                        help='path to ground truth file for evaluation')
    parser.add_argument('--gpus', type=int, default=None,
                        help='number of gpus to use (default: from config)')
    parser.add_argument('--output_dir', type=str, default='saved/',
                        help='output directory for results (default: saved/)')
    parser.add_argument('--debug', action='store_true',
                        help='enable debug mode with smaller dataset')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def load_trained_model(model_path: str, config: dict = None, data_module: CausalFormerDataModule = None) -> CausalFormerLightningModule:
    """Load trained model from checkpoint with automatic configuration detection"""
    print(f"Loading trained model from: {model_path}")
    
    # Try to load model with automatic configuration from checkpoint
    try:
        # First attempt: load model with saved hyperparameters
        model = CausalFormerLightningModule.load_from_checkpoint(model_path)
        print("✓ Model loaded with saved hyperparameters")
        
        # If config was provided, verify consistency
        if config is not None:
            saved_config = model.hparams.get('config')
            if saved_config and not configs_match(saved_config, config):
                print("⚠ Warning: Saved configuration differs from provided config")
                print("Using saved configuration from model checkpoint")
        
    except Exception as e:
        print(f"Failed to load with saved hyperparameters: {e}")
        print("Falling back to manual configuration...")
        
        if config is None or data_module is None:
            raise ValueError("Manual configuration fallback requires both config and data_module")
        
        # Fallback: manual configuration
        data_module_params = {
            'time_step': data_module.time_step,
            'output_window': data_module.output_window,
            'feature_dim': data_module.feature_dim,
            'output_dim': data_module.output_dim,
            'series_num': data_module.get_series_num()
        }
        
        model = CausalFormerLightningModule.load_from_checkpoint(
            model_path,
            config=config,
            data_module_params=data_module_params
        )
    
    # Set model to evaluation mode
    model.eval()
    
    return model


def find_model_config(model_path: str) -> dict:
    """Automatically find configuration associated with a model"""
    experiment_dir = os.path.dirname(model_path)
    
    # Try to load JSON config first
    config_path = os.path.join(experiment_dir, 'config.json')
    if os.path.exists(config_path):
        print(f"Found configuration at: {config_path}")
        return load_config(config_path)
    
    # Try to load from hparams.yaml
    hparams_path = os.path.join(experiment_dir, 'hparams.yaml')
    if os.path.exists(hparams_path):
        print(f"Found hyperparameters at: {hparams_path}")
        return load_hparams_config(hparams_path)
    
    # Try to extract from model checkpoint
    try:
        model = CausalFormerLightningModule.load_from_checkpoint(model_path)
        saved_config = model.hparams.get('config')
        if saved_config:
            print("✓ Configuration extracted from model checkpoint")
            return saved_config
    except:
        pass
    
    # If we can't find config in the experiment directory, try parent directories
    # This handles the case where config might be in a parent version directory
    parent_dir = os.path.dirname(experiment_dir)
    if parent_dir and parent_dir != experiment_dir:
        parent_config_path = os.path.join(parent_dir, 'config.json')
        if os.path.exists(parent_config_path):
            print(f"Found configuration in parent directory: {parent_config_path}")
            return load_config(parent_config_path)
    
    raise FileNotFoundError(f"Could not find configuration for model: {model_path}")


def load_hparams_config(hparams_path: str) -> dict:
    """Load configuration from hparams.yaml file"""
    import yaml
    with open(hparams_path, 'r') as f:
        hparams = yaml.safe_load(f)
    return hparams.get('config', {})


def configs_match(config1: dict, config2: dict) -> bool:
    """Check if two configurations are essentially the same"""
    # Compare key hyperparameters
    keys_to_compare = [
        ('arch', 'args', 'd_model'),
        ('arch', 'args', 'n_head'), 
        ('arch', 'args', 'n_layers'),
        ('optimizer', 'args', 'lr'),
        ('data_loader', 'args', 'time_step'),
        ('data_loader', 'args', 'output_window')
    ]
    
    for key_path in keys_to_compare:
        try:
            val1 = config1
            val2 = config2
            for key in key_path:
                val1 = val1[key]
                val2 = val2[key]
            
            if val1 != val2:
                return False
        except (KeyError, TypeError):
            continue
    
    return True


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
        shuffle=False  # Don't shuffle for causal discovery
    )


def setup_interpret_model(
    trained_model: CausalFormerLightningModule, 
    config: dict, 
    data_module: CausalFormerDataModule,
    ground_truth: str = None
) -> CausalInterpretLightningModule:
    """Setup causal interpretation model"""
    # Get the underlying PredictModel
    trained_predict_model = trained_model.get_model()
    
    # Get data parameters
    data_module_params = {
        'time_step': data_module.time_step,
        'output_window': data_module.output_window,
        'feature_dim': data_module.feature_dim,
        'output_dim': data_module.output_dim,
        'series_num': data_module.get_series_num()
    }
    
    # Create the interpretation model
    interpret_model = CausalInterpretLightningModule(
        trained_model=trained_predict_model,
        config=config,
        ground_truth=ground_truth,
        data_module_params=data_module_params
    )
    
    # Set column names from data module
    if hasattr(data_module, 'df_data'):
        interpret_model.columns = list(data_module.df_data.columns)
    
    return interpret_model


def main():
    """Main causal discovery function"""
    args = parse_args()
    
    # Set random seeds for reproducibility (same as original version)
    seed_everything(123, workers=True)
    
    # Try to automatically find configuration from model directory
    try:
        config = find_model_config(args.model_path)
        print(f"✓ Automatically found configuration for model: {args.model_path}")
        auto_config = True
    except FileNotFoundError:
        # Fallback to provided config file
        config = load_config(args.config)
        print(f"Loaded configuration from {args.config}")
        auto_config = False
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup data module
    data_module = setup_data_module(config, args)
    data_module.setup(stage="test")  # Only need test data for causal discovery
    print("Data module setup complete")
    
    # Load trained model (will use automatic config if available)
    if auto_config:
        trained_model = load_trained_model(args.model_path, config=None, data_module=None)
    else:
        trained_model = load_trained_model(args.model_path, config, data_module)
    print("Trained model loaded successfully")
    
    # Setup causal interpretation model
    interpret_model = setup_interpret_model(
        trained_model, config, data_module, args.ground_truth
    )
    print("Causal interpretation model setup complete")
    
    # Determine number of GPUs
    if args.gpus is not None:
        gpus = args.gpus
    else:
        gpus = config.get('n_gpu', 1)
    
    # Setup logger for TensorBoard logging with unified directory structure
    # Use saved/causal_discovery/ with dataset/hyperparameter structure
    causal_discovery_dir = os.path.join(args.output_dir, 'causal_discovery')
    
    # Extract dataset name and hyperparameters from model path for consistent naming
    try:
        # Try to extract from model path
        model_dir = os.path.dirname(args.model_path)
        # Get the version directory name (e.g., lr0.01_d512_h8)
        hyperparam_dir = os.path.basename(os.path.dirname(model_dir))
        # Get the dataset name
        dataset_dir = os.path.basename(os.path.dirname(os.path.dirname(model_dir)))
        run_name = f"{dataset_dir}/{hyperparam_dir}"
    except:
        # Fallback to simple name
        run_name = "causal_discovery"
    
    logger = pl.loggers.TensorBoardLogger(
        save_dir=causal_discovery_dir,
        name=run_name,
        version=None  # Auto-versioning to avoid overwrites
    )
    
    print("Starting causal discovery...")
    print(f"Using {gpus} GPU(s)")
    
    if args.ground_truth:
        print(f"Ground truth file: {args.ground_truth}")
    else:
        print("No ground truth provided - will only perform causal discovery without evaluation")
    
    # Use Lightning training workflow for causal discovery (solves gradient issues and OOM problems)
    print("Starting causal discovery with Lightning training workflow...")
    
    # Setup trainer for causal discovery
    trainer = pl.Trainer(
        accelerator='auto',
        devices=gpus,
        max_epochs=1,  # Only need one epoch for causal discovery
        logger=logger,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        log_every_n_steps=1
    )
    
    # Run causal discovery using training workflow
    trainer.fit(interpret_model, data_module)
    
    # Get results after causal discovery
    causal_results = interpret_model.get_causal_results()
    evaluation_metrics = interpret_model.get_evaluation_metrics()
    columns = interpret_model.columns if interpret_model.columns else [f"Series_{i}" for i in range(interpret_model.series_num)]
    
    print("\n" + "="*50)
    print("Causal Discovery Completed!")
    print("="*50)
    print(f"Discovered {len(causal_results)} causal relationships")
    
    # Save causal relationships to CSV file in the experiment directory
    experiment_dir = logger.log_dir
    os.makedirs(experiment_dir, exist_ok=True)
    results_file = os.path.join(experiment_dir, "causal_discovery_results.csv")
    import pandas as pd
    if causal_results:
        df = pd.DataFrame([
            (columns[cause] if columns else f"Series_{cause}",
             columns[effect] if columns else f"Series_{effect}",
             delay)
            for cause, effect, delay in causal_results
        ], columns=["Cause", "Effect", "Delay"])
        df.to_csv(results_file, index=False)
    else:
        # Create empty file with headers
        pd.DataFrame(columns=["Cause", "Effect", "Delay"]).to_csv(results_file, index=False)
    
    # Log evaluation metrics to TensorBoard
    if args.ground_truth:
        print("\nEvaluation Metrics Summary:")
        print(f"  Precision': {evaluation_metrics['precision_prime']:.4f}")
        print(f"  Recall': {evaluation_metrics['recall_prime']:.4f}")
        print(f"  F1': {evaluation_metrics['f1_prime']:.4f}")
        print(f"  Precision: {evaluation_metrics['precision']:.4f}")
        print(f"  Recall: {evaluation_metrics['recall']:.4f}")
        print(f"  F1: {evaluation_metrics['f1']:.4f}")
        print(f"  PoD: {evaluation_metrics['pod']:.2f}%")
        
        # Log metrics to TensorBoard using the proper method
        # We need to create a trainer to properly log to TensorBoard
        trainer = pl.Trainer(
            accelerator='auto',
            devices=1,
            logger=logger,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False
        )
        
        # Log the metrics
        trainer.logger.log_metrics(evaluation_metrics)
        trainer.logger.save()
    
    print(f"\nResults saved to: {experiment_dir}")
    print(f"CSV results saved to: {results_file}")
    print("All done!")


if __name__ == '__main__':
    main()
