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


def set_random_seed(seed=123):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


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
    parser.add_argument('--output_dir', type=str, default='saved/lightning_interpret',
                        help='output directory for results (default: saved/lightning_interpret)')
    parser.add_argument('--debug', action='store_true',
                        help='enable debug mode with smaller dataset')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def load_trained_model(model_path: str, config: dict, data_module: CausalFormerDataModule) -> CausalFormerLightningModule:
    """Load trained model from checkpoint"""
    print(f"Loading trained model from: {model_path}")
    
    # Setup data parameters for model initialization
    data_module_params = {
        'time_step': data_module.time_step,
        'output_window': data_module.output_window,
        'feature_dim': data_module.feature_dim,
        'output_dim': data_module.output_dim,
        'series_num': data_module.get_series_num()
    }
    
    # Load the model
    model = CausalFormerLightningModule.load_from_checkpoint(
        model_path,
        config=config,
        data_module_params=data_module_params
    )
    
    # Set model to evaluation mode
    model.eval()
    
    return model


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
    set_random_seed(123)
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup data module
    data_module = setup_data_module(config, args)
    data_module.setup(stage="test")  # Only need test data for causal discovery
    print("Data module setup complete")
    
    # Load trained model
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
    
    print("Starting causal discovery...")
    print(f"Using {gpus} GPU(s)")
    
    if args.ground_truth:
        print(f"Ground truth file: {args.ground_truth}")
    else:
        print("No ground truth provided - will only perform causal discovery without evaluation")
    
    # Use direct method instead of Lightning test workflow
    test_loader = data_module.test_dataloader()
    interpret_model.run_causal_discovery(test_loader)
    
    # Get results
    causal_results = interpret_model.get_causal_results()
    evaluation_metrics = interpret_model.get_evaluation_metrics()
    
    print("\n" + "="*50)
    print("Causal Discovery Completed!")
    print("="*50)
    print(f"Discovered {len(causal_results)} causal relationships")
    
    if args.ground_truth:
        print("\nEvaluation Metrics Summary:")
        print(f"  Precision': {evaluation_metrics['precision_prime']:.4f}")
        print(f"  Recall': {evaluation_metrics['recall_prime']:.4f}")
        print(f"  F1': {evaluation_metrics['f1_prime']:.4f}")
        print(f"  Precision: {evaluation_metrics['precision']:.4f}")
        print(f"  Recall: {evaluation_metrics['recall']:.4f}")
        print(f"  F1: {evaluation_metrics['f1']:.4f}")
        print(f"  PoD: {evaluation_metrics['pod']:.2f}%")
    
    print(f"\nResults saved to: {args.output_dir}")
    print("All done!")


if __name__ == '__main__':
    main()
