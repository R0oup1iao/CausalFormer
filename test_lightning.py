#!/usr/bin/env python3
"""
Test script to verify PyTorch Lightning implementation of CausalFormer
"""

import os
import sys
import torch
import pytorch_lightning as pl

# Add current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader.lightning_data_module import CausalFormerDataModule
from model.lightning_module import CausalFormerLightningModule


def test_data_module():
    """Test the data module with a small dataset"""
    print("Testing Data Module...")
    
    try:
        # Use a small dataset for testing
        data_module = CausalFormerDataModule(
            data_dir="data/basic/v/data_0.csv",
            batch_size=4,
            time_step=16,
            output_window=15,
            feature_dim=1,
            output_dim=1,
            validation_split=0.2,
            num_workers=0,  # Use 0 workers for testing
            shuffle=True
        )
        
        # Setup the data module
        data_module.setup(stage="fit")
        
        # Get a sample batch
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        
        # Check batch shapes
        for batch_idx, (data, target) in enumerate(train_loader):
            print(f"Train batch {batch_idx}: data shape {data.shape}, target shape {target.shape}")
            if batch_idx >= 1:  # Just check first 2 batches
                break
        
        for batch_idx, (data, target) in enumerate(val_loader):
            print(f"Val batch {batch_idx}: data shape {data.shape}, target shape {target.shape}")
            if batch_idx >= 1:  # Just check first 2 batches
                break
        
        print("✓ Data module test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Data module test failed: {e}")
        return False


def test_model_module():
    """Test the model module with a simple configuration"""
    print("\nTesting Model Module...")
    
    try:
        # Create a simple config
        config = {
            'n_gpu': 0,
            'arch': {
                'type': 'PredictModel',
                'args': {
                    'd_model': 128,  # Smaller for testing
                    'n_head': 4,
                    'n_layers': 1,
                    'ffn_hidden': 128,
                    'drop_prob': 0,
                    'tau': 10
                }
            },
            'data_loader': {
                'type': 'TimeseriesDataLoader',
                'args': {
                    'data_dir': 'data/basic/v/data_0.csv',
                    'batch_size': 4,
                    'time_step': 16,
                    'output_window': 15,
                    'feature_dim': 1,
                    'output_dim': 1,
                    'shuffle': True,
                    'validation_split': 0.2,
                    'num_workers': 0
                }
            },
            'optimizer': {
                'type': 'Adam',
                'args': {
                    'lr': 0.001,
                    'weight_decay': 0,
                    'amsgrad': True
                }
            },
            'loss': 'masked_mse_torch',
            'metrics': ['masked_mse_torch'],
            'lr_scheduler': {
                'type': 'StepLR',
                'args': {
                    'step_size': 10,
                    'gamma': 0.1
                }
            },
            'trainer': {
                'epochs': 2,  # Just 2 epochs for testing
                'save_dir': 'saved/',
                'save_freq': 1,
                'verbosity': 0,
                'monitor': 'min val_loss',
                'early_stop': 0,
                'lam': 5e-4,
                'tensorboard': False
            }
        }
        
        # Setup data module
        data_module = CausalFormerDataModule(
            data_dir="data/basic/v/data_0.csv",
            batch_size=4,
            time_step=16,
            output_window=15,
            feature_dim=1,
            output_dim=1,
            validation_split=0.2,
            num_workers=0,
            shuffle=True
        )
        data_module.setup(stage="fit")
        
        # Get series number from data
        series_num = data_module.get_series_num()
        
        # Setup model
        data_module_params = {
            'time_step': 16,
            'output_window': 15,
            'feature_dim': 1,
            'output_dim': 1,
            'series_num': series_num
        }
        
        model = CausalFormerLightningModule(
            config=config,
            data_module_params=data_module_params,
            learning_rate=0.001,
            weight_decay=0.0,
            lam=5e-4,
            lr_scheduler_config=config.get('lr_scheduler')
        )
        
        # Test forward pass
        sample_batch = next(iter(data_module.train_dataloader()))
        data, target = sample_batch
        output = model(data)
        
        print(f"Input shape: {data.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Target shape: {target.shape}")
        
        # Test training step
        loss = model.training_step(sample_batch, 0)
        print(f"Training loss: {loss.item():.6f}")
        
        # Test validation step
        val_loss = model.validation_step(sample_batch, 0)
        print(f"Validation loss: {val_loss.item():.6f}")
        
        # Test optimizer configuration
        optimizer = model.configure_optimizers()
        print("✓ Optimizer configured successfully")
        
        print("✓ Model module test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Model module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_loop():
    """Test a quick training loop"""
    print("\nTesting Training Loop...")
    
    try:
        # Create a simple config
        config = {
            'n_gpu': 0,
            'arch': {
                'type': 'PredictModel',
                'args': {
                    'd_model': 128,
                    'n_head': 4,
                    'n_layers': 1,
                    'ffn_hidden': 128,
                    'drop_prob': 0,
                    'tau': 10
                }
            },
            'data_loader': {
                'type': 'TimeseriesDataLoader',
                'args': {
                    'data_dir': 'data/basic/v/data_0.csv',
                    'batch_size': 4,
                    'time_step': 16,
                    'output_window': 15,
                    'feature_dim': 1,
                    'output_dim': 1,
                    'shuffle': True,
                    'validation_split': 0.2,
                    'num_workers': 0
                }
            },
            'optimizer': {
                'type': 'Adam',
                'args': {
                    'lr': 0.001,
                    'weight_decay': 0,
                    'amsgrad': True
                }
            },
            'loss': 'masked_mse_torch',
            'metrics': ['masked_mse_torch'],
            'trainer': {
                'epochs': 2,
                'lam': 5e-4
            }
        }
        
        # Setup data module
        data_module = CausalFormerDataModule(
            data_dir="data/basic/v/data_0.csv",
            batch_size=4,
            time_step=16,
            output_window=15,
            feature_dim=1,
            output_dim=1,
            validation_split=0.2,
            num_workers=0,
            shuffle=True
        )
        
        # Setup model
        data_module.setup(stage="fit")
        series_num = data_module.get_series_num()
        
        data_module_params = {
            'time_step': 16,
            'output_window': 15,
            'feature_dim': 1,
            'output_dim': 1,
            'series_num': series_num
        }
        
        model = CausalFormerLightningModule(
            config=config,
            data_module_params=data_module_params,
            learning_rate=0.001,
            weight_decay=0.0,
            lam=5e-4
        )
        
        # Setup trainer for quick test
        trainer = pl.Trainer(
            max_epochs=1,  # Just 1 epoch for testing
            accelerator='cpu',
            devices=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False
        )
        
        # Run one epoch
        trainer.fit(model, datamodule=data_module)
        
        print("✓ Training loop test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Training loop test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("Running PyTorch Lightning CausalFormer Tests...")
    print("=" * 50)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_data_module()
    all_passed &= test_model_module()
    all_passed &= test_training_loop()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! The PyTorch Lightning implementation is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
        sys.exit(1)


if __name__ == '__main__':
    main()
