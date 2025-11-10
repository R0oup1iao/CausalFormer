#!/usr/bin/env python3
"""
Test script for Causal Discovery in PyTorch Lightning CausalFormer
"""

import os
import sys
import torch
import pytorch_lightning as pl
import tempfile
import shutil

# Add current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader.lightning_data_module import CausalFormerDataModule
from model.lightning_module import CausalFormerLightningModule
from model.interpret_lightning_module import CausalInterpretLightningModule


def test_causal_interpret_module():
    """Test the causal interpretation module with a simple configuration"""
    print("Testing Causal Interpretation Module...")
    
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
            'explainer': {
                'm': 1,
                'n': 3
            },
            'trainer': {
                'epochs': 1,
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
            shuffle=False  # Don't shuffle for causal discovery
        )
        data_module.setup(stage="fit")
        
        # Get series number from data
        series_num = data_module.get_series_num()
        
        # Setup prediction model
        data_module_params = {
            'time_step': 16,
            'output_window': 15,
            'feature_dim': 1,
            'output_dim': 1,
            'series_num': series_num
        }
        
        prediction_model = CausalFormerLightningModule(
            config=config,
            data_module_params=data_module_params,
            learning_rate=0.001,
            weight_decay=0.0,
            lam=5e-4
        )
        
        # Get the underlying PredictModel
        trained_model = prediction_model.get_model()
        
        # Setup causal interpretation model
        interpret_model = CausalInterpretLightningModule(
            trained_model=trained_model,
            config=config,
            ground_truth=None,  # No ground truth for basic test
            data_module_params=data_module_params
        )
        
        print("✓ Causal interpretation model created successfully")
        
        # Test with a sample batch
        sample_batch = next(iter(data_module.train_dataloader()))
        data, target = sample_batch
        
        # Test that the model can process data
        with torch.no_grad():
            output = trained_model(data)
            print(f"Prediction model output shape: {output.shape}")
        
        print("✓ Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Causal interpretation module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_causal_discovery_workflow():
    """Test the complete causal discovery workflow"""
    print("\nTesting Causal Discovery Workflow...")
    
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
                    'shuffle': False,  # Important for consistent causal discovery
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
            'explainer': {
                'm': 1,
                'n': 3
            },
            'trainer': {
                'epochs': 1,
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
            shuffle=False
        )
        data_module.setup(stage="test")  # Use test data for causal discovery
        
        # Get series number from data
        series_num = data_module.get_series_num()
        
        # Setup prediction model
        data_module_params = {
            'time_step': 16,
            'output_window': 15,
            'feature_dim': 1,
            'output_dim': 1,
            'series_num': series_num
        }
        
        prediction_model = CausalFormerLightningModule(
            config=config,
            data_module_params=data_module_params,
            learning_rate=0.001,
            weight_decay=0.0,
            lam=5e-4
        )
        
        # Get the underlying PredictModel
        trained_model = prediction_model.get_model()
        
        # Setup causal interpretation model
        interpret_model = CausalInterpretLightningModule(
            trained_model=trained_model,
            config=config,
            ground_truth=None,  # No ground truth for workflow test
            data_module_params=data_module_params
        )
        
        # Ensure model is in eval mode but with gradients enabled
        trained_model.eval()
        for param in trained_model.parameters():
            param.requires_grad = True
        
        # Use the direct method instead of Lightning test workflow
        test_loader = data_module.test_dataloader()
        interpret_model.run_causal_discovery(test_loader)
        
        # Check that causal results were generated
        causal_results = interpret_model.get_causal_results()
        evaluation_metrics = interpret_model.get_evaluation_metrics()
        
        print(f"✓ Discovered {len(causal_results)} causal relationships")
        print(f"✓ Evaluation metrics: {evaluation_metrics}")
        
        # Verify that metrics are logged
        assert 'precision_prime' in evaluation_metrics
        assert 'recall_prime' in evaluation_metrics
        assert 'f1_prime' in evaluation_metrics
        assert 'precision' in evaluation_metrics
        assert 'recall' in evaluation_metrics
        assert 'f1' in evaluation_metrics
        assert 'pod' in evaluation_metrics
        
        print("✓ Causal discovery workflow test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Causal discovery workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_compatibility():
    """Test compatibility between prediction and interpretation models"""
    print("\nTesting Model Compatibility...")
    
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
                    'shuffle': False,
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
            'explainer': {
                'm': 3,
                'n': 10
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
            shuffle=False
        )
        data_module.setup(stage="fit")
        
        # Get series number from data
        series_num = data_module.get_series_num()
        
        # Setup prediction model
        data_module_params = {
            'time_step': 16,
            'output_window': 15,
            'feature_dim': 1,
            'output_dim': 1,
            'series_num': series_num
        }
        
        prediction_model = CausalFormerLightningModule(
            config=config,
            data_module_params=data_module_params,
            learning_rate=0.001,
            weight_decay=0.0,
            lam=5e-4
        )
        
        # Get the underlying PredictModel
        trained_model = prediction_model.get_model()
        
        # Verify that the trained model is compatible with RRP
        from explainer.explainer import RRP
        attribution_generator = RRP(trained_model)
        
        # Test that RRP can be initialized
        assert attribution_generator is not None
        print("✓ RRP attribution generator initialized successfully")
        
        # Test that the model has the required methods for causal discovery
        assert hasattr(trained_model, 'encoder')
        assert hasattr(trained_model.encoder, 'layers')
        assert len(trained_model.encoder.layers) > 0
        
        print("✓ Model compatibility test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Model compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all causal discovery tests"""
    print("Running PyTorch Lightning CausalFormer Causal Discovery Tests...")
    print("=" * 60)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_causal_interpret_module()
    all_passed &= test_causal_discovery_workflow()
    all_passed &= test_model_compatibility()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All causal discovery tests passed! The implementation is working correctly.")
        print("\nUsage:")
        print("1. Train a model: python train_lightning.py -c config/config_lorenz.json")
        print("2. Run causal discovery: python interpret_lightning.py -m <model_path> -g <ground_truth>")
    else:
        print("❌ Some causal discovery tests failed. Please check the implementation.")
        sys.exit(1)


if __name__ == '__main__':
    main()
