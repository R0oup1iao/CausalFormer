import pytorch_lightning as pl
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import math

from model.model import PredictModel
from model.loss import masked_mse_torch
from model.metric import masked_mse_torch as masked_mse_metric


class CausalFormerLightningModule(pl.LightningModule):
    """
    PyTorch Lightning Module for CausalFormer
    """
    def __init__(
        self,
        config: Dict[str, Any],
        data_module_params: Dict[str, Any],
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        lam: float = 5e-4,
        lr_scheduler_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.config = config
        self.data_module_params = data_module_params
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lam = lam
        self.lr_scheduler_config = lr_scheduler_config
        
        # Extract model parameters
        arch_config = config['arch']['args']
        self.d_model = arch_config['d_model']
        self.n_head = arch_config['n_head']
        self.n_layers = arch_config['n_layers']
        self.ffn_hidden = arch_config['ffn_hidden']
        self.drop_prob = arch_config['drop_prob']
        self.tau = arch_config['tau']
        
        # Extract data parameters
        self.series_num = data_module_params['series_num']
        self.input_window = data_module_params['time_step']
        self.output_window = data_module_params['output_window']
        self.feature_dim = data_module_params['feature_dim']
        self.output_dim = data_module_params['output_dim']
        
        # Update config with series_num for PredictModel
        self.config['data_loader']['args']['series_num'] = self.series_num
        
        # Initialize model
        self.model = PredictModel(
            config=config,
            d_model=self.d_model,
            n_head=self.n_head,
            n_layers=self.n_layers,
            ffn_hidden=self.ffn_hidden,
            drop_prob=self.drop_prob,
            tau=self.tau
        )
        
        # Loss function
        self.criterion = masked_mse_torch
        
        # Metrics
        self.train_metrics = {}
        self.val_metrics = {}
        self.test_metrics = {}
        
        # Initialize metrics
        self._init_metrics()

    def _init_metrics(self):
        """Initialize metrics for training, validation and testing"""
        metric_names = ['loss', 'masked_mse']
        
        for stage in ['train', 'val', 'test']:
            metrics_dict = {}
            for name in metric_names:
                metrics_dict[name] = []
            setattr(self, f'{stage}_metrics', metrics_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model"""
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training step with regularization"""
        data, target = batch
        output = self.model(data)
        
        # Calculate loss with regularization
        prediction_loss = self.criterion(output, target)
        regularization_loss = self.lam * self.model.regularization()
        total_loss = prediction_loss + regularization_loss
        
        # Log metrics
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_prediction_loss', prediction_loss, on_step=True, on_epoch=True)
        self.log('train_regularization_loss', regularization_loss, on_step=True, on_epoch=True)
        
        # Calculate and log masked MSE metric
        masked_mse = masked_mse_metric(output, target)
        self.log('train_masked_mse', masked_mse, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Validation step"""
        data, target = batch
        output = self.model(data)
        
        # Calculate loss without regularization for validation
        val_loss = self.criterion(output, target)
        
        # Log metrics
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Calculate and log masked MSE metric
        masked_mse = masked_mse_metric(output, target)
        self.log('val_masked_mse', masked_mse, on_step=True, on_epoch=True, prog_bar=True)
        
        return val_loss

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Test step"""
        data, target = batch
        output = self.model(data)
        
        # Calculate loss without regularization for testing
        test_loss = self.criterion(output, target)
        
        # Log metrics
        self.log('test_loss', test_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Calculate and log masked MSE metric
        masked_mse = masked_mse_metric(output, target)
        self.log('test_masked_mse', masked_mse, on_step=True, on_epoch=True, prog_bar=True)
        
        return test_loss

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        optimizer_config = self.config['optimizer']['args']
        
        # Create optimizer parameters, avoiding duplicates
        optimizer_params = {
            'lr': self.learning_rate,
            'weight_decay': self.weight_decay
        }
        
        # Add other optimizer arguments, excluding lr and weight_decay to avoid duplicates
        for k, v in optimizer_config.items():
            if k not in ['lr', 'weight_decay']:
                optimizer_params[k] = v
        
        optimizer = torch.optim.Adam(
            self.parameters(),
            **optimizer_params
        )
        
        if self.lr_scheduler_config:
            scheduler_type = self.lr_scheduler_config['type']
            scheduler_args = self.lr_scheduler_config['args']
            
            if scheduler_type == 'StepLR':
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=scheduler_args['step_size'],
                    gamma=scheduler_args['gamma']
                )
            elif scheduler_type == 'ReduceLROnPlateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    patience=scheduler_args.get('patience', 10),
                    factor=scheduler_args.get('gamma', 0.1),
                    verbose=True
                )
            else:
                # Default to StepLR if scheduler type is not recognized
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss' if self.lr_scheduler_config['type'] == 'ReduceLROnPlateau' else None,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        
        return optimizer

    def on_train_epoch_end(self):
        """Log additional information at the end of training epoch"""
        # Log learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_epoch=True, prog_bar=True)

    def get_model(self) -> PredictModel:
        """Get the underlying PredictModel instance"""
        return self.model

    def predict_step(self, batch: tuple, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """Prediction step for inference"""
        data, _ = batch
        return self.model(data)

    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        """Extract attention weights for interpretability"""
        attention_weights = {}
        
        # Extract weights from causal convolution
        if hasattr(self.model.encoder.layers[0].attention.Wv, 'get_wgt'):
            causal_conv_weights = self.model.encoder.layers[0].attention.Wv.get_wgt()
            if causal_conv_weights is not None:
                attention_weights['causal_conv'] = causal_conv_weights
        
        # Extract weights from multi-variate causal attention
        if hasattr(self.model.encoder.layers[0].attention.attention, 'get_wgt'):
            attention_weights_attn = self.model.encoder.layers[0].attention.attention.get_wgt()
            if attention_weights_attn is not None:
                attention_weights['multi_variate_attention'] = attention_weights_attn
        
        return attention_weights
