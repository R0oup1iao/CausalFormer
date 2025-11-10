import pytorch_lightning as pl
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn import preprocessing
from typing import Optional, Tuple, Dict, Any
import numpy as np


class TimeseriesDataset(Dataset):
    """
    Timeseries dataset for temporal causal discovery
    """
    def __init__(self, data_dir: str, time_step: int, output_window: int, 
                 feature_dim: int = 1, output_dim: int = 1, training: bool = True):
        self.data_dir = data_dir
        self.df_data = pd.read_csv(self.data_dir)
        self.data_len = len(self.df_data.index)
        self.data = self.df_data.values.astype('float32')

        self.time_step = time_step
        self.output_window = output_window
        self.series_num = self.data.shape[1]
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.training = training
        
        # Normalize data for regression relevance propagation
        scaler = preprocessing.MinMaxScaler(feature_range=(0.5, 1))
        self.data = scaler.fit_transform(self.data)
        
        # Construct input samples
        self.dataset = []
        assert self.time_step < len(self.data) + 1, "Input window length must be shorter than whole data"
        assert self.output_window < self.time_step, (
            "Output window length must be shorter than input window. "
            "Practically, we ignore the prediction of the first time slot for "
            "the sake of fairness, because the observations of each time series "
            "do not contribute to their own predictions in the first time slot "
            "due to the right shifting of self-convolution result, which is "
            "different from other time slots."
        )
        
        for i in range(self.time_step, len(self.data) + 1):
            input_data = self.data[i - self.time_step:i].reshape(
                self.time_step, self.series_num, self.feature_dim
            )
            target_data = self.data[i - self.output_window:i].reshape(
                self.output_window, self.series_num, self.output_dim
            )
            self.dataset.append((input_data, target_data))
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_data, target_data = self.dataset[idx]
        return torch.from_numpy(input_data), torch.from_numpy(target_data)


class CausalFormerDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for CausalFormer
    """
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        time_step: int = 32,
        output_window: int = 31,
        feature_dim: int = 1,
        output_dim: int = 1,
        validation_split: float = 0.1,
        num_workers: int = 4,
        shuffle: bool = True
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.time_step = time_step
        self.output_window = output_window
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.validation_split = validation_split
        self.num_workers = num_workers
        self.shuffle = shuffle
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Store dataset parameters for later use
        self.dataset_params = {
            'time_step': time_step,
            'output_window': output_window,
            'feature_dim': feature_dim,
            'output_dim': output_dim
        }

    def setup(self, stage: Optional[str] = None):
        """Load data and split into train/val/test sets"""
        if stage == "fit" or stage is None:
            full_dataset = TimeseriesDataset(
                data_dir=self.data_dir,
                time_step=self.time_step,
                output_window=self.output_window,
                feature_dim=self.feature_dim,
                output_dim=self.output_dim,
                training=True
            )
            
            # Split dataset
            dataset_size = len(full_dataset)
            val_size = int(self.validation_split * dataset_size)
            train_size = dataset_size - val_size
            
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )
            
        if stage == "test" or stage is None:
            self.test_dataset = TimeseriesDataset(
                data_dir=self.data_dir,
                time_step=self.time_step,
                output_window=self.output_window,
                feature_dim=self.feature_dim,
                output_dim=self.output_dim,
                training=False
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def get_series_num(self) -> int:
        """Get the number of time series in the dataset"""
        if self.train_dataset is not None:
            sample_input, _ = self.train_dataset[0]
            return sample_input.shape[1]  # series_num dimension
        else:
            # Fallback: load a small sample to get series_num
            temp_dataset = TimeseriesDataset(
                data_dir=self.data_dir,
                time_step=self.time_step,
                output_window=self.output_window,
                feature_dim=self.feature_dim,
                output_dim=self.output_dim
            )
            sample_input, _ = temp_dataset[0]
            return sample_input.shape[1]
