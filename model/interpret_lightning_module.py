import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from sklearn.cluster import KMeans

from explainer.explainer import RRP
from evaluator.evaluator import evaluate, getextendeddelays, evaluatedelay
from model.model import PredictModel


class CausalInterpretLightningModule(pl.LightningModule):
    """
    PyTorch Lightning Module for Causal Discovery in CausalFormer
    This module performs causal discovery using RRP analysis on a trained prediction model.
    """
    
    def __init__(
        self,
        trained_model: PredictModel,
        config: Dict[str, Any],
        ground_truth: Optional[str] = None,
        data_module_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the causal interpretation module.
        
        Args:
            trained_model: Pre-trained PredictModel instance
            config: Configuration dictionary
            ground_truth: Path to ground truth file for evaluation
            data_module_params: Parameters from data module (time_step, series_num, etc.)
        """
        super().__init__()
        
        self.trained_model = trained_model
        self.config = config
        self.ground_truth = ground_truth
        self.data_module_params = data_module_params or {}
        
        # Extract parameters
        self.time_step = self.data_module_params.get('time_step', config['data_loader']['args']['time_step'])
        self.series_num = self.data_module_params.get('series_num', config['data_loader']['args']['series_num'])
        
        # Extract explainer parameters
        explainer_config = config.get('explainer', {})
        self.m = explainer_config.get('m', 3)
        self.n = explainer_config.get('n', 10)
        
        assert self.m < self.n, "the number of selected top m clusters must be smaller than the total number of n clusters"
        
        # Initialize RRP attribution generator
        self.attribution_generator = RRP(self.trained_model)
        
        # Storage for causal results
        self.causal_results = []
        self.all_causes = {}
        self.all_delays = {}
        self.columns = []
        
        # Initialize metrics storage
        self.test_metrics = {
            'precision_prime': 0.0,
            'recall_prime': 0.0,
            'f1_prime': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'pod': 0.0,
            'fp': 0,
            'tp': 0,
            'fp_direct': 0,
            'tp_direct': 0,
            'fn': 0
        }

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        """
        Perform causal discovery on a batch of data.
        This method collects data for causal analysis but defers actual analysis
        to on_test_epoch_end to have access to all data.
        """
        data, _ = batch
        
        # Store data for later analysis
        if not hasattr(self, 'test_data'):
            self.test_data = []
            self.test_labels = []
        
        self.test_data.append(data.cpu().numpy())
        
        # Store column names if not already stored
        if not self.columns and hasattr(self.trainer.datamodule, 'df_data'):
            self.columns = list(self.trainer.datamodule.df_data.columns)

    def on_test_epoch_end(self) -> None:
        """
        Perform causal discovery analysis after all test data has been collected.
        """
        if not hasattr(self, 'test_data') or len(self.test_data) == 0:
            self.logger.warning("No test data available for causal discovery")
            return
        
        # Combine all test data
        all_data = np.concatenate(self.test_data, axis=0)
        
        # Convert to tensor and move to device, and require gradients
        device = next(self.trained_model.parameters()).device
        data_tensor = torch.tensor(all_data, dtype=torch.float).to(device)
        data_tensor.requires_grad_(True)  # Enable gradients for RRP
        
        # Perform causal discovery
        self._perform_causal_discovery(data_tensor)
        
        # Evaluate results if ground truth is available
        if self.ground_truth:
            self._evaluate_causal_results()
        
        # Log all metrics
        self._log_causal_metrics()

    def run_causal_discovery(self, data_loader) -> None:
        """
        Run causal discovery directly on a data loader.
        This method provides an alternative to the Lightning test workflow.
        """
        # Collect all data from the data loader - use the same approach as original
        test_data = []
        test_labels = []
        for batch in data_loader:
            data, labels = batch
            test_data.append(data.cpu().numpy())
            test_labels.append(labels.cpu().numpy())
        
        if not test_data:
            print("No data available for causal discovery")
            return
        
        # Combine all test data - use mean aggregation like original version for better results
        all_data = np.concatenate(test_data, axis=0)
        all_labels = np.concatenate(test_labels, axis=0)
        
        print(f"Data shape: {all_data.shape}")
        print(f"Labels shape: {all_labels.shape}")
        
        # Use mean aggregation like original version for better causal discovery
        # This helps reduce noise and get more stable causal relationships
        if len(all_data) > 1:
            print("Using mean aggregation for causal discovery (like original version)")
            data_mean = np.mean(all_data, axis=0, keepdims=True)
            labels_mean = np.mean(all_labels, axis=0, keepdims=True)
        else:
            data_mean = all_data
            labels_mean = all_labels
        
        # Convert to tensor and move to device, and require gradients
        device = next(self.trained_model.parameters()).device
        data_tensor = torch.tensor(data_mean, dtype=torch.float).to(device)
        data_tensor.requires_grad_(True)  # Enable gradients for RRP
        
        # Perform causal discovery
        self._perform_causal_discovery(data_tensor)
        
        # Evaluate results if ground truth is available
        if self.ground_truth:
            self._evaluate_causal_results()
        
        # Log all metrics
        self._log_causal_metrics()

    def _perform_causal_discovery(self, data: torch.Tensor) -> None:
        """
        Perform causal discovery using RRP analysis.
        
        Args:
            data: Input data tensor of shape (batch_size, time_step, series_num)
        """
        batch_size = data.shape[0]
        relA = []
        relK = []
        
        # Interpret each time series
        for interpreted_series in range(self.series_num):
            rel_a, rel_k = self.attribution_generator.generate_RRP(
                batch_size, data, interpreted_series
            )
            relA.append(rel_a.detach().cpu().numpy()[interpreted_series])
            
            # Process causal convolution relevance
            relk_align = rel_k.detach().cpu().numpy()[:, interpreted_series, -1, :].copy()
            # The relK[i][i][-1] is zero vector due to the time_step th data cannot be used 
            # to predict the time_step th future itself.
            relk_align[interpreted_series, :] = rel_k.detach().cpu().numpy()[
                interpreted_series, interpreted_series, -2, :
            ]
            relK.append(relk_align)
        
        # Analyze causal relationships
        self.causal_results = self._analyze_causal_relationships(relA, relK)
        
        # Store causes and delays for evaluation
        self.all_causes = {i: [] for i in range(self.series_num)}
        self.all_delays = {}
        
        for causal in self.causal_results:
            self.all_causes[causal[1]].append(causal[0])
            self.all_delays[(causal[1], causal[0])] = causal[2]

    def _analyze_causal_relationships(
        self, relA: List[np.ndarray], relK: List[np.ndarray]
    ) -> List[Tuple[int, int, int]]:
        """
        Analyze causal relationships using clustering.
        
        Args:
            relA: List of relevance scores of attention matrix for each time series
            relK: List of relevance scores of causal convolution kernels for each time series
            
        Returns:
            List of tuples representing causal graph edges (cause, effect, lag)
        """
        estimator = KMeans(n_clusters=self.n)
        ans = []
        
        # Find causes of series i
        for i, relAi in enumerate(relA):
            if relAi.sum() == 0.0:  # all the weights to series i are zero
                continue
            
            data = np.array(relAi)
            estimator.fit(data.reshape(-1, 1))
            cluster_labels = estimator.labels_
            cluster_centers = estimator.cluster_centers_.reshape(-1)
            
            largest_m_clusters = np.argsort(cluster_centers)[-self.m:]
            
            for j in range(len(relAi)):
                if cluster_labels[j] in largest_m_clusters:
                    relKij = relK[i][j]
                    indices = np.argsort(-1 * relKij)
                    ans.append((j, i, self.time_step - 1 - indices[0]))
        
        return ans

    def _evaluate_causal_results(self) -> None:
        """
        Evaluate causal discovery results against ground truth.
        """
        if not self.ground_truth:
            print("No ground truth provided for evaluation")
            return
        
        # Import here to avoid circular imports
        from evaluator.evaluator import evaluate, getextendeddelays, evaluatedelay
        
        # Create a simple logger for evaluation
        class SimpleLogger:
            def info(self, msg):
                print(msg)
        
        logger = SimpleLogger()
        
        # Get column names - use default names if not available
        if not self.columns:
            print("Warning: No column names available, using default names")
            self.columns = [f"Series_{i}" for i in range(self.series_num)]
        
        # Evaluate causal relationships
        print(f"Columns: {self.columns}")
        print(f"All causes: {self.all_causes}")
        print(f"Number of discovered relationships: {len(self.causal_results)}")
        
        # Call evaluate function and capture all return values
        evaluation_results = evaluate(
            logger, self.ground_truth, self.all_causes, self.columns
        )
        
        # Unpack results - note the correct variable names from evaluator.py
        fp, tp, fp_direct, tp_direct, fn, fps, fps_direct, tps, tps_direct, fns, f1_prime, f1 = evaluation_results
        
        print(f"Evaluation results - TP: {tp}, FP: {fp}, FN: {fn}, TP_direct: {tp_direct}, FP_direct: {fp_direct}")
        print(f"F1': {f1_prime}, F1: {f1}")
        
        # Evaluate delay discovery
        extended_delays, read_gt, extended_read_gt = getextendeddelays(
            self.ground_truth, self.columns
        )
        pod = evaluatedelay(extended_delays, self.all_delays, tps, 1) * 100
        
        # Calculate precision and recall (for consistency, but F1 scores are already computed)
        precision_prime = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall_prime = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        
        precision = tp_direct / float(tp_direct + fp_direct) if (tp_direct + fp_direct) > 0 else 0.0
        recall = tp_direct / float(tp_direct + fn) if (tp_direct + fn) > 0 else 0.0
        
        # Store metrics - use the F1 scores computed by the evaluator
        self.test_metrics.update({
            'precision_prime': precision_prime,
            'recall_prime': recall_prime,
            'f1_prime': f1_prime,  # Use F1' from evaluator
            'precision': precision,
            'recall': recall,
            'f1': f1,  # Use F1 from evaluator
            'pod': pod,
            'fp': fp,
            'tp': tp,
            'fp_direct': fp_direct,
            'tp_direct': tp_direct,
            'fn': fn
        })

    def _log_causal_metrics(self) -> None:
        """
        Log all causal discovery metrics.
        """
        # Log primary metrics
        self.log('test_precision_prime', self.test_metrics['precision_prime'], on_epoch=True)
        self.log('test_recall_prime', self.test_metrics['recall_prime'], on_epoch=True)
        self.log('test_f1_prime', self.test_metrics['f1_prime'], on_epoch=True)
        self.log('test_precision', self.test_metrics['precision'], on_epoch=True)
        self.log('test_recall', self.test_metrics['recall'], on_epoch=True)
        self.log('test_f1', self.test_metrics['f1'], on_epoch=True)
        self.log('test_pod', self.test_metrics['pod'], on_epoch=True)
        
        # Log detailed statistics
        self.log('test_fp', self.test_metrics['fp'], on_epoch=True)
        self.log('test_tp', self.test_metrics['tp'], on_epoch=True)
        self.log('test_fp_direct', self.test_metrics['fp_direct'], on_epoch=True)
        self.log('test_tp_direct', self.test_metrics['tp_direct'], on_epoch=True)
        self.log('test_fn', self.test_metrics['fn'], on_epoch=True)
        
        # Print results for visibility
        print("\n" + "="*50)
        print("Causal Discovery Results")
        print("="*50)
        print(f"Precision': {self.test_metrics['precision_prime']:.4f}")
        print(f"Recall': {self.test_metrics['recall_prime']:.4f}")
        print(f"F1': {self.test_metrics['f1_prime']:.4f}")
        print(f"Precision: {self.test_metrics['precision']:.4f}")
        print(f"Recall: {self.test_metrics['recall']:.4f}")
        print(f"F1: {self.test_metrics['f1']:.4f}")
        print(f"PoD: {self.test_metrics['pod']:.2f}%")
        print("="*50)
        
        # Print discovered causal relationships
        if self.causal_results:
            print("\nDiscovered Causal Relationships:")
            for cause, effect, delay in self.causal_results:
                cause_name = self.columns[cause] if self.columns else f"Series_{cause}"
                effect_name = self.columns[effect] if self.columns else f"Series_{effect}"
                print(f"  {cause_name} -> {effect_name} (delay: {delay})")

    def get_causal_results(self) -> List[Tuple[int, int, int]]:
        """
        Get the discovered causal relationships.
        
        Returns:
            List of (cause, effect, delay) tuples
        """
        return self.causal_results

    def get_evaluation_metrics(self) -> Dict[str, Any]:
        """
        Get the evaluation metrics.
        
        Returns:
            Dictionary containing all evaluation metrics
        """
        return self.test_metrics

    def configure_optimizers(self):
        """
        This module doesn't require optimization as it's for inference only.
        """
        return None
