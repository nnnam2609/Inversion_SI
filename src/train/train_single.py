import sys
import os
import torch
import numpy as np
from src.train.train import Train
#from src.model.baseline_8 import BaselineModel
from src.model.baseline_5 import BaselineModel
#from src.model.cnn import BaselineModel
#from src.model.gru import BaselineModel
#from src.model.tcn import BaselineModel
#from src.model.model_lstm import BaselineModel
import utils.metrics as metrics
from src.utils.temporal_loss import contour_velocity_loss
from tqdm import tqdm
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader
from utils.evaluation import test_model
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.nn.functional as F


class TrainSingle(Train):
    """
    A class for training a single model (BaselineModel) in a supervised learning setup.

    This class inherits from the base `Train` class and is specifically designed for training a single model 
    using the baseline architecture. It includes methods for initializing the model, processing batches of data, 
    testing the model, and training the model on each batch. The model utilizes various loss functions and metrics
    for performance evaluation.

    Args:
        gpu_id (int): The ID of the GPU to use for training.
        config (dict): The configuration dictionary containing model parameters and dataset information.
        ddp_model (nn.Module): The model to be trained, wrapped for distributed data parallel training.
        best_model (object): The best model, used for evaluation during testing.

    Methods:
        - initialize_model: Initializes the baseline model with specified parameters.
        - process_batch: Processes a batch of training data for training.
        - test_batch: Tests a batch of data using the best trained model.
        - train_batch: Performs one training step, including forward pass, loss calculation, and backpropagation.
    """
    
    def initialize_model(self):
        """
        Initializes the baseline model with parameters from the configuration.

        This method creates an instance of `BaselineModel` using the model's configuration parameters such as 
        input layer size, hidden layers, output layer size, number of phonemes, and other necessary configurations 
        related to the phoneme set and dataset.

        The model is then transferred to the appropriate GPU device based on `self.gpu_id`.

        Args:
            None

        Returns:
            None
        """
        self.model = BaselineModel(
            self.config['input_layer'],
            self.config['hidden_layer'],
            self.config['num_layers'],
            self.config['output_layer'],
            len(self.config['classes']),
            self.config['nbr_phonemes'],
            self.config['phonemesdir'],
            #False  # Phonemes set to None
        ).to(self.gpu_id)
        
    def process_batch(self, batch: dict):
        """
        Processes a batch of data for training.

        This method extracts features, labels, sequence lengths, and normalization parameters from the input batch.
        It then forwards these to the `train_batch` method to compute the necessary gradients and loss.

        Args:
            batch (dict): A dictionary containing:
                - 'features': The input features for the model.
                - 'labels': The target labels.
                - 'sequences_length': The length of each sequence.
                - 'std' and 'mean': The standard deviation and mean for normalization.

        Returns:
            tuple: A tuple of loss values including:
                - Loss MSE
                - Loss MSE mean unnormalized
                - Loss MSE mean
                - Loss Pearson correlation
                - Loss criterion
                - Loss phonemes (currently set to 0)
        """
        x_batch = batch['features']
        y_batch = batch['labels']
        length_sequences = batch['sequences_length']
        std_batch = batch['std']
        mean_batch = batch['mean']
        return self.train_batch(x_batch, y_batch, length_sequences, std_batch, mean_batch)
    
    def train_batch(self, x_batch: torch.Tensor, y_batch: torch.Tensor, length_sequences: torch.Tensor,
                    std: torch.Tensor, mean: torch.Tensor) -> tuple:
        """
        Performs one training step for a batch of data.

        This method executes the forward pass of the model, computes the loss using various criteria (MSE, Pearson correlation, etc.),
        and returns the computed loss values. It also performs normalization and backpropagation.

        Args:
            x_batch (torch.Tensor): The input features for the model.
            y_batch (torch.Tensor): The target labels.
            length_sequences (torch.Tensor): The length of each sequence.
            std (torch.Tensor): The standard deviation for normalization.
            mean (torch.Tensor): The mean for normalization.

        Returns:
            tuple: A tuple containing multiple loss values:
                - loss_mse (float): Mean squared error loss.
                - loss_mse_mean_unormalized (float): MSE loss for unnormalized data.
                - loss_mse_mean (float): MSE mean loss.
                - loss_pearson (float): Pearson correlation loss.
                - loss_criterion (float): A combined loss from a custom criterion function.
                - loss_phonemes (float): Loss related to phoneme predictions (currently 0).
        """
        self.ddp_model.train()
        # Transfer data to the specified rank
        x_batch, y_batch, length_sequences, std, mean = x_batch.to(self.gpu_id), y_batch.to(self.gpu_id), length_sequences.to(self.gpu_id), std.to(self.gpu_id), mean.to(self.gpu_id)
 
        # Forward pass
        self.ddp_model = self.ddp_model
        y_pred, _, _ = self.ddp_model(x_batch, length_sequences)
        y_batch = y_batch
        y_pred = y_pred
        std = std
        mean = mean

        #dim = y_batch.size(1) - y_pred.size(1)
        y_batch = y_batch[:, :y_pred.size(1)]

        loss_mse_mean = metrics.loss_rmse(y_batch, y_pred)
        loss_mse = metrics.loss_mse(y_batch, y_pred)
        velocity_weight = float(self.config.get("contour_velocity_loss_weight", 0.0))
        if velocity_weight:
            velocity_reduction = str(self.config.get("contour_velocity_loss_reduction", "mean"))
            loss_mse = loss_mse + velocity_weight * contour_velocity_loss(
                y_batch,
                y_pred,
                length_sequences,
                reduction=velocity_reduction,
            )

        y_batch_unormalized = (y_batch * std) + mean
        y_pred_unormalized = (y_pred * std) + mean
        
        loss_mse_mean_unormalized = metrics.loss_rmse(y_batch_unormalized, y_pred_unormalized)
        loss_pearson = metrics.pearson_correlation(y_batch, y_pred)
        alpha = 90
        loss_criterion = metrics.criterion_both(y_batch, y_pred,alpha, True, self.gpu_id)

        loss_phonemes = 0
        return loss_mse, loss_mse_mean_unormalized.item(), loss_mse_mean.item(), loss_pearson.item(), loss_criterion.item(), loss_phonemes
    
    
    def test_batch(self, batch: dict, folder_run: str):
        """
        Tests a batch of data using the best trained model.

        This method extracts features, labels, sequence lengths, phonemes, and other necessary data from the input batch
        and uses the best model to perform evaluation.

        Args:
            batch (dict): A dictionary containing:
                - 'features': The input features for the model.
                - 'labels': The target labels.
                - 'frames': The frames associated with the input data.
                - 'phonemes': Phoneme data.
                - 'sequences_length': The length of each sequence.
                - 'std' and 'mean': The standard deviation and mean for normalization.

        Returns:
            The evaluation result from the `best_model.test` method, which includes performance metrics.
        """
        x_test = batch['features']
        y_test = batch['labels']
        f_test = batch['frames']
        z_test = batch['phonemes']
        std_test = batch['std']
        mean_test = batch['mean']
        length_sequences = batch['sequences_length']
        datadir = self.config['datadir']
        return test_model(
            self.model,
            x_test,
            y_test,
            length_sequences,
            self.config['sequence_length'],
            self.gpu_id,
            std_test,
            mean_test,
            f_test,
            z_test,
            folder_run,
            datadir,
            skip_tract_variables=self.config.get("skip_tract_variables", False),
        )

    
    
