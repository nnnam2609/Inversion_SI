import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure

def loss_rmse(y, y_pred):
    mse = torch.nn.MSELoss(reduction='mean')(y, y_pred)
    rmse = torch.sqrt(mse)
    return rmse

def loss_rmse_np(y, y_pred):
    mse = np.mean((y - y_pred) ** 2)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    return rmse

def loss_mse(y, y_pred):
    mse = torch.nn.MSELoss(reduction='sum')(y, y_pred)
    return mse

def loss_phoneme(y, y_pred):
    criterion = torch.nn.CrossEntropyLoss()
    loss_phonemes = criterion(y_pred.view(-1), y.view(-1))
    return loss_phonemes

def loss_mse_phoneme(y, y_pred, z, z_pred):
    # mse = loss_mse(y, y_pred)
    mse = F.mse_loss(y, y_pred)
    phoneme_loss = loss_phoneme(z, z_pred)
    total_loss = mse + phoneme_loss
    return total_loss
def loss_mri(real_irm, pred_irm):
    data_min = torch.min(torch.min(real_irm), torch.min(pred_irm))
    data_max = torch.max(torch.max(real_irm), torch.max(pred_irm))
    data_range = data_max - data_min
    loss_ssim = 1.0 - structural_similarity_index_measure(pred_irm, real_irm, data_range=data_range)
    return loss_ssim

def loss_mse_mri(y, y_pred, i, i_pred):
    mse = loss_mse(y, y_pred)
    ssim = loss_mri(i, i_pred)
    a = 1.0
    b = 1000
    return a * mse + b * ssim
    
def loss_mse_variance(y, y_pred):
    #return torch.nn.MSELoss(reduction='sum')(y, y_pred).double()
    # # Compute the derivative of each articulator
    # # We use finite differences to approximate the derivative
    # derivatives = torch.diff(y, dim=1)  # Shape: [batch_size, sequence_length-1, articulators, coordinates]

    # # Pad the derivatives to match the original sequence length
    # derivatives = torch.cat([derivatives, torch.zeros_like(derivatives[:, :1, :, :])], dim=1)

    # # Compute the MSE loss
    # mse_loss = nn.MSELoss(reduction='none')(y, y_pred)  # Shape: [batch_size, sequence_length, articulators, coordinates]

    # # Weight the MSE loss by the derivatives
    # weighted_mse_loss = mse_loss * derivatives.abs()  # Use absolute value of derivatives for weighting

    # # Sum the weighted MSE loss
    # total_loss = weighted_mse_loss.sum()

    # # Convert to double if needed
    # total_loss = total_loss.double()

    # return total_loss
    
    num_articulators = y.size(2)  # Number of articulators (8)
    # Step 1: Compute Variance for Each Articulator
    articulator_variance = y.var(dim=(0, 1, 3))  # Variance over batch, sequence, and coordinates
    
    #[Batch, sequence, articulators, coordinates]
    # # Step 2: Normalize the Variance to Create Weights
    # articulator_weights = articulator_variance / articulator_variance.sum()
    # # Step 3: Reshape to Enable Broadcasting
    # articulator_weights = articulator_weights.view(1, 1, num_articulators, 1)  # Shape: [1, 1, 8, 1]
    
    # Step 2: Standardize (Z-score normalization)
    mean_variance = articulator_variance.mean()
    std_variance = articulator_variance.std()
    standardized_weights = (articulator_variance - mean_variance) / (std_variance + 1e-8)  # Avoid division by zero
    # Step 3: Shift and scale to keep weights positive
    standardized_weights = (standardized_weights - standardized_weights.min()) / (
        standardized_weights.max() - standardized_weights.min() + 1e-8
    )  # Normalize between 0 and 1

    # Step 4: Reshape for broadcasting
    articulator_weights = standardized_weights.view(1, 1, num_articulators, 1)
    
    
    
    
    
    # Step 4: Compute the Weighted MSE Loss
    loss = torch.nn.MSELoss(reduction='none')(y, y_pred).double()  # Compute per-element MSE
    weighted_loss = loss * articulator_weights  # Apply weights to each articulator
    final_loss = weighted_loss.sum()  # Sum over all dimensions
    return final_loss
    

def weighted_mse_loss(y, y_pred):
    # Define weights: 1 for the first 50 coordinates, 2 for the last 50
    weights = torch.ones_like(y)
    weights[..., 50:] = 1 # Apply a higher weight to the last 50 coordinates

    # Calculate squared differences
    squared_diff = (y - y_pred) ** 2

    # Apply weights to the squared differences and sum them
    weighted_mse = (squared_diff * weights).sum()

    return weighted_mse.double()

def pearson_correlation (y, y_pred):
    
    epsilon = torch.tensor(0.000001,dtype=torch.float64)
    y_cov = y.sub(torch.mean(y, dim=1, keepdim=True))
    y_pred_cov = y_pred.sub(torch.mean(y_pred,dim=1, keepdim=True))
    
  
    
    numerator = torch.sum(y_cov * y_pred_cov, dim=1, keepdim=True)
    denominator = torch.sqrt(torch.sum(y_cov ** 2, dim=1, keepdim=True)) * \
                  torch.sqrt(torch.sum(y_pred_cov ** 2, dim=1, keepdim=True))
    
    
    numerator = numerator + epsilon
    denominator = denominator + epsilon

    pearson = torch.div(numerator, denominator)


    mean_coordinates = torch.mean(pearson,dim=1, keepdim=True)
    pearson = torch.mean(mean_coordinates)
    return pearson

def loss_function(y, y_pred):
    alpha = 1000
    pearson = pearson_correlation(y, y_pred)
    loss = loss_rmse(y, y_pred)
    loss_final = loss - (alpha*pearson)
    #loss_function = rmse(y, y_pred) - pearson_correlation (y, y_pred) * beta
    return loss_final

def criterion_both(my_y,my_ypred,alpha,cuda_avail,nb_gpu):
    compl = torch.tensor(1. - float(alpha) / 100., dtype=torch.float64)
    alpha = torch.tensor(float(alpha) / 100., dtype = torch.float64)
    multip = torch.tensor(float(1000), dtype = torch.float64)
    if cuda_avail:
        alpha = alpha
        multip = multip
        compl = compl
    a = alpha * criterion_pearson(my_y, my_ypred, cuda_avail, nb_gpu)*multip
    b = compl * torch.nn.MSELoss(reduction='mean')(my_y, my_ypred)
    new_loss = a + b
    return new_loss

def criterion_pearson(y, y_pred, cuda_avail , nb_gpu):
    """
    :param y: nparray (B,K,18) target trajectories of the batch (size B) , padded (K = maxlenght)
    :param y_pred: nparray (B,K,18) predicted trajectories of the batch (size B), padded (K = maxlenght
    :param cuda_avail: bool whether gpu is available
    :param device: the device
    :return: loss function for this prediction for loss = pearson correlation
    for each pair of trajectories (target & predicted) we calculate the pearson correlation between the two
    we sum all the pearson correlation to obtain the loss function
    // Idea : integrate the range of the traj here, making the loss for each sentence as the weighted average of the
    losses with weight proportional to the range of the traj (?)
    """
    y_1 = y.sub(torch.mean(y, dim=1, keepdim=True))
    y_pred_1 = y_pred.sub(torch.mean(y_pred,dim=1, keepdim=True))
    nume = torch.sum(y_1 * y_pred_1, dim=1, keepdim=True)  # (B,1,18)
    deno = torch.sqrt(torch.sum(y_1 ** 2, dim=1, keepdim=True)) * \
        torch.sqrt(torch.sum(y_pred_1 ** 2, dim=1, keepdim=True))  # (B,1,18)

    minim = torch.tensor(0.000001,dtype=torch.float64)  # avoid division by 0
    if cuda_avail:
        minim = minim.to(nb_gpu)
        deno = deno.to(nb_gpu)
        nume = nume.to(nb_gpu)
    nume = nume + minim
    deno = deno + minim
    my_loss = torch.div(nume, deno)  # (B,1,18)
    my_loss = torch.sum(my_loss)
    return -my_loss
