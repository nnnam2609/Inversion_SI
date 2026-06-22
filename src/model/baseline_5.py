import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from src.utils.tools import get_phonemes_list, one_hot_to_phonemes_np, create_folder
from src.utils.metrics import pearson_correlation, loss_function, loss_rmse
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

from src.utils.evaluation import tract_variables
class BaselineModel(nn.Module):
    def __init__(self,input_dimension, hidden_dimension, num_layers, output_dimension, nbr_articulators, num_phonemes, list_phonemes, phonemes= False, image= False, batch_norma=False):
        super(BaselineModel, self).__init__()
        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension
        self.output_dimension = output_dimension
        self.nbr_articulators = nbr_articulators
        self.num_layers = num_layers
        self.num_phonemes = num_phonemes
        self.list_phonemes = list_phonemes
        self.phonemes = phonemes
        self.image = image
        # Feed forward layer
        # Concatenate, not multiplied
        ff_input_dim = input_dimension + num_phonemes if phonemes else input_dimension
        self.ff_layer_1 = torch.nn.Linear(ff_input_dim, hidden_dimension) # First layer
        self.ff_layer_2 = torch.nn.Linear(hidden_dimension, hidden_dimension) # Second layer

        # Bidirectional LSTM layer
        self.lstm_layer_1 = nn.LSTM(input_size = hidden_dimension, hidden_size= hidden_dimension, num_layers= num_layers, batch_first=True, bidirectional=True)
        self.lstm_layer_2 = nn.LSTM(input_size = hidden_dimension*2, hidden_size= hidden_dimension, num_layers= num_layers, batch_first=True, bidirectional=True)
        
        # Readout layer
        self.readout_layer = torch.nn.Linear(hidden_dimension*2, output_dimension*nbr_articulators)
        
        # Phonemes layer 
        self.phoneme_layer = torch.nn.Linear(hidden_dimension*2, 44)
        
        # Image layer
        self.image_layer = torch.nn.Linear(hidden_dimension*2, 136*136)
        
        self.dropout = torch.nn.Dropout(p=0.5)
        
        #Batch
        #self.batch_norm_1 =  torch.nn.BatchNorm1d(hidden_dimension*2)
        #self.batch_norm_2 =  torch.nn.BatchNorm1d(hidden_dimension*2)
        
        self.batch_norma = batch_norma
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=output_dimension)
        self.cutoff = 10
        self.sampling_rate = 100
        self.lowpass = None
        self.filter_type = 'fix'
        self.init_filter_layer()
        # Dictionnaire pour stocker les activations
        self.activations = {}

        # Ajouter les hooks pour capturer les activations
        self.hooks = []
                
    
    def forward(self, x_mfccs, lengths, x_phonemes = None, x_images= None, filter_output=None):
        lengths = lengths.cpu()
        # Si les phonèmes sont présents, on les concatène aux MFCCs
        if x_phonemes is not None:
            # Si x_phonemes a 4 dimensions, on le squeeze pour obtenir 3 dimensions
            if x_phonemes.dim() == 4:
                x_phonemes = x_phonemes.squeeze(2)  # Remove extra dimension
            
            # Concatenate MFCCs and phonemes along the feature dimension
            x = torch.cat([x_mfccs, x_phonemes], dim=-1)  # Shape: [batch, seq_len, input_dim + num_phonemes]
            
        else :            
            x = x_mfccs 
        output_ff_layer_1 = torch.nn.functional.relu(self.ff_layer_1(x))
        #output_ff_layer_1 = self.dropout(output_ff_layer_1)

        output_ff_layer_2 = torch.nn.functional.relu(self.ff_layer_2(output_ff_layer_1))
        #output_ff_layer_2 = self.dropout(output_ff_layer_2)

        # 1 Bi-lstm
        packed_input = pack_padded_sequence(output_ff_layer_2, lengths, batch_first=True, enforce_sorted=False)
        packed_output_lstm_layer_1, _ = self.lstm_layer_1(packed_input)

        # Unpack the sequences to return to padded form
        output_lstm_layer_1, _ = pad_packed_sequence(packed_output_lstm_layer_1, batch_first=True)

        dim = output_lstm_layer_1.shape[0]
        if self.batch_norma :
            lstm_batch = output_lstm_layer_1.view(dim, 2*self.hidden_dimension,-1)
            lstm_batch = torch.nn.functional.relu(self.batch_norm_1(lstm_batch))
            output_lstm_layer_1= lstm_batch.view(dim,  -1,2 * self.hidden_dimension)
        output_lstm_layer_1 = torch.nn.functional.relu(output_lstm_layer_1)
        #output_lstm_layer_1 = self.dropout(output_lstm_layer_1)

        # 2nd Bi-LSTM layer
        packed_input_lstm_2 = pack_padded_sequence(output_lstm_layer_1, lengths, batch_first=True, enforce_sorted=False)
        packed_output_lstm_layer_2, _ = self.lstm_layer_2(packed_input_lstm_2)
    
        # Unpack sequences after the second LSTM
        output_lstm_layer_2, _ = pad_packed_sequence(packed_output_lstm_layer_2, batch_first=True)
        if self.batch_norma :
            lstm_batch = output_lstm_layer_2.view(dim, 2*self.hidden_dimension,-1)
            lstm_batch = torch.nn.functional.relu(self.batch_norm_2(lstm_batch))
            output_lstm_layer_2= lstm_batch.view(dim,  -1,2 * self.hidden_dimension)
        output_lstm_layer_2 = torch.nn.functional.relu(output_lstm_layer_2)
        #output_lstm_layer_2 = self.dropout(output_lstm_layer_2)

        output_readout = self.readout_layer(output_lstm_layer_2)
        output_contours = output_readout.view(output_readout.size(0),  output_readout.size(1), self.nbr_articulators, self.output_dimension)

        phoneme_output = None
        if self.phonemes and self.image:
            # Phoneme and image prediction
            phoneme_output = self.phoneme_layer(output_lstm_layer_2)
            image_output = self.image_layer(output_lstm_layer_2)
        
        elif self.phonemes:
            # Phoneme prediction
            phoneme_output = self.phoneme_layer(output_lstm_layer_2)
            image_output = None
        
        elif self.image:
            # Image prediction
            image_output = self.image_layer(output_lstm_layer_2)
            

        else :
            phoneme_output = None
            image_output = None
            
        return output_contours, phoneme_output, image_output
    
    def get_filter_weights(self):
        """
        :return: low pass filter weights based on calculus exclusively using tensors so pytorch compatible
        """
        cutoff = torch.tensor(self.cutoff, dtype=torch.float64,requires_grad=True).view(1, 1)
        fc = torch.div(cutoff,
              self.sampling_rate)  # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
        if fc > 0.5:
            raise Exception("cutoff frequency must be at least twice sampling rate")
        b = 0.08  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
        N = int(np.ceil((4 / b)))  # le window
        if not N % 2:
            N += 1  # Make sure that N is odd .
        self.N = N

        n = torch.arange(N)
        alpha = torch.mul(fc, 2 * (n - (N - 1) / 2))
        minim = torch.tensor(0.01, dtype=torch.float64) #utile ?
        alpha = torch.max(alpha,minim)#utile ?
        h = torch.div(torch.sin(alpha), alpha)
        beta = n * 2 * math.pi / (N - 1)
        w = 0.5 * (1 - torch.cos(beta))  # Compute hanning window.
        h = torch.mul(h, w)  # Multiply sinc filter with window.
        h = torch.div(h, torch.sum(h))
        return h

    def get_filter_weights_en_dur(self):
        """
        :return:  low pass filter weights using classical primitive (not on tensor)
        """
        fc = self.cutoff / self.sampling_rate
        if fc > 0.5:
            raise Exception("La frequence de coupure doit etre au moins deux fois la frequence dechantillonnage")
        b = 0.08  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
        N = int(np.ceil((4 / b)))  # le window
        if not N % 2:
            N += 1  # Make sure that N is odd.
        self.N = N
        n = np.arange(N)
        h = np.sinc(fc * 2 * (n - (N - 1) / 2))
        w = 0.5 * (1 - np.cos(n * 2 * math.pi / (N - 1)))  # Compute hanning window.
        h = h * w
        h = h / np.sum(h)
        return torch.tensor(h)

    def init_filter_layer(self):
        """
        intialize the weights of the convolution in the NN,  so that it acts as a lowpass filter.
        the typefilter determines if the weights will be updated during the optim of the NN
        """


        # maybe the two functions do exactly the same...
        if self.filter_type in ["out","fix"] :
            weight_init = self.get_filter_weights_en_dur()
        elif self.filter_type == "unfix":
            weight_init = self.get_filter_weights()
        C_in = 1
        stride = 1
        must_be_5 = 5
        padding = int(0.5 * ((C_in - 1) * stride - C_in + must_be_5)) + 23
        weight_init = weight_init.view((1, 1, -1))
        lowpass = torch.nn.Conv1d(C_in, self.output_dimension, self.N, stride=1, padding=padding, bias=False)

        if self.filter_type == "unfix":  # we let the weights move
            lowpass.weight = torch.nn.Parameter(weight_init,requires_grad=True)

        else :  # "out" we don't care the filter won't be applied, or "fix" the wieghts are fixed
            lowpass.weight = torch.nn.Parameter(weight_init,requires_grad=False)

        lowpass = lowpass
        self.lowpass = lowpass

    def filter_layer(self, y):
        """
        :param y: (B,L,18) articulatory prediction not smoothed
        :return:  smoothed articulatory prediction
        apply the convolution to each articulation, maybe not the best solution (changing n_channel of the conv layer ?)
        """
        B = len(y)
        L = len(y[0])
        y = y
        y_smoothed = torch.zeros(B, L, self.output_dimension)
        for i in range(self.output_dimension):
            traj_arti = y[:, :, i].view(B, 1, L)
            traj_arti_smoothed = self.lowpass(traj_arti)
            traj_arti_smoothed = traj_arti_smoothed.view(B, L)
            y_smoothed[:, :, i] = traj_arti_smoothed
        return y_smoothed
    
    def test(self, X_test, Y_test, length_sequences, length_seq, rank, std_test, mean_test, F_test, Z_test, phonemes = None, autoencoder=None):
        output_dimension = Y_test.size(-1)
        sequences_dimension = length_seq
        rmse_unormalized_articulators = np.empty((0, self.nbr_articulators, output_dimension))
        total_rmse_articulators = np.empty((0, self.nbr_articulators))
        total_mse_articulators = np.empty((0, self.nbr_articulators))
        total_pearson_articulators = np.empty((0, self.nbr_articulators))
        total_loss_articulators = np.empty((0, self.nbr_articulators))
        x_phonemes = None
        correct_predictions = 0
        total_predictions = 0
        
        concatenated_y_pred = {}
        concatenated_y = {}

        padded_concatenated_y_pred = {}
        padded_concatenated_y = {}
        
        concatenated_phonemes = None
        concatenated_frames = None
        self.eval()  # Set model to evaluation mode
        torch.cuda.empty_cache()  # Clear unused memory
        torch.cuda.memory_summary(device=rank, abbreviated=False)  # Print memory summary for debugging
        with torch.no_grad():  # Disable gradient computation
            for i in range(len(X_test)):
                x_torch = X_test[i]
                x_torch = x_torch.unsqueeze(0)
                x_torch = x_torch.to(rank)
                       
                y = Y_test[i]
                y = y.to(rank)
                frames = F_test[i].detach().cpu().numpy()
                phoemes_lab = Z_test[i].detach().cpu().numpy()
                if phonemes is not None :
                    z = Z_test[i]
                    x_phonemes = z.unsqueeze(0)
                    x_phonemes = x_phonemes.to(rank)
                    z = z.to(rank)

                length_sequence = length_sequences[i]
                length = length_sequences[i]
                length_sequence = length_sequence.to(rank)
                length_sequence = length_sequence.unsqueeze(0)

                y_pred, z_pred = self(x_torch, length_sequence, x_phonemes)
                y = y.transpose(0, 1)
                y_pred = y_pred.transpose(1, 2)
                y_pred = y_pred.to(rank)
                if autoencoder :
                    y_pred = autoencoder.module.decoder(y_pred)
                y_pred_np = y_pred.squeeze(0)

                y_pred_np = y_pred_np.cpu().detach().numpy()

                y_np = y.cpu().detach().numpy()
                
                y = y.unsqueeze(0)
                
                std_tensor = std_test[i]
                mean_tensor = mean_test[i]
                
                std_tensor = std_tensor.transpose(0, 1)
                mean_tensor = mean_tensor.transpose(0, 1)
                
                std_np = std_tensor.cpu().detach().numpy()
                mean_np = mean_tensor.cpu().detach().numpy()

                # std_np = std_test[i].cpu().detach().numpy()
                # mean_np = mean_test[i].cpu().detach().numpy()
                
                
                rmse = np.zeros((self.nbr_articulators, output_dimension))

                #Padded 
 

                frame_rmse_articulators = np.empty(self.nbr_articulators)
                frame_mse_articulators = np.empty(self.nbr_articulators)
                frame_pearson_articulators = np.empty(self.nbr_articulators)
                frame_loss_articulators = np.empty(self.nbr_articulators)
                
                y_np_sliced = y_np[:,:length.item()]
                phoemes_sliced = phoemes_lab[:length.item()]
                frames_sliced = frames[:length.item()]
                # original 
                std_np_sliced = std_np[:,:length.item()]
                mean_np_sliced = mean_np[:,:length.item()]

                # Add first array
                if concatenated_phonemes is None:
                    concatenated_phonemes = phoemes_sliced
                    concatenated_frames = frames_sliced
                else:
                    concatenated_phonemes = np.concatenate((concatenated_phonemes, phoemes_sliced))
                    concatenated_frames = np.concatenate((concatenated_frames, frames_sliced))

                #Not padded 
                for j in range(self.nbr_articulators):
                    y_np_sliced_art = np.expand_dims(y_np_sliced[j], axis=0)
                    std_np_art = np.expand_dims(std_np_sliced[j], axis=0)
                    mean_np_art = np.expand_dims(mean_np_sliced[j], axis=0)
                    y_pred_np_art = np.expand_dims(y_pred_np[j], axis=0)
                    y_np_denormalized = (y_np_sliced_art*std_np_art)+mean_np_art
                    y_pred_np_denormalized = (y_pred_np_art*std_np_art)+mean_np_art
                    y_np_denormalized_reshaped = y_np_denormalized.reshape(length.item(), output_dimension)
                    y_pred_np_denormalized_reshaped  = y_pred_np_denormalized.reshape(length.item(), output_dimension)
                    # If this is the first iteration for this articulator, initialize the concatenated array
                    if j not in concatenated_y:
                        concatenated_y[j] = y_np_denormalized_reshaped
                        concatenated_y_pred[j] = y_pred_np_denormalized_reshaped
                    else:
                        # Concatenate the reshaped array with the existing concatenated array
                        concatenated_y[j] = np.concatenate([concatenated_y[j], y_np_denormalized_reshaped], axis=0)
                        concatenated_y_pred[j] = np.concatenate([concatenated_y_pred[j], y_pred_np_denormalized_reshaped], axis=0)

                    #Padded 
                    pad_dim = (sequences_dimension - length.item())
                    padded_y_pred = F.pad(y_pred, (0, 0, pad_dim, 0), mode='constant', value=0)
                    padded_y_pred_np = padded_y_pred.squeeze(0).cpu().detach().numpy()
                    
                    y_np_art = np.expand_dims(y_np[j], axis=0)
                    std_np_art_padded = np.expand_dims(std_np[j], axis=0)
                    mean_np_art_padded = np.expand_dims(mean_np[j], axis=0)
                    padded_y_pred_np_art = np.expand_dims(padded_y_pred_np[j], axis=0)
                    
                    if phonemes is not None :
                        padded_z_pred = F.pad(z_pred, (0, 0, pad_dim, 0), mode='constant', value=0)

                    padded_y_np_denormalized = (y_np_art*std_np_art_padded)+mean_np_art_padded
                    padded_y_pred_np_denormalized = (padded_y_pred_np_art*std_np_art_padded)+mean_np_art_padded
                
                    padded_y_np_denormalized_reshaped = padded_y_np_denormalized.reshape(sequences_dimension, output_dimension)
                    padded_y_pred_np_denormalized_reshaped  = padded_y_pred_np_denormalized.reshape(sequences_dimension, output_dimension)
                    
                    # If this is the first iteration, initialize the concatenated array
                    if j not in padded_concatenated_y:
                        padded_concatenated_y[j] = padded_y_np_denormalized_reshaped
                        padded_concatenated_y_pred[j] = padded_y_pred_np_denormalized_reshaped
                    else:
                        # Concatenate the reshaped array with the existing concatenated array
                        padded_concatenated_y[j] = np.concatenate([padded_concatenated_y[j], padded_y_np_denormalized_reshaped], axis=0)
                        padded_concatenated_y_pred[j] = np.concatenate([padded_concatenated_y_pred[j], padded_y_pred_np_denormalized_reshaped], axis=0)



                for j in range(self.nbr_articulators):
                    
                    rmse_j = np.sqrt(np.mean(np.square(y_np_sliced[j] - y_pred_np[j]), axis=0))  # calculate rmse
                    rmse_j = np.reshape(rmse_j, (1, output_dimension))

                    # Original
                    rmse[j] = rmse_j*std_np[j][0]
                    y_articulators = y[:, j, :, :]
                    y_pred_articulators = padded_y_pred[:, j, :, :]
                    
                    frame_rmse_articulators[j] = loss_rmse(y_articulators, y_pred_articulators)
                    frame_mse_articulators[j] = torch.nn.MSELoss(reduction='mean')(y_articulators, y_pred_articulators)
                    frame_pearson_articulators[j] = pearson_correlation(y_articulators, y_pred_articulators)
                    frame_loss_articulators[j] = loss_function(y_articulators, y_pred_articulators)

                    
                
                rmse_unormalized_articulators = np.concatenate((rmse_unormalized_articulators, np.expand_dims(rmse, axis=0)), axis=0)
                total_rmse_articulators = np.concatenate((total_rmse_articulators, np.expand_dims(frame_rmse_articulators, axis=0)), axis=0)
                total_mse_articulators = np.concatenate((total_mse_articulators, np.expand_dims(frame_mse_articulators, axis=0)), axis=0)
                total_pearson_articulators = np.concatenate((total_pearson_articulators, np.expand_dims(frame_pearson_articulators, axis=0)), axis=0)
                total_loss_articulators = np.concatenate((total_loss_articulators, np.expand_dims(frame_loss_articulators, axis=0)), axis=0)
                

                torch.cuda.empty_cache()  # Clear unused memory

                if phonemes is not None :
                    sequence_length, _, _ = z.shape
                    z_reduced = z.mean(dim=1)  # Now shape: [batch, sequence, phonemes]
                    z_pred_reduced = padded_z_pred.squeeze(0)
                    # Step 2: Iterate over each batch and sequence to compare predictions
                    for seq in range(sequence_length):
                        # Get predicted phoneme (argmax)
                        pred_phoneme = torch.argmax(z_pred_reduced[seq])

                        # Get true phoneme (argmax of averaged values)
                        true_phoneme = torch.argmax(z_reduced[seq])

                        # Compare and count correct predictions
                        if pred_phoneme == true_phoneme:
                            correct_predictions += 1
                        total_predictions += 1
        
        # Not padded
        
        phoneme_list = get_phonemes_list(self.list_phonemes)
        all_phonemes = one_hot_to_phonemes_np(concatenated_phonemes, phoneme_list)
        non_silent_mask = all_phonemes.flatten() != '#'
        all_phonemes = all_phonemes[non_silent_mask]
        non_silent_count = np.sum(non_silent_mask)
        flattened_phonemes = [''.join(phoneme) for phoneme in all_phonemes]
        
        rmse_mean_per_image = np.zeros((self.nbr_articulators, non_silent_count))
        
        unique_phonemes = set(map(str, flattened_phonemes))
        rmse_mean_per_image_per_articulator = np.zeros((self.nbr_articulators))
        rmse_median_per_image_per_articulator = np.zeros((self.nbr_articulators))
        rmse_std_per_image_per_articulator = np.zeros((self.nbr_articulators))
        rmse_per_phoneme = []
        
        rmse_mean_per_point = np.zeros((self.nbr_articulators,output_dimension))
        rmse_mean_per_point_per_articulator = np.zeros((self.nbr_articulators))
        rmse_median_per_point_per_articulator = np.zeros((self.nbr_articulators))
        rmse_std_per_point_per_articulator = np.zeros((self.nbr_articulators))
        
        
        concatenated_y_list = list(concatenated_y.values())
        concatenated_y_np = np.stack(concatenated_y_list, axis=0)
        
        concatenated_y_pred_list = list(concatenated_y_pred.values())
        concatenated_y_pred_np = np.stack(concatenated_y_pred_list, axis=0)
        
        

        
        concatenated_y_vt = concatenated_y_np[:,non_silent_mask]
        concatenated_y_pred_vt = concatenated_y_pred_np[:,non_silent_mask]
        
        frames_np = np.stack(concatenated_frames, axis=0)
        frames_vt = frames_np[non_silent_mask]
        
        #tract_variables(concatenated_y_vt, concatenated_y_pred_vt, all_phonemes, frames_vt)
        
        for j in range(self.nbr_articulators):
            rmse_mean_per_image[j] = np.sqrt(np.mean(np.square(concatenated_y[j][non_silent_mask] - concatenated_y_pred[j][non_silent_mask]), axis=1))  # calculate rmse
            #rmse_means = np.sqrt(np.mean(np.square(concatenated_y[j][non_silent_mask] - concatenated_y_pred[j][non_silent_mask]), axis=1))  # calculate rmse
            
            rmse_mean_per_image_per_articulator[j] = np.mean(rmse_mean_per_image[j])
            rmse_median_per_image_per_articulator[j] = np.median(rmse_mean_per_image[j])
            rmse_std_per_image_per_articulator[j] = np.std(rmse_mean_per_image[j])

            rmse_mean_per_point[j] = np.sqrt(np.mean(np.square(concatenated_y[j] - concatenated_y_pred[j]), axis=0))  # calculate rmse
            rmse_mean_per_point_per_articulator[j] = np.mean(rmse_mean_per_point[j])
            rmse_median_per_point_per_articulator[j] = np.median(rmse_mean_per_point[j])
            rmse_std_per_point_per_articulator[j] = np.std(rmse_mean_per_point[j])

            
            
            rmse_phoneme = {}
            for idx, phoneme in enumerate(unique_phonemes):
                phoneme_indices = np.where(np.array(flattened_phonemes) == phoneme)[0]
                rmse_mean_per_phoneme = np.mean(rmse_mean_per_image[j, phoneme_indices])
                rmse_median_per_phoneme  = np.median(rmse_mean_per_image[j, phoneme_indices])
                rmse_std_per_phoneme  = np.std(rmse_mean_per_image[j, phoneme_indices])
                
                rmse_phoneme[phoneme] = {
                                        "mean": rmse_mean_per_phoneme,
                                        "median": rmse_median_per_phoneme,
                                        "std": rmse_std_per_phoneme
                                    }
            rmse_per_phoneme.append(rmse_phoneme)
        rmse_mean_all_image = np.mean(rmse_mean_per_image)
        rmse_median_all_image = np.median(rmse_mean_per_image)
        rmse_std_all_image = np.std(rmse_mean_per_image)
        rmse_mean_per_image_all_articulator = np.mean(rmse_mean_per_image, axis =0)
        
        rmse_mean_all_point = np.mean(rmse_mean_per_point)
        rmse_median_all_point = np.median(rmse_mean_per_point)
        rmse_std_all_point = np.std(rmse_mean_per_point)
        rmse_mean_per_point_all_articulator = np.mean(rmse_mean_per_point, axis =0)
        rmse_phoneme_all_articulators = {}
            
        for idx, phoneme in enumerate(unique_phonemes):
            phoneme_indices = np.where(np.array(flattened_phonemes) == phoneme)[0]
            rmse_mean_per_phoneme = np.mean(rmse_mean_per_image[:, phoneme_indices])
            rmse_median_per_phoneme  = np.median(rmse_mean_per_image[:, phoneme_indices])
            rmse_std_per_phoneme  = np.std(rmse_mean_per_image[:, phoneme_indices])
            
            rmse_phoneme_all_articulators[phoneme] = {
                                    "mean": rmse_mean_per_phoneme,
                                    "median": rmse_median_per_phoneme,
                                    "std": rmse_std_per_phoneme
                                }
            
        rmse_articulators_mean = np.mean(total_rmse_articulators, axis=0)
        rmse_mean = np.mean(rmse_articulators_mean)

        mse_articulators_mean = np.mean(total_mse_articulators, axis=0)
        mse_mean = np.mean(mse_articulators_mean)
        
        pearson_articulators_mean = np.mean(total_pearson_articulators, axis=0)
        pearson_mean = np.mean(pearson_articulators_mean)
        
        loss_articulators_mean = np.mean(total_loss_articulators, axis=0)
        loss_mean = np.mean(loss_articulators_mean)

        
        # Unormalized
        rmse_unormalized_mean_per_articulator_per_point = np.mean(rmse_unormalized_articulators, axis=0)
        rmse_unormalized_mean_per_articulator = np.mean(rmse_unormalized_mean_per_articulator_per_point, axis=1)
        rmse_unormalized_mean = np.mean(rmse_unormalized_mean_per_articulator)
        
        rmse_unormalized_median_art = np.median(rmse_unormalized_mean_per_articulator_per_point, axis=1)
        rmse_unormalized_std_art= np.std(rmse_unormalized_mean_per_articulator_per_point, axis=1)

        rmse_unormalized_median_all = np.median(rmse_unormalized_articulators)
        rmse_unormalized_std_all = np.std(rmse_unormalized_articulators)
        if phonemes is not None :
            accuracy = (correct_predictions / total_predictions) * 100
        else :
            accuracy = 0
            # Returning the values in a dictionary
            
        y_list = list(concatenated_y.values())  # Extract arrays
        # Stack along the second axis to get shape (13822, 8, 100)
        y_array = np.stack(y_list, axis=1)

        
        
        y_pred_list = list(concatenated_y_pred.values())  # Extract arrays
        # Stack along the second axis to get shape (13822, 8, 100)
        y_pred_array = np.stack(y_pred_list, axis=1)
        
        

        print(f'accuracy : {accuracy}')
        result_dict = {
            'rmse_mean': rmse_mean,
            'mse_mean': mse_mean,
            'pearson_mean': pearson_mean,
            'loss_mean': loss_mean,
            'rmse_unormalized_mean_per_articulator_per_point': rmse_unormalized_mean_per_articulator_per_point,
            'rmse_unormalized_mean_per_articulator': rmse_unormalized_mean_per_articulator,
            'rmse_unormalized_mean': rmse_unormalized_mean,
            'accuracy': accuracy,
            'rmse_unormalized_median_art': rmse_unormalized_median_art,
            'rmse_unormalized_std_art': rmse_unormalized_std_art,
            'rmse_unormalized_median_all': rmse_unormalized_median_all,
            'rmse_unormalized_std_all': rmse_unormalized_std_all,
            'rmse_mean_per_image': rmse_mean_per_image,
            'rmse_mean_per_point': rmse_mean_per_point,
            'rmse_mean_per_image_all_articulator': rmse_mean_per_image_all_articulator,
            'rmse_mean_per_point_all_articulator': rmse_mean_per_point_all_articulator,
            'rmse_mean_all_image': rmse_mean_all_image,
            'rmse_median_all_image': rmse_median_all_image,
            'rmse_std_all_image': rmse_std_all_image,
            'rmse_mean_all_point': rmse_mean_all_point,
            'rmse_median_all_point': rmse_median_all_point,
            'rmse_std_all_point': rmse_std_all_point,
            'rmse_mean_per_image_per_articulator': rmse_mean_per_image_per_articulator,
            'rmse_median_per_image_per_articulator': rmse_median_per_image_per_articulator,
            'rmse_std_per_image_per_articulator': rmse_std_per_image_per_articulator,
            'rmse_mean_per_point_per_articulator': rmse_mean_per_point_per_articulator,
            'rmse_median_per_point_per_articulator': rmse_median_per_point_per_articulator,
            'rmse_std_per_point_per_articulator': rmse_std_per_point_per_articulator,
            'all_phonemes': all_phonemes,
            'all_frames' : concatenated_frames,
            'rmse_per_phoneme' : rmse_per_phoneme,
            'rmse_phoneme_all_articulators' : rmse_phoneme_all_articulators
        }
        return result_dict, y_array, y_pred_array