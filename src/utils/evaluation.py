import os
import pandas as pd
from src.utils.tract_variable import calculate_vocal_tract_variables
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure
from src.utils.tools import get_phonemes_list, one_hot_to_phonemes_np, one_hot_to_phonemes
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score
import joblib
from src.common.resources import log_process_memory as log_memory
                        
def read_npy(arr, path):

    int_arr = np.round(arr).astype(int)
    if int_arr[2] == arr[2]:
        lower_path = os.path.join(path, f"{int_arr[0]}", f"S{int_arr[1]}", "inference_contours", f"{int_arr[2]:04d}_lower-incisor.npy")
        upper_path = os.path.join(path, f"{int_arr[0]}", f"S{int_arr[1]}", "inference_contours", f"{int_arr[2]:04d}_upper-incisor.npy")
        image_lower = np.load(lower_path)
        image_upper = np.load(upper_path)
    else:
        rounded_down = int(np.floor(arr[2]))
        rounded_up = rounded_down + 1
                        
        down_lower_path = os.path.join(path, f"{int_arr[0]}", f"S{int_arr[1]}", "inference_contours", f"{rounded_down:04d}_lower-incisor.npy")
        up_lower_path   = os.path.join(path, f"{int_arr[0]}", f"S{int_arr[1]}", "inference_contours", f"{rounded_up:04d}_lower-incisor.npy")
        down_image_lower = np.load(down_lower_path)
        up_image_lower = np.load(up_lower_path)
        image_lower = (down_image_lower + up_image_lower) / 2
        
        
        down_upper_path = os.path.join(path, f"{int_arr[0]}", f"S{int_arr[1]}", "inference_contours", f"{rounded_down:04d}_upper-incisor.npy")
        up_upper_path   = os.path.join(path, f"{int_arr[0]}", f"S{int_arr[1]}", "inference_contours", f"{rounded_up:04d}_upper-incisor.npy")
        down_image_upper = np.load(down_upper_path)
        up_image_upper = np.load(up_upper_path)
        image_upper = (down_image_upper + up_image_upper) / 2
    return image_upper, image_lower


def tract_variables(y, y_pred, phonemes, frames_name, folder_run, datadir):
    """
    Args:
        sentences_ids (str): Unique id of each sentence to save the results.
        frame_ids (List)
        outputs (torch.tensor): Tensor with shape (bs, seq_len, n_articulators, 2, n_samples).
        targets (torch.tensor): Tensor with shape (bs, seq_len, n_articulators, 2, n_samples).
        lengths (List): List with the length of each sentence in the batch.
        phonemes (List): List with the sequence of phonemes for each sentence in the batch.
        articulators (List[str]): List of articulators.
        save_to (str): Path to the directory to save the results.
    """
    articulators = [
    "arytenoid-cartilage",
    "epiglottis",
    "lower-lip",
    "pharynx",
    "soft-palate-midline",
    "tongue",
    "upper-lip",
    "vocal-folds",
    "upper-incisor",
    #"lower-incisor"
    ]
    
    path = datadir
    TVs_data = []
    TVs_data_pred = []
    pearson_data = []
    
    print(f"Folder run: {folder_run}")
    # Initialisation des trajectoires TV (par TV)
    TV_trajectories_target = {}
    TV_trajectories_pred = {}

    # pca_c1 = np.load("config/pca/pca_c1.npy")
    # pca_mean = np.load("config/pca/pca_mean.npy")
    
    # Load saved scaler and PCA
    pca_scaler = joblib.load("config/pca/scaler_with_silence_25.pkl")
    pca_c1 = joblib.load("config/pca/pca_with_silence_25.pkl")
    # Initialiser les noms des TVs une fois au début
    sample_upper, sample_lower = read_npy(frames_name[0], path)
    sample_frame = y[:, 0].reshape(8, 50, 2)
    #sample_concat = np.concatenate((sample_frame, sample_upper[np.newaxis, :, :], sample_lower[np.newaxis, :, :]), axis=0)
    sample_concat = np.concatenate((sample_frame, sample_upper[np.newaxis, :, :]), axis=0)
    sample_dict = {art: tensor for art, tensor in zip(articulators, sample_concat)}
    TV_names = list(calculate_vocal_tract_variables(sample_dict, frames_name[0], phonemes[0], folder_run, pca_c1, pca_scaler))
    
    
    for TV in TV_names:
        TV_trajectories_target[TV] = []
        TV_trajectories_pred[TV] = []
    
    for index in range(y.shape[1]):
        upper, lower = read_npy(frames_name[index], path)
        y_frame = y[:, index].reshape(8, 50, 2)
        y_frame_contatenate = np.concatenate((y_frame, upper[np.newaxis, :, :], lower[np.newaxis, :, :]), axis=0)
        y_dict = {
                art: tensor for art, tensor in zip(articulators, y_frame_contatenate)
            }
        y_TVs = calculate_vocal_tract_variables(y_dict, frames_name[index], phonemes[index], folder_run, pca_c1, pca_scaler, prefix="target")
        
        y_pred_frame = y_pred[:, index].reshape(8, 50, 2)
        y_pred_frame_contatenate = np.concatenate((y_pred_frame, upper[np.newaxis, :, :], lower[np.newaxis, :, :]), axis=0)
        
        
        y_pred_dict = {
                art: tensor for art, tensor in zip(articulators, y_pred_frame_contatenate)
            }
        y_pred_TVs = calculate_vocal_tract_variables(y_pred_dict, frames_name[index], phonemes[index], folder_run, pca_c1, pca_scaler,prefix="pred")
        
        item = {
            "frame": frames_name[index],
            "phoneme": phonemes[index]
        }
                
        # Remplissage des valeurs target * 1.62
        for TV, TV_dict in y_TVs.items():
            if TV_dict is None:
                item[f"{TV}_target"] = np.nan
                TV_trajectories_target[TV].append(np.nan)
            item.update({
                f"{TV}_target": TV_dict["value"],
                f"{TV}_target_poc_1_x": TV_dict["poc_1"][0].item(),
                f"{TV}_target_poc_1_y": TV_dict["poc_1"][1].item(),
                f"{TV}_target_poc_2_x": TV_dict["poc_2"][0].item(),
                f"{TV}_target_poc_2_y": TV_dict["poc_2"][1].item()
            })
            TV_trajectories_target[TV].append(TV_dict["value"])
        TVs_data.append(item.copy())  # copie pour éviter d'écraser les données
          
            
        pred_item = {
        "frame": frames_name[index],
        "phoneme": phonemes[index]
        }
          
        for TV, TV_dict in y_pred_TVs.items():
            if TV_dict is None:
                pred_item[f"{TV}_pred"] = np.nan
                TV_trajectories_pred[TV].append(np.nan)

            pred_item.update({
                f"{TV}_pred": TV_dict["value"],
                f"{TV}_pred_poc_1_x": TV_dict["poc_1"][0].item(),
                f"{TV}_pred_poc_1_y": TV_dict["poc_1"][1].item(),
                f"{TV}_pred_poc_2_x": TV_dict["poc_2"][0].item(),
                f"{TV}_pred_poc_2_y": TV_dict["poc_2"][1].item()
            })
            TV_trajectories_pred[TV].append(TV_dict["value"])
        TVs_data_pred.append(pred_item.copy())  
            
            
    # === Pearson des trajectoires (corrélation sur toute la séquence) ===
    trajectory_corr_data = []
    for TV in TV_names:
        true_vals = np.array(TV_trajectories_target[TV])
        pred_vals = np.array(TV_trajectories_pred[TV])

        if np.isnan(true_vals).any() or np.isnan(pred_vals).any():
            corr = np.nan
        else:
            if TV == "VEL":
                true_vals = np.sign(true_vals)  # → [-1, -1,  0,  1,  1]
                true_vals[true_vals == 0] = 1
                
                pred_vals = np.sign(pred_vals)  # → [-1, -1,  0,  1,  1]
                pred_vals[pred_vals == 0] = 1
                
                accuracy = accuracy_score(true_vals, pred_vals)
                trajectory_corr_data.append({
                    "TV_name": f"accuracy_{TV}",
                    "trajectory_pearson": accuracy
                    }) 
            corr, _ = pearsonr(true_vals, pred_vals)
        trajectory_corr_data.append({
            "TV_name": TV,
            "trajectory_pearson": corr
        })


            
    # Sauvegarde dans CSV
    pd.DataFrame(TVs_data).to_csv(os.path.join(folder_run, "tract_variables_target.csv"), index=False)
    pd.DataFrame(TVs_data_pred).to_csv(os.path.join(folder_run, "tract_variables_pred.csv"), index=False)
    pd.DataFrame(trajectory_corr_data).to_csv(os.path.join(folder_run, "trajectory_pearson_correlation.csv"), index=False)

def test_model(self, 
               mfccs_test, contours_test, length_sequences, length_seq, rank, std_test, mean_test, 
               frames_test, phonemes_test, folder_run, datadir, 
               mris_test=None, std_mri=None, mean_mri=None, 
               phonemes=None, autoencoder=None, skip_tract_variables=False):
    
    contour_dimension = contours_test.size(-1)
    sequences_dimension = length_seq
    
    rmse_contours_per_sequence_per_articulators = np.empty((0, self.nbr_articulators, contour_dimension))
    
    
    x_phonemes = None
    correct_predictions = 0
    total_predictions = 0
    
    concatenated_contours_pred = {}
    concatenated_contours = {}
    
    concatenated_contours_pred_normalized = {}
    concatenated_contours_normalized = {}
    
    concatenated_phonemes = None
    concatenated_frames = None
    
    concatenated_mris = None
    concatenated_mris_pred = None
    
    ssim_scores_all_images = 0
    phoneme_list = get_phonemes_list(self.list_phonemes)
    self.eval()  # Set model to evaluation mode
    torch.cuda.empty_cache()  # Clear unused memory
    torch.cuda.memory_summary(device=rank, abbreviated=False)  # Print memory summary for debugging
    with torch.no_grad():  # Disable gradient computation
        for i in range(len(mfccs_test)):
            #log_memory(stage=f"Test {i}")
            mfcc_torch = mfccs_test[i].unsqueeze(0).to(rank)       
            contour_torch = contours_test[i].to(rank)
            frames = frames_test[i]
            phoemes_lab = phonemes_test[i]
            
            if phonemes is not None :
                phoneme_torch = phonemes_test[i].unsqueeze(0).to(rank)
                x_phonemes = phoneme_torch
                x_phonemes = x_phonemes.to(rank)


            if mris_test is not None :
                mri_torch = mris_test[i].unsqueeze(0).to(rank)
                images = mri_torch
                
            length_sequence = length_sequences[i]
            length = length_sequences[i]
            length_sequence = length_sequence.to(rank)
            length_sequence = length_sequence.unsqueeze(0)

            contour_pred, phoneme_pred, mri_pred = self(mfcc_torch, length_sequence, x_phonemes)
            contour_pred = contour_pred.to(rank)
            
            if autoencoder :
                contour_pred = autoencoder.module.decoder(contour_pred)
                
            # Fin des prédictions
            
            # From Tensor to Numpy
            
            contour_np = contour_torch.cpu().detach().numpy() # numpy[sequence_length, nb_artculators, coordinates]
            contour_pred_np = np.squeeze(contour_pred.cpu().detach().numpy(), axis=0) # numpy[1, sequence_length, nb_artculators, coordinates]
            std_np  = std_test[i].cpu().detach().numpy() # numpy[sequence_length, nb_artculators, coordinates]
            mean_np = mean_test[i].cpu().detach().numpy() # numpy[sequence_length, nb_artculators, coordinates]
            contour_torch = contour_torch.unsqueeze(0) # tensor[batch, sequence_length, nb_artculators, coordinates]
            
            rmse_normalized_sequence_contours_articulators = np.zeros((self.nbr_articulators, contour_dimension)) # numpy[nb_artculators, coordinates]
            
            contour_np_sliced = contour_np[:length.item()] # numpy[sequence_length_sliced, nb_artculators, coordinates]
            phoneme_sliced_tensor = phoemes_lab[:length.item()]
            phoneme_sliced = phoneme_sliced_tensor.detach().cpu().numpy() # numpy[sequence_length_sliced, 1, one_hot(44)]
            frame_sliced = frames[:length.item()].detach().cpu().numpy() # numpy[sequence_length_sliced, frame_title(3)]
            
            
            
            
            # original 
            # std_np_sliced = std_np[:length.item()] # numpy[sequence_length_sliced, nb_artculators, coordinates]
            # mean_np_sliced = mean_np[:length.item()] # numpy[sequence_length_sliced, nb_artculators, coordinates]
            
            std_np_sliced = std_np # numpy[sequence_length_sliced, nb_artculators, coordinates]
            mean_np_sliced = mean_np # numpy[sequence_length_sliced, nb_artculators, coordinates]

            # Add first array
            if concatenated_phonemes is None:
                concatenated_phonemes = phoneme_sliced
                concatenated_frames = frame_sliced
            else:
                concatenated_phonemes = np.concatenate((concatenated_phonemes, phoneme_sliced))
                concatenated_frames = np.concatenate((concatenated_frames, frame_sliced))
                
            for j in range(self.nbr_articulators):
                
                contour_np_sliced_art = contour_np_sliced[:,j] # numpy[1, sequence_length_sliced, coordinates]
                std_np_art = std_np_sliced[:,j] # numpy[1, sequence_length_sliced, coordinates]
                mean_np_art = mean_np_sliced[:,j] # numpy[1, sequence_length_sliced, coordinates]
                contour_pred_np_art = contour_pred_np[:,j] # numpy[1, sequence_length_sliced, coordinates]
                
                #rmse_contours_per_point_per_articulator = np.sqrt(np.mean(np.square(concatenated_contours[j] - concatenated_contours_pred[j]), axis=0))
                
                # rmse_contours_unormalized_per_point_per_articulator = np.sqrt(np.mean(np.square(concatenated_contours[j] - concatenated_contours_pred[j]), axis=0))
                
                contour_np_denormalized = (contour_np_sliced_art*std_np_art)+mean_np_art # numpy[1, sequence_length_sliced, coordinates]
                contour_pred_np_denormalized = (contour_pred_np_art*std_np_art)+mean_np_art # numpy[1, sequence_length_sliced, coordinates]
                contour_np_denormalized= contour_np_denormalized.reshape(length.item(), contour_dimension) # numpy[sequence_length_sliced, coordinates]
                contour_pred_np_denormalized  = contour_pred_np_denormalized.reshape(length.item(), contour_dimension) # numpy[sequence_length_sliced, coordinates]
                
                # If this is the first iteration for this articulator, initialize the concatenated array
                if j not in concatenated_contours:
                    concatenated_contours[j] = contour_np_denormalized
                    concatenated_contours_pred[j] = contour_pred_np_denormalized
                    # Normaalized
                    concatenated_contours_normalized[j] = contour_np_sliced_art
                    concatenated_contours_pred_normalized[j] = contour_pred_np_art
                else:
                    # Concatenate the reshaped array with the existing concatenated array
                    concatenated_contours[j] = np.concatenate([concatenated_contours[j], contour_np_denormalized], axis=0)
                    concatenated_contours_pred[j] = np.concatenate([concatenated_contours_pred[j], contour_pred_np_denormalized], axis=0)
                    # Normalized
                    concatenated_contours_normalized[j] = np.concatenate([concatenated_contours_normalized[j], contour_np_sliced_art], axis=0)
                    concatenated_contours_pred_normalized[j] = np.concatenate([concatenated_contours_pred_normalized[j], contour_pred_np_art], axis=0)
                    
            if mris_test is not None :

                mri_np = mri_torch.cpu().detach().numpy().squeeze(0)
                mri_pred_np = mri_pred.cpu().detach().numpy().squeeze(0)
                std_mri_np  = std_mri[i].detach().numpy()
                mean_mri_np  = mean_mri[i].detach().numpy()
                
                
                mri_np = mri_np[:length.item()]
                # std_mri_np = std_mri_np[:length.item()]
                # mean_mri_np = mean_mri_np[:length.item()]
                 # numpy[sequence_length, nb_artculators, coordinates]
                
                
                
                 # numpy[sequence_length, nb_artculators, coordinates]
                
                mri_np_denormalized = (mri_np*std_mri_np)+mean_mri_np # numpy[1, sequence_length_sliced, coordinates]
                mri_pred_np_denormalized = (mri_pred_np*std_mri_np)+mean_mri_np # numpy[1, sequence_length_sliced, coordinates]

                if concatenated_mris is None:
                    concatenated_mris = mri_np_denormalized
                    concatenated_mris_pred = mri_pred_np_denormalized
                else:
                    concatenated_mris = np.concatenate([concatenated_mris, mri_np_denormalized], axis=0)
                    concatenated_mris_pred = np.concatenate([concatenated_mris_pred, mri_pred_np_denormalized], axis=0)

            #phoneme_numpy = phoneme_sliced.detach().cpu().numpy()
            non_slient_phonemes = one_hot_to_phonemes(phoneme_sliced_tensor, phoneme_list)
            non_silent_mask = non_slient_phonemes.flatten() != '#'
            contour_non_silent = contour_np_sliced[non_silent_mask]
            contour_pred_non_silent = contour_pred_np[non_silent_mask]
            for k in range(self.nbr_articulators):
                if np.any(non_silent_mask):
                    rmse_normalized_sequence_contours_articulator = np.sqrt(np.mean(np.square(contour_non_silent[:,k] - contour_pred_non_silent[:,k]), axis=0))  # calculate rmse
                    rmse_normalized_sequence_contours_articulator = np.reshape(rmse_normalized_sequence_contours_articulator, (1, contour_dimension))
                    rmse_normalized_sequence_contours_articulators[k] = rmse_normalized_sequence_contours_articulator*std_np[0,k]
            
            if np.any(non_silent_mask):
                rmse_contours_per_sequence_per_articulators = np.concatenate((rmse_contours_per_sequence_per_articulators, np.expand_dims(rmse_normalized_sequence_contours_articulators, axis=0)), axis=0)
            
            torch.cuda.empty_cache()  # Clear unused memory

            if phonemes is not None :
                # phoneme_torch has shape [batch, seq_len, 1, num_phonemes] or [batch, seq_len, num_phonemes]
                if phoneme_torch.dim() == 4:
                    batch_size, sequence_length, _, num_phonemes = phoneme_torch.shape
                    # Squeeze the extra dimension: [batch, seq_len, 1, 44] -> [batch, seq_len, 44]
                    phoneme_torch = phoneme_torch.squeeze(2)
                else:
                    batch_size, sequence_length, num_phonemes = phoneme_torch.shape
                
                z_reduced = phoneme_torch.squeeze(0)  # Remove batch dimension: [seq_len, num_phonemes]
                pad_dim = (sequences_dimension - length.item())
                padded_z_pred = F.pad(phoneme_pred, (0, 0, pad_dim, 0), mode='constant', value=0)
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
                    
                
                    
                # Step 2: Iterate over each batch and sequence to compare predictions   
                
    phoneme_list = get_phonemes_list(self.list_phonemes)
    all_phonemes = one_hot_to_phonemes_np(concatenated_phonemes, phoneme_list)
    non_silent_mask = all_phonemes.flatten() != '#'
    #non_silent_mask = all_phonemes.flatten() != '/'
    all_phonemes = all_phonemes[non_silent_mask]
    non_silent_count = np.sum(non_silent_mask)
    flattened_phonemes = [''.join(phoneme) for phoneme in all_phonemes]
    
    
    rmse_contours_per_image_per_articulator = np.zeros((self.nbr_articulators, non_silent_count))
    rmse_mris_per_image = np.zeros(non_silent_count)
    
    unique_phonemes = set(map(str, flattened_phonemes))
    
    rmse_contours_mean_all_images_per_articulator = np.zeros((self.nbr_articulators))
    rmse_contours_median_all_images_per_articulator = np.zeros((self.nbr_articulators))
    rmse_contours_std_all_image_per_articulator = np.zeros((self.nbr_articulators))
    rmse_contours_per_phoneme_per_articulator = []
    rmse_contours_per_point_per_articulator = np.zeros((self.nbr_articulators,contour_dimension))
    rmse_contours_mean_all_point_per_articulator = np.zeros((self.nbr_articulators))
    rmse_contours_median_all_point_per_articulator = np.zeros((self.nbr_articulators))
    rmse_contours_std_all_point_per_articulator = np.zeros((self.nbr_articulators))
    
    
    
    
    concatenated_contours_list = list(concatenated_contours.values())
    concatenated_contours_np = np.stack(concatenated_contours_list, axis=0)
    
    concatenated_contours_pred_list = list(concatenated_contours_pred.values())
    concatenated_contours_pred_np = np.stack(concatenated_contours_pred_list, axis=0)
    
    

    
    has_valid_frames = concatenated_frames is not None and concatenated_frames.size and np.any(concatenated_frames)
    frames_vt = concatenated_frames[non_silent_mask] if has_valid_frames else None

    if skip_tract_variables:
        print("Skipping tract-variable metrics because skip_tract_variables=true.")
    elif self.nbr_articulators == 8 and has_valid_frames:
        concatenated_contours_vt = concatenated_contours_np[:,non_silent_mask]
        concatenated_contours_pred_vt = concatenated_contours_pred_np[:,non_silent_mask]
        tract_variables(concatenated_contours_vt, concatenated_contours_pred_vt, all_phonemes, frames_vt, folder_run, datadir)
        log_memory(stage=f"Tract variable")
    else:
        print(
            "Skipping tract-variable metrics because they require 8 articulators and valid frame ids, "
            f"got {self.nbr_articulators} articulators and valid_frames={bool(has_valid_frames)}. "
            "Contour RMSE metrics will still be computed.",
            flush=True,
        )
    if mris_test is not None :
        device = torch.device(f"cuda:{rank}")
        rmse_mris_per_image, rmse_mris_mean_all_images, rmse_mris_median_all_images, rmse_mris_std_all_images, ssim_scores_all_images, ssim_scores_per_image = mri_calcul(concatenated_mris, concatenated_mris_pred, non_silent_mask, device)
    else :
        rmse_mris_mean_all_images= rmse_mris_median_all_images= rmse_mris_std_all_images= ssim_scores_all_images= 0
        rmse_mris_per_image = ssim_scores_per_image = []    
    
    for j in range(self.nbr_articulators):
        rmse_contours_per_image_per_articulator[j] = np.sqrt(np.mean(np.square(concatenated_contours[j][non_silent_mask] - concatenated_contours_pred[j][non_silent_mask]), axis=1))  # calculate rmse
        rmse_contours_mean_all_images_per_articulator[j] = np.mean(rmse_contours_per_image_per_articulator[j])
        rmse_contours_median_all_images_per_articulator[j] = np.median(rmse_contours_per_image_per_articulator[j])
        rmse_contours_std_all_image_per_articulator[j] = np.std(rmse_contours_per_image_per_articulator[j])

        rmse_contours_per_point_per_articulator[j] = np.sqrt(np.mean(np.square(concatenated_contours[j] - concatenated_contours_pred[j]), axis=0))  # calculate rmse
        rmse_contours_mean_all_point_per_articulator[j] = np.mean(rmse_contours_per_point_per_articulator[j])
        rmse_contours_median_all_point_per_articulator[j] = np.median(rmse_contours_per_point_per_articulator[j])
        rmse_contours_std_all_point_per_articulator[j] = np.std(rmse_contours_per_point_per_articulator[j])

        
        
        rmse_contours_phoneme_per_articulator = {}
        for idx, phoneme in enumerate(unique_phonemes):
            phoneme_indices = np.where(np.array(flattened_phonemes) == phoneme)[0]
            rmse_mean_per_phoneme = np.mean(rmse_contours_per_image_per_articulator[j, phoneme_indices])
            rmse_median_per_phoneme  = np.median(rmse_contours_per_image_per_articulator[j, phoneme_indices])
            rmse_std_per_phoneme  = np.std(rmse_contours_per_image_per_articulator[j, phoneme_indices])
            
            rmse_contours_phoneme_per_articulator[phoneme] = {
                                    "mean": rmse_mean_per_phoneme,
                                    "median": rmse_median_per_phoneme,
                                    "std": rmse_std_per_phoneme
                                }
        rmse_contours_per_phoneme_per_articulator.append(rmse_contours_phoneme_per_articulator)
    log_memory(stage=f"calculate contours")
    rmse_contours_mean_all_images_all_articulators = np.mean(rmse_contours_per_image_per_articulator)
    rmse_contours_median_all_images_all_articulators = np.median(rmse_contours_per_image_per_articulator)
    rmse_contours_std_all_images_all_articulators = np.std(rmse_contours_per_image_per_articulator)
    rmse_contours_mean_per_image_all_articulators = np.mean(rmse_contours_per_image_per_articulator, axis =0)
    
    rmse_contours_mean_all_points_all_articulators = np.mean(rmse_contours_per_point_per_articulator)
    rmse_contours_median_all_points_all_articulators = np.median(rmse_contours_per_point_per_articulator)
    rmse_contours_std_all_points_all_articulators = np.std(rmse_contours_per_point_per_articulator)
    rmse_contours_mean_per_point_all_articulators = np.mean(rmse_contours_per_point_per_articulator, axis =0)
    
    
    rmse_contours_phoneme_all_articulators = {}
        
    for idx, phoneme in enumerate(unique_phonemes):
        phoneme_indices = np.where(np.array(flattened_phonemes) == phoneme)[0]
        rmse_mean_per_phoneme = np.mean(rmse_contours_per_image_per_articulator[:, phoneme_indices])
        rmse_median_per_phoneme  = np.median(rmse_contours_per_image_per_articulator[:, phoneme_indices])
        rmse_std_per_phoneme  = np.std(rmse_contours_per_image_per_articulator[:, phoneme_indices])
        
        rmse_contours_phoneme_all_articulators[phoneme] = {
                                "mean": rmse_mean_per_phoneme,
                                "median": rmse_median_per_phoneme,
                                "std": rmse_std_per_phoneme
                            }
        


    log_memory(stage=f"Begin unormalized")
    # Unormalized
    rmse_contours_mean_all_sequences_per_articulator_per_point = np.mean(rmse_contours_per_sequence_per_articulators, axis=0)
    rmse_contours_mean_all_sequences_per_articulator = np.mean(rmse_contours_mean_all_sequences_per_articulator_per_point, axis=1)
    rmse_contours_mean_all_sequences_all_articulators = np.mean(rmse_contours_mean_all_sequences_per_articulator)
    
    rmse_contours_median_all_sequences_per_articulator = np.median(rmse_contours_mean_all_sequences_per_articulator_per_point, axis=1)
    rmse_contours_std_all_sequences_per_articulator= np.std(rmse_contours_mean_all_sequences_per_articulator_per_point, axis=1)

    rmse_contours_median_all_sequences_all_articulators = np.median(rmse_contours_per_sequence_per_articulators)
    rmse_contours_std_all_sequences_all_articulators = np.std(rmse_contours_per_sequence_per_articulators)
    
    
    if phonemes is not None :
        accuracy = (correct_predictions / total_predictions) * 100
    else :
        accuracy = 0
        # Returning the values in a dictionary
        
    y_list = list(concatenated_contours.values())  # Extract arrays
    # Stack along the second axis to get shape (13822, 8, 100)
    y_array = np.stack(y_list, axis=1)
    y_array = y_array[non_silent_mask]
    
    
    y_pred_list = list(concatenated_contours_pred.values())  # Extract arrays
    # Stack along the second axis to get shape (13822, 8, 100)
    y_pred_array = np.stack(y_pred_list, axis=1)
    y_pred_array = y_pred_array[non_silent_mask]
    
    log_memory(stage=f"Dict")
    print(f'accuracy : {accuracy}')
    result_dict = {
        'rmse_contours_mean_all_sequences_per_articulator_per_point': rmse_contours_mean_all_sequences_per_articulator_per_point,
        'rmse_contours_mean_all_sequences_per_articulator': rmse_contours_mean_all_sequences_per_articulator,
        'rmse_contours_mean_all_sequences_all_articulators': rmse_contours_mean_all_sequences_all_articulators,
        'rmse_contours_median_all_sequences_per_articulator': rmse_contours_median_all_sequences_per_articulator,
        'rmse_contours_median_all_sequences_all_articulators': rmse_contours_median_all_sequences_all_articulators,
        'rmse_contours_std_all_sequences_per_articulator': rmse_contours_std_all_sequences_per_articulator,
        'rmse_contours_std_all_sequences_all_articulators': rmse_contours_std_all_sequences_all_articulators,
        
        
        'rmse_contours_per_image_per_articulator': rmse_contours_per_image_per_articulator,
        'rmse_contours_mean_per_image_all_articulators': rmse_contours_mean_per_image_all_articulators,
        'rmse_contours_mean_all_images_all_articulators': rmse_contours_mean_all_images_all_articulators,
        'rmse_contours_mean_all_images_per_articulator': rmse_contours_mean_all_images_per_articulator,
        'rmse_contours_median_all_images_all_articulators': rmse_contours_median_all_images_all_articulators,
        'rmse_contours_median_all_images_per_articulator': rmse_contours_median_all_images_per_articulator,
        'rmse_contours_std_all_image_per_articulator': rmse_contours_std_all_image_per_articulator,
        'rmse_contours_std_all_images_all_articulators': rmse_contours_std_all_images_all_articulators,
        
        
        
        'rmse_contours_per_point_per_articulator': rmse_contours_per_point_per_articulator,
        'rmse_contours_mean_per_point_all_articulators': rmse_contours_mean_per_point_all_articulators,
        'rmse_contours_mean_all_points_all_articulators': rmse_contours_mean_all_points_all_articulators,
        'rmse_contours_mean_all_point_per_articulator': rmse_contours_mean_all_point_per_articulator,
        
        'rmse_contours_median_all_points_all_articulators': rmse_contours_median_all_points_all_articulators,
        'rmse_contours_median_all_point_per_articulator': rmse_contours_median_all_point_per_articulator,
        
        'rmse_contours_std_all_points_all_articulators': rmse_contours_std_all_points_all_articulators,
        'rmse_contours_std_all_point_per_articulator': rmse_contours_std_all_point_per_articulator,
        
        'rmse_mris_per_image': rmse_mris_per_image,
        'rmse_mris_mean_all_images': rmse_mris_mean_all_images,
        'rmse_mris_median_all_images': rmse_mris_median_all_images,
        'rmse_mris_std_all_images': rmse_mris_std_all_images,
        'ssim_scores_all_images': ssim_scores_all_images,
        'ssim_scores_per_image': ssim_scores_per_image,
        
        
        
        'accuracy': accuracy,
        'all_phonemes': all_phonemes,
        'all_frames' : frames_vt,
        'rmse_contours_per_phoneme_per_articulator' : rmse_contours_per_phoneme_per_articulator,
        'rmse_contours_phoneme_all_articulators' : rmse_contours_phoneme_all_articulators
    }
    return result_dict, y_array, y_pred_array

def mri_calcul(concatenated_mris, concatenated_mris_pred, non_silent_mask, device):
        concatenated_mris = concatenated_mris[non_silent_mask]
        concatenated_mris_pred = concatenated_mris_pred[non_silent_mask]
        
        rmse_mris_per_image = np.sqrt(np.mean(np.square(concatenated_mris - concatenated_mris_pred), axis=(1,2)))  # calculate rmse
        rmse_mris_mean_all_images = np.mean(rmse_mris_per_image)
        rmse_mris_median_all_images = np.median(rmse_mris_per_image)
        rmse_mris_std_all_images = np.std(rmse_mris_per_image)
        
        ssim_scores_per_image = np.zeros((len(concatenated_mris)))
        for i in range(len(concatenated_mris)):
            mri_tensor = torch.tensor(concatenated_mris[i], dtype=torch.float32).reshape(1, 1, 136, 136)
            mri_pred_tensor = torch.tensor(concatenated_mris_pred[i], dtype=torch.float32).reshape(1, 1, 136, 136)

            data_min = torch.min(torch.min(mri_tensor), torch.min(mri_pred_tensor))
            data_max = torch.max(torch.max(mri_tensor), torch.max(mri_pred_tensor))
            data_range = data_max - data_min

            ssim_scores_per_image[i] = structural_similarity_index_measure(mri_tensor, mri_pred_tensor, data_range=data_range)

        ssim_scores_all_images = np.mean(ssim_scores_per_image)
        return rmse_mris_per_image, rmse_mris_mean_all_images, rmse_mris_median_all_images, rmse_mris_std_all_images, ssim_scores_all_images, ssim_scores_per_image 
