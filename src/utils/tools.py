import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import json
import re






        
         
                

def get_phonemes_list(phonemes_json):
    with open(phonemes_json, 'r') as f:
        phoneme_list = json.load(f)
    return phoneme_list

def one_hot_to_phonemes(one_hot_tensor, phoneme_list):
    
    zero_mask = torch.all(one_hot_tensor == 0, dim=-1)  # Mask où tous les éléments sont zéro
    
    # Utiliser argmax pour récupérer les indices des phonèmes
    phoneme_indices = torch.argmax(one_hot_tensor, dim=-1)  # dim=-1 pour appliquer sur la dernière dimension (one-hot)
    
    # Remplacer les indices là où il n'y a pas de correspondance (tout zéro)
    phoneme_indices[zero_mask] = -1  # Utiliser un indice invalide (-1) là où tous les éléments sont zéro
    
    # Associer les indices aux phonèmes, avec gestion des valeurs à zéro
    phonemes = []
    for i in phoneme_indices.view(-1).tolist():
        if i == -1:
            phonemes.append('0')  # Ajouter le phonème par défaut pour les zéros
        else:
            phonemes.append(phoneme_list[i])  # Ajouter le phonème correspondant
    # Associer les indices aux phonèmes
    #phonemes = [phoneme_list[i] for i in phoneme_indices.view(-1).tolist()]  # view(-1) pour aplatir la tensor
    np_phonemes = np.array(phonemes)

    # Reshaper le tableau pour obtenir la forme [10, 30, 1]
    reshaped_np_phonemes = np_phonemes.reshape(one_hot_tensor.size(0), 1)
    return reshaped_np_phonemes



def one_hot_to_phonemes_np(one_hot_tensor, phoneme_list):
    # Create a mask for all-zero rows
    zero_mask = np.all(one_hot_tensor == 0, axis=-1)  # axis=-1 applies along the last dimension

    # Use argmax to retrieve indices of phonemes
    phoneme_indices = np.argmax(one_hot_tensor, axis=-1)  # axis=-1 for the last dimension

    # Replace indices where all elements are zero
    phoneme_indices[zero_mask] = -1  # Use an invalid index (-1) where all elements are zero

    # Map indices to phonemes, handling all-zero cases
    phonemes = []
    for i in phoneme_indices.flatten():  # Flatten the array to iterate
        if i == -1:
            phonemes.append('0')  # Add the default phoneme for all-zero cases
        else:
            phonemes.append(phoneme_list[i])  # Add the corresponding phoneme

    # Convert the list of phonemes to a numpy array
    np_phonemes = np.array(phonemes)

    # Reshape the array to have shape [10, 30, 1]
    reshaped_np_phonemes = np_phonemes.reshape(one_hot_tensor.shape[0], 1)
    return reshaped_np_phonemes



def clean_filename(phoneme):
    # Remplace les caractères interdits par des underscores
    return re.sub(r'[<>:"/\\|?*]', '_', phoneme)



def create_folder(folder_path: str) -> str:
    """
    Creates a folder if it doesn't already exist.

    This function checks if a folder exists at the specified path, and if not, creates it. It also prints a message indicating
    whether the folder was created or already exists.

    Args:
        folder_path (str): The path where the folder will be created.

    Returns:
        str: The path to the created or existing folder.

    Example:
        folder = create_folder("results/experiment1")
        # If the folder doesn't exist, it will be created and the path will be returned.
    """
    current_directory = os.getcwd()
    folder = os.path.join(current_directory, folder_path)
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Folder '{folder}' created successfully.")
    else:
        print(f"Folder '{folder}' already exists.")
    return folder


