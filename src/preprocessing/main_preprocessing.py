# project/preprocessing/main_preprocessing.py
import os
from sympy import im
import torchaudio
import textgrid
import json
import librosa
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import psutil
import subprocess
import time
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from transformers import AutoFeatureExtractor, HubertModel


def _numeric_id(value):
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if not digits:
        raise ValueError(f"Cannot extract numeric id from {value!r}")
    return float(digits)


def _dataset_type_for_sequence(config, sequence):
    sequence_key = str(sequence)
    dataset_types = config.get("dataset_types", {})
    if sequence_key in dataset_types:
        return str(dataset_types[sequence_key]).lower()
    dataset_type = str(config.get("dataset_type", "asd2")).lower()
    if dataset_type == "mixed":
        return "asd1" if sequence_key.upper().startswith("P") else "asd2"
    return dataset_type


def _canonicalize_contour_array(contour):
    """Return flat interleaved x/y contour coordinates: [x1, y1, x2, y2, ...]."""
    contour = np.asarray(contour)
    if contour.shape == (50, 2):
        canonical = contour
    elif contour.shape == (2, 50):
        canonical = contour.T
    elif contour.size == 100:
        canonical = contour.reshape(50, 2)
    else:
        raise ValueError(f"Unexpected contour shape {contour.shape}")
    return canonical.reshape(100)


class Corpus(Dataset):
    """
    A custom dataset class for handling speech and articulatory data.

    This class extends `torch.utils.data.Dataset` and is designed to process a dataset containing speech-related data,
    including phonemes, audio files, text grids, and images. It extracts and organizes features for model training.

    Args:
        config (dict): Configuration dictionary containing dataset parameters, including:
            - 'n_mfcc' (int): Number of MFCC coefficients to extract.
            - 'window_length_ms' (int): Window length in milliseconds for feature extraction.
            - 'hop_length_ratio' (float): Hop length ratio for spectrogram calculation.
            - 'added_frames' (int): Number of additional frames to consider.
            - 'ms_image' (int): Duration of each image frame in milliseconds.
            - 'phonemesdir' (str): Path to the phoneme dictionary file.
        sequences (list): A list of sequence names to include in the dataset.

    Attributes:
        sequences (list): The list of sequences used in the dataset.
        config (dict): Configuration dictionary with dataset parameters.
        all_phonemes (dict): A dictionary mapping phonemes to one-hot encoded representations.
        audio_files (list): List of paths to audio files in the dataset.
        textgrid_files (list): List of paths to TextGrid annotation files.
        images_files (list): List of paths to articulatory image data files.
        n_mfcc (int): Number of MFCC features to extract.
        window_length_ms (int): Window length for processing.
        hop_length_ratio (float): Hop length ratio for feature extraction.
        added_frames (int): Number of additional frames considered in the dataset.
        ms_image (int): Duration of each articulatory image frame in milliseconds.
        skip_ms (int): The number of milliseconds to skip in processing, calculated as `added_frames * ms_image`.

    Methods:
        - __len__: Returns the number of samples in the dataset.
        - __getitem__: Retrieves a single sample from the dataset.
        - get_phonemes_list: Loads and processes the phoneme-to-one-hot encoding dictionary.
        - collect_files: (Assumed) Collects and organizes dataset files (not implemented here).
    """
    
    def __init__(self, config: dict, sequences: list, rank:int):
        """
        Initializes the Corpus dataset by loading phoneme mappings, collecting files, 
        and storing dataset parameters.

        Args:
            config (dict): Configuration dictionary containing dataset parameters.
            sequences (list): List of sequence names to be included in the dataset.

        Returns:
            None
        """
        self.rank = rank
        self.sequences = sequences
        self.config = config
        self.all_phonemes = self.get_phonemes_list()
        self.audio_files, self.textgrid_files, self.images_files = self.collect_files()
        print(f"Finish collecting files")
        self.n_mfcc = config['n_mfcc']
        self.window_length_ms = config['window_length_ms']
        self.hop_length_ratio = config['hop_length_ratio']
        self.added_frames = config['added_frames']
        self.ms_image = config['ms_image']
        self.skip_ms = self.added_frames * self.ms_image
        
    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The total number of samples available in the dataset.

        Example:
            dataset_length = len(dataset)
        """
        return len(self.concatenated_input)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves a single sample from the dataset based on the index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - 'features' (torch.Tensor): Input features for the model.
                - 'labels' (torch.Tensor): Ground truth labels.
                - 'frames' (torch.Tensor): Frame indices for the data sample.
                - 'phonemes' (torch.Tensor): One-hot encoded phoneme representation.
                - 'std' (torch.Tensor): Standard deviation for normalization.
                - 'mean' (torch.Tensor): Mean value for normalization.
                - 'mean_datas' (torch.Tensor): Mean values for all dataset features.
                - 'length_datas' (torch.Tensor): Length of dataset sequences.
                - 'sequences_length' (int): The length of the sequence for this sample.
        """
        return {'features': self.concatenated_input[idx],
                'labels': self.concatenated_labels[idx],
                'frames': self.concatenated_frames[idx],
                # 'images': self.concatenated_images[idx],
                # 'p5': self.concatenated_p5_mris[idx],
                # 'p95': self.concatenated_p95_mris[idx],
                'phonemes': self.concatenated_phonemes_one_hot[idx], 
                'std': self.concatenated_std[idx],
                'mean': self.concatenated_mean[idx],
                'mean_datas': self.concatenated_mean_datas,
                'length_datas' : self.concatenated_length_datas,
                'sequences_length' : self.sequences_length[idx]}
    
    def get_phonemes_list(self) -> dict:
        """
        Loads and processes a phoneme-to-one-hot encoding dictionary from a JSON file.

        This function reads the phoneme list from a JSON file specified in the `config['phonemesdir']` path.
        It then generates a mapping of phonemes to their corresponding one-hot encoded vectors.

        Returns:
            dict: A dictionary mapping phonemes (str) to one-hot encoded NumPy arrays.

        Example:
            phonemes = dataset.get_phonemes_list()
            phoneme_vector = phonemes['aa']  # Retrieves the one-hot encoding for 'aa'
        """
        with open(self.config['phonemesdir'], 'r') as f:
            phoneme_list = json.load(f)
        phoneme_to_one_hot = {phoneme: np.eye(len(phoneme_list))[i] for i, phoneme in enumerate(phoneme_list)}
        return phoneme_to_one_hot  
              
    def collect_files(self) -> tuple:
        """
        Collects and organizes dataset files for audio, text, and image data.

        This method scans directories to locate and store paths for audio files, TextGrid annotation files, 
        and articulatory image data.

        Returns:
            tuple: A tuple containing three lists:
                - audio_files (list): Paths to audio files.
                - textgrid_files (list): Paths to TextGrid annotation files.
                - images_files (list): Paths to articulatory image data.
        """
        audio_files = {}
        textgrid_files = {}
        images_files = {}

        for sequence, sessions in tqdm(self.config[self.sequences].items(), desc=f"Reading {self.sequences}"):
            audio_files[sequence] = []
            textgrid_files[sequence] = []
            images_files[sequence] = []
            for session in sessions:
                dataset_type = _dataset_type_for_sequence(self.config, sequence)
                if dataset_type == "asd1":
                    audio_file, textgrid_file, image_folder = self._resolve_asd1_session(sequence, session)
                elif dataset_type == "asd2":
                    audio_file, textgrid_file, image_folder = self._resolve_asd2_session(sequence, session)
                else:
                    raise ValueError(
                        f"Unsupported dataset_type={dataset_type!r} for sequence={sequence!r}; "
                        "expected 'asd1', 'asd2', or global 'mixed'"
                    )
                audio_files[sequence].append(audio_file)
                textgrid_files[sequence].append(textgrid_file)
                images_files[sequence].append(image_folder)
        return audio_files, textgrid_files, images_files

    def _resolve_asd2_session(self, sequence: str, session: str) -> tuple:
        datadir = self.config.get("asd2_datadir", self.config["datadir"])
        audio_folder = os.path.join(datadir, sequence, session)
        if not os.path.isdir(audio_folder):
            raise FileNotFoundError(f"Missing ASD2 session folder: {audio_folder}")
        contour_candidates = [
            os.path.join(audio_folder, 'inference_contours_registered'),
            os.path.join(audio_folder, 'inference_contours'),
        ]
        image_folder = next((path for path in contour_candidates if os.path.isdir(path)), None)
        wav_candidates = [
            file_name
            for file_name in sorted(os.listdir(audio_folder))
            if file_name.endswith('.wav') and not file_name.endswith('_mocap.wav')
        ]
        if not wav_candidates:
            raise FileNotFoundError(f"No non-mocap WAV found in ASD2 session: {audio_folder}")
        audio_file = wav_candidates[0]
        audio_path = os.path.join(audio_folder, audio_file)
        name = os.path.splitext(audio_file)[0]
        textgrid_path = os.path.join(audio_folder, f'{name}_adjusted.textgrid')
        if not os.path.exists(textgrid_path):
            raise FileNotFoundError(f"Missing ASD2 TextGrid: {textgrid_path}")
        if image_folder is None:
            raise FileNotFoundError(
                f"Missing ASD2 contour folder: {contour_candidates[0]} or {contour_candidates[1]}"
            )
        return audio_path, textgrid_path, image_folder

    def _resolve_asd1_session(self, sequence: str, session: str) -> tuple:
        speaker = str(sequence)
        datadir = self.config.get("asd1_datadir", self.config["datadir"])
        other_folder = os.path.join(datadir, speaker, "OTHER", session)
        dcm_folder = os.path.join(datadir, speaker, "DCM_2D", session)
        if not os.path.isdir(other_folder):
            raise FileNotFoundError(f"Missing ASD1 OTHER session folder: {other_folder}")
        if not os.path.isdir(dcm_folder):
            raise FileNotFoundError(f"Missing ASD1 DCM_2D session folder: {dcm_folder}")

        audio_path = os.path.join(other_folder, f"DENOISED_SOUND_{speaker}_{session}.wav")
        textgrid_path = os.path.join(other_folder, f"TEXT_ALIGNMENT_{speaker}_{session}.textgrid")
        contour_root = self.config.get("asd1_annotation_dir")
        if not contour_root:
            raise ValueError("dataset_type='asd1' requires config['asd1_annotation_dir']")
        image_folder = os.path.join(contour_root, speaker, session, "contours")

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Missing ASD1 WAV: {audio_path}")
        if not os.path.exists(textgrid_path):
            raise FileNotFoundError(f"Missing ASD1 TextGrid: {textgrid_path}")
        if not os.path.isdir(image_folder):
            raise FileNotFoundError(f"Missing ASD1 BF contour folder: {image_folder}")
        return audio_path, textgrid_path, image_folder
    
    def print_gpu_memory(self):
        try:
            result = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total", "--format=csv,noheader,nounits"])
            print(f"GPU Memory (used/free/total): {result.decode('utf-8').strip()}")
        except Exception as e:
            print(f"Erreur avec nvidia-smi : {e}")

    def print_ram_usage(self):
        memory_info = psutil.virtual_memory()
        print(f"RAM Usage: {memory_info.percent}% (Used: {memory_info.used / (1024**3):.2f} GB, Total: {memory_info.total / (1024**3):.2f} GB)")
    
    def read_sequence(self) -> tuple:
        """
        Reads and processes the dataset sequences, extracting features and labels.

        This method iterates through the collected audio, annotation, and image files, processing each sequence using
        `compute_datas()`. It extracts:
            - MFCC features from audio
            - Contour-based articulatory data from images
            - Frame information
            - One-hot encoded phoneme representations
            - Normalization parameters (mean, std)

        The data is then converted into PyTorch tensors, padded where necessary.

        Returns:
            tuple:
                - all_padded_features (torch.Tensor): Padded MFCC feature sequences.
                - all_padded_labels (torch.Tensor): Padded articulatory labels (contours).
                - all_padded_frames (torch.Tensor): Padded frame indices.
                - all_padded_phonemes_one_hot (torch.Tensor): Padded one-hot phoneme representations.
                - all_padded_std (torch.Tensor): Padded standard deviation values for normalization.
                - all_padded_mean (torch.Tensor): Padded mean values for normalization.
                - mean_datas (torch.Tensor): Mean values for all dataset features.
                - all_sequences_length (list): Sequence lengths before padding.
                - sequences_length (list): Sequence lengths after processing.
        """
        all_features = None
        all_labels = None
        all_frames = None
        all_phonemes_one_hot = None
        all_std_labels = None
        all_mean_labels = None
        all_p5_mris = None
        all_mris = None
        all_p95_mris = None
        all_sequences_length = None
        
        use_mri = False
        print("Starting to read and process sequences...")
        if self.config['input_type'] == 'wav2vec':    
            self.wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(self.rank)
            self.wav2vec_model.eval()
        
        if self.config['input_type'] == 'hubert':
            self.hubert_processor = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
            self.hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(self.rank)
            # self.hubert_processor = AutoFeatureExtractor.from_pretrained("facebook/hubert-large-ll60k")
            # self.hubert_model = HubertModel.from_pretrained("facebook/hubert-large-ll60k").to(self.rank)
            self.hubert_model.eval()
        for (serie, audio_serie), (serie, tg_serie), (serie, contour_serie) in zip(self.audio_files.items(), self.textgrid_files.items(), self.images_files.items()) :
            mfccs_serie, contours_serie, frames_serie, phonemes_serie_one_hot, std_contour_serie, mean_contour_serie, sequences_length_serie, mris_serie, p5_mris_serie, p95_mris_serie, = self.compute_datas(serie, audio_serie, tg_serie, contour_serie, self.all_phonemes)
            
            # self.print_ram_usage()
            # self.print_gpu_memory()
            
            if all_features is None:
                all_features = mfccs_serie
                all_labels = contours_serie
                all_mris = mris_serie
                all_frames = frames_serie
                all_phonemes_one_hot = phonemes_serie_one_hot
                all_std_labels = std_contour_serie
                all_mean_labels = mean_contour_serie
                all_p5_mris = p5_mris_serie
                all_p95_mris = p95_mris_serie
                all_sequences_length = sequences_length_serie
            else :
                all_features = all_features + mfccs_serie
                all_sequences_length = all_sequences_length + sequences_length_serie
                all_frames = all_frames + frames_serie
                all_phonemes_one_hot = all_phonemes_one_hot + phonemes_serie_one_hot
                all_labels = all_labels + contours_serie
                all_mris = all_mris + mris_serie
                all_std_labels = all_std_labels + std_contour_serie
                all_mean_labels = all_mean_labels + mean_contour_serie
                
                all_p5_mris = list(all_p5_mris) + list(p5_mris_serie)
                all_p95_mris = list(all_p95_mris) + list(p95_mris_serie)
                
                # all_p5_mris = np.concatenate((all_p5_mris, std_mris_serie), axis=0)
                # all_p95_mris = np.concatenate((all_p95_mris, mean_mris_serie), axis=0)
            #Convert input sequences to PyTorch tensors and pad them
        
        # print(f"All tensors :")
        # self.print_ram_usage()
        # self.print_gpu_memory()
            
        tensors_inputs = [torch.tensor(seq, dtype=torch.float32) for seq in all_features]
        all_padded_features = pad_sequence(tensors_inputs, batch_first=True)
        sequences_length = [length.shape[0] for length in tensors_inputs]

        # Pad sequences
        tensor_outputs = [torch.tensor(seq, dtype=torch.float32) for seq in all_labels]
        padded_labels = pad_sequence(tensor_outputs, batch_first=True)  # Adjust if necessary for padding
        
        if use_mri:
            tensor_mris = [torch.tensor(seq, dtype=torch.float32) for seq in all_mris]
            padded_mris = pad_sequence(tensor_mris, batch_first=True)  # Adjust if necessary for padding
            
            tensors_p5_mris = [torch.tensor(seq, dtype=torch.float32) for seq in all_p5_mris]
            all_padded_p5_mris = pad_sequence(tensors_p5_mris, batch_first=True)

            tensors_p95_mris = [torch.tensor(seq, dtype=torch.float32) for seq in all_p95_mris]
            all_padded_p95_mris = pad_sequence(tensors_p95_mris, batch_first=True)
        else: 
            padded_mris = torch.tensor([])
            all_padded_p5_mris = torch.tensor([])
            all_padded_p95_mris = torch.tensor([])

        
        tensor_frames = [torch.tensor(seq, dtype=torch.float32) for seq in all_frames]
        all_padded_frames = pad_sequence(tensor_frames, batch_first=True)  # Adjust if necessary for padding

        tensors_phonemes_one_hot = [torch.tensor(seq, dtype=torch.float32) for seq in all_phonemes_one_hot]
        all_padded_phonemes_one_hot = pad_sequence(tensors_phonemes_one_hot, batch_first=True)
        art = len(tensor_outputs)
        nbr_articulators = len(self.config['classes'])
        all_padded_labels = padded_labels.view(art, self.config['sequence_length'], nbr_articulators, self.config['output_layer'])
        
        mean_datas = self.compute_mean_datas(tensor_outputs)


        tensors_std = [torch.tensor(seq, dtype=torch.float32) for seq in all_std_labels]
        all_padded_std = pad_sequence(tensors_std, batch_first=True)
        
        tensors_mean = [torch.tensor(seq, dtype=torch.float32) for seq in all_mean_labels]
        all_padded_mean = pad_sequence(tensors_mean, batch_first=True)

        return all_padded_features, all_padded_labels, all_padded_frames, all_padded_phonemes_one_hot, all_padded_std, all_padded_mean, padded_mris, mean_datas, all_sequences_length, sequences_length, all_padded_p5_mris, all_padded_p95_mris
    
    def cepstre_librosa(self, audio_signal: np.ndarray, sample_rate: int) -> tuple:
        """
        Computes the cepstrum of an audio signal using Librosa.

        This method extracts cepstral features by performing:
            1. **Short-Time Fourier Transform (STFT)** to obtain the magnitude spectrum.
            2. **Log-power transformation** to compress spectral amplitudes.
            3. **Inverse FFT (cepstrum computation)** to reveal spectral periodicities.
            4. **Dimensionality reduction** (keeping only the first 30 components).

        Args:
            audio_signal (np.ndarray): The raw audio signal.
            sample_rate (int): The sampling rate of the audio signal.

        Returns:
            tuple:
                - cepstrum (np.ndarray): The computed cepstrum with reduced dimensionality (first 30 coefficients).
                - window_length_samples (int): Window length in samples.
                - hop_length_samples (int): Hop length in samples.

        Example:
            cepstrum, win_length, hop_length = dataset.cepstre_librosa(audio_signal, sample_rate)
        """

        window_length_samples = int(sample_rate * self.window_length_ms / 1000)
        hop_length_samples = int(sample_rate * self.hop_length_ratio/1000)
        n_fft = 2 ** int(np.ceil(np.log2(window_length_samples)))
        
        # Compute FFT using librosa
        spectrum = librosa.stft(audio_signal, n_fft=n_fft, hop_length=hop_length_samples, win_length=window_length_samples, window='hamming')
        magnitude_spectrum = np.abs(spectrum)  # Extract first frame
        
        # Compute Cepstrum
        log_power_spectrum = np.log1p(magnitude_spectrum)
        cepstrum = np.fft.ifft(log_power_spectrum)
        cepstrum = cepstrum.T  
        # Take only the first 30 columns → shape (8075, 30)
        cepstrum = cepstrum[:, :30]
        return cepstrum, window_length_samples, hop_length_samples

    
    



    def compute_cepstre(self, audio_signal: torch.Tensor, sample_rate: int) -> tuple:
        """
        Compute the Linear Cepstral Coefficients (LCC) for the entire audio signal using PyTorch.
        
        Parameters:
        - audio_signal (torch.Tensor): The input audio signal as a 1D PyTorch tensor.
        - sr (int): Sampling rate of the audio signal.
        - n_lcc (int): Number of cepstral coefficients (not explicitly used in this function).
        - n_fft (int): FFT size.
        - hop_length (int): Hop length for windowing.
        - win_length (int): Window length for STFT.

        Returns:
        - lcc_features (torch.Tensor): Tensor of computed LCC coefficients for all frames.
        """
        audio_signal = audio_signal.squeeze()
        window_length_samples = int(sample_rate * self.window_length_ms / 1000)

        hop_length_samples = int(sample_rate * self.hop_length_ratio/1000)
        n_fft = 2 ** int(np.ceil(np.log2(window_length_samples)))
        
        num_frames = (audio_signal.shape[0] - window_length_samples) // hop_length_samples + 1
        fr = torch.arange(n_fft, dtype=torch.float32)  # Frequency axis
        hamming = torch.hamming_window(window_length_samples, dtype=torch.float32)  # Hamming window
        
        lcc_features = []

        
        for i in range(num_frames):
            start = i * hop_length_samples
            end = start + window_length_samples

            if end > audio_signal.shape[0]:  # Ensure valid slicing
                break

            # Extract frame
            u = audio_signal[start:end + 1]

            # First difference
            uu = u[:-1] - u[1:] * 0.99

            # Apply Hamming window
            x = hamming * uu

            # Zero padding
            zeroes_padding = torch.zeros(n_fft - window_length_samples, dtype=torch.float32)
            xx = torch.cat((x, zeroes_padding))  # Zero padding

            # Compute FFT
            y = torch.fft.fft(xx, n=n_fft)

            # Compute magnitude spectrum and apply log
            yy = 20 * torch.log10(torch.abs(y) + 1e-6)

            # Compute inverse FFT (Cepstrum)
            yyy = torch.fft.ifft(yy, n=n_fft)
            lcc_cosine = self.dual_cosine_lifter(yyy, 30) # Apply cosine lifter
            # Filter cepstral coefficients (set certain range to zero)
            #yyy[30:482] = 0

            # Compute final FFT after filtering
            #z = torch.abs(torch.fft.fft(yyy, n=n_fft))

            # Store LCC features for this frame
            lcc_features.append(lcc_cosine[:30])  # Keep only the first n_lcc coefficients
        # Convert list to tensor
        lcc_features = torch.stack(lcc_features)

        return lcc_features, window_length_samples, hop_length_samples
        
    def dual_cosine_lifter(self, cepstrum, keep=30):
        """
        Applies a cosine liftering to preserve the first and last `keep` coefficients.

        Parameters:
        - cepstrum: 1D or 2D tensor of shape (..., 512)
        - keep: number of coefficients to preserve at both ends.

        Returns:
        - cepstrum_lifted: cepstrum after applying the cosine window.
        """
        n_coeffs = cepstrum.shape[-1]
        device = cepstrum.device

        # Création de la fenêtre (tensor 1D de taille 512)
        lifter = torch.zeros(n_coeffs, device=device)

        # Partie basse
        n1 = torch.arange(keep, device=device)
        lifter_low = 0.5 * (1 + torch.cos(np.pi * n1 / keep))
        lifter[:keep] = lifter_low

        # Partie haute
        n2 = torch.arange(keep, device=device)
        lifter_high = 0.5 * (1 + torch.cos(np.pi * n2.flip(0) / keep))
        lifter[-keep:] = lifter_high

        # Adaptation du shape si le cepstrum est 2D ou plus
        while len(lifter.shape) < len(cepstrum.shape):
            lifter = lifter.unsqueeze(0)

        return cepstrum * lifter
       
    def compute_mfcc(self, audio_signal: np.ndarray, sample_rate: int) -> tuple:
        """
        Computes Mel-Frequency Cepstral Coefficients (MFCCs) with delta and delta-delta features.

        This function extracts MFCCs from an audio signal using the Hamming window,
        and also calculates the first-order and second-order derivatives (delta and delta-delta).

        Args:
            audio_signal (np.ndarray): A 1D NumPy array representing the audio signal.
            sample_rate (int): The sampling rate of the audio signal.

        Returns:
            tuple: (mfccs_context, window_length_samples, hop_length_samples)
                - mfccs_context (np.ndarray): An array of shape (num_frames, feature_dim) 
                  containing MFCCs with delta and delta-delta features.
                - window_length_samples (int): The number of samples per window.
                - hop_length_samples (int): The number of samples between frames.
        """

        window_length_samples = int(sample_rate * self.window_length_ms / 1000)

        hop_length_samples = int(sample_rate * self.hop_length_ratio/1000)
        n_fft = 2 ** int(np.ceil(np.log2(window_length_samples)))
        mfccs = librosa.feature.mfcc(y=audio_signal,
                                              sr=sample_rate,
                                              n_mfcc=self.n_mfcc,
                                              n_fft=n_fft,
                                              center=True,
                                              hop_length=hop_length_samples,
                                              win_length=window_length_samples,
                                              window='hamming'
                                              ).T

        context_window = self.config['context_window']
        #mfccs_final = mfccs    # Use only MFCC
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(delta_mfccs, order=2)
        mfccs_final = np.concatenate((mfccs, delta_mfccs, delta2_mfccs), axis=1) #Use Delta1 and Delta2 
        
        padding = np.zeros((context_window, mfccs_final.shape[1]))

        # Concatenate padding, original mfccs, and additional padding
        frames = np.concatenate([padding, mfccs_final, padding])

        # Use a loop to create the windowed representation
        mfccs_context = np.array([frames[i:i + 2*context_window + 1].flatten() for i in range(len(mfccs_final))])

        return mfccs_context, window_length_samples, hop_length_samples

    def calculate_norm(self, sequence: str, mfcc_sequence: list, contour_sequence: list) -> tuple:
        """
        Computes normalization statistics for MFCC and articulatory contour sequences.

        This function calculates the mean and standard deviation of both MFCC and contour 
        sequences, as well as a moving average for contour normalization.

        Args:
            sequence (str): Identifier for the sequence, used in saved filenames.
            mfcc_sequence (list): List of MFCC feature arrays for each frame.
            contour_sequence (list): List of articulatory contour arrays per frame.

        Returns:
            tuple: (std_mfcc, mean_mfcc, std_contour, mean_contour, moving_average)
                - std_mfcc (np.ndarray): Standard deviation of MFCC features.
                - mean_mfcc (np.ndarray): Mean of MFCC features.
                - std_contour (np.ndarray): Standard deviation of articulatory contour features.
                - mean_contour (np.ndarray): Mean of articulatory contour features.
                - moving_average (np.ndarray): Smoothed moving average of contour sequences.
        """
        pad = 30
    
        # Reshape des séquences de contours
        
        contour_sequence_reshaped = [
            traj.reshape(traj.shape[0], traj.shape[1] * traj.shape[2])
            for traj in contour_sequence
        ] # [number of sequences, sequence_length, articulator*contours]
        all_mean_contour = np.array([np.mean(traj, axis=0) for traj in contour_sequence_reshaped])
        all_mean_contour = np.concatenate([np.expand_dims(np.pad(all_mean_contour[:, k], (pad, pad), "symmetric"), 1)
                                    for k in range(all_mean_contour.shape[1])], axis=1)
        moving_average = np.array(
        [np.mean(all_mean_contour[k - pad:k + pad], axis=0) for k in range(pad, len(all_mean_contour) - pad)])
        std_contour = np.mean(np.array([np.std(frame, axis=0) for frame in contour_sequence_reshaped]), axis=0)
        mean_contour = np.mean(np.array([np.mean(frame, axis=0) for frame in contour_sequence_reshaped]), axis=0)
        num_features = contour_sequence[0].shape[1]  # Par exemple, 3 articulateurs
        feature_length = contour_sequence[0].shape[2]  # Longueur d'un contour
        
        std_contour = std_contour.reshape(num_features, feature_length)
        mean_contour = mean_contour.reshape(num_features, feature_length)
        moving_average = moving_average.reshape(moving_average.shape[0], num_features, feature_length)  # Calculé sur plusieurs frames
        
        std_mfcc = np.mean(np.array([np.std(frame, axis=0) for frame in mfcc_sequence]), axis=0)
        mean_mfcc = np.mean(np.array([np.mean(frame, axis=0) for frame in mfcc_sequence]), axis=0)
        
        np.save(os.path.join("normalization_values", f"moving_average_contour_{sequence}"), moving_average)
        np.save(os.path.join("normalization_values", f"std_contour_{sequence}"), std_contour)
        np.save(os.path.join("normalization_values", f"mean_contour_{sequence}"), mean_contour)
    
    
        np.save(os.path.join("normalization_values", "std_mfcc_" + sequence), std_mfcc)
        np.save(os.path.join("normalization_values", "mean_mfcc_" + sequence), mean_mfcc)
        # Correction des reshapes (supposant que toutes les trajectoires ont les mêmes dimensions)
        
        return std_mfcc,mean_mfcc, std_contour, mean_contour, moving_average
    
    def calculate_norm0(self, sequence: str, mfcc_sequence: list, contour_sequence: list) -> tuple:
        """
        Computes normalization statistics for MFCC and articulatory contour sequences.

        This function calculates the mean and standard deviation of both MFCC and contour 
        sequences, as well as a moving average for contour normalization.

        Args:
            sequence (str): Identifier for the sequence, used in saved filenames.
            mfcc_sequence (list): List of MFCC feature arrays for each frame.
            contour_sequence (list): List of articulatory contour arrays per frame.

        Returns:
            tuple: (std_mfcc, mean_mfcc, std_contour, mean_contour, moving_average)
                - std_mfcc (np.ndarray): Standard deviation of MFCC features.
                - mean_mfcc (np.ndarray): Mean of MFCC features.
                - std_contour (np.ndarray): Standard deviation of articulatory contour features.
                - mean_contour (np.ndarray): Mean of articulatory contour features.
                - moving_average (np.ndarray): Smoothed moving average of contour sequences.
        """
    
        # contour_sequence_reshaped = [
        # traj.reshape(traj.shape[0], traj.shape[1] * traj.shape[2])
        # for traj in contour_sequence
        # ]
        
        std_contour = np.mean(np.array([np.std(frame, axis=0) for frame in contour_sequence]), axis=0)
        mean_contour = np.mean(np.array([np.mean(frame, axis=0) for frame in contour_sequence]), axis=0)


        std_mfcc = np.mean(np.array([np.std(frame, axis=0) for frame in mfcc_sequence]), axis=0)
        mean_mfcc = np.mean(np.array([np.mean(frame, axis=0) for frame in mfcc_sequence]), axis=0)

        np.save(os.path.join("normalization_values", f"std_contour_{sequence}"), std_contour)
        np.save(os.path.join("normalization_values", f"mean_contour_{sequence}"), mean_contour)
    
    
        np.save(os.path.join("normalization_values", "std_mfcc_" + sequence), std_mfcc)
        np.save(os.path.join("normalization_values", "mean_mfcc_" + sequence), mean_mfcc)
        # Correction des reshapes (supposant que toutes les trajectoires ont les mêmes dimensions)
        
        return std_mfcc,mean_mfcc, std_contour, mean_contour
    
    def normalize_images(self, sequence: str, image_sequence: list) -> tuple:

        contour_sequence_reshaped = [
        traj.reshape(traj.shape[0], traj.shape[1] * traj.shape[2])
        for traj in image_sequence
        ]    # [number of sequences, sequence_length, articulator*contours]


        std_mri = np.mean(np.array([np.std(frame, axis=0) for frame in contour_sequence_reshaped]), axis=0)
        mean_mri = np.mean(np.array([np.mean(frame, axis=0) for frame in contour_sequence_reshaped]), axis=0)     
        np.save(os.path.join("normalization_values", "std_mri_" + sequence), std_mri)
        np.save(os.path.join("normalization_values", "mean_mri_" + sequence), mean_mri)

        
        return std_mri, mean_mri
    
    def robust_global_minmax_normalize(self, data_list, lower_percentile=5, upper_percentile=95):
        # 1. Rassembler toutes les valeurs dans un seul tableau plat
        all_values = np.concatenate([seq.flatten() for seq in data_list])
        # 2. Calcul des bornes robustes
        p5 = np.percentile(all_values, lower_percentile)
        p95 = np.percentile(all_values, upper_percentile)

        # 3. Appliquer la normalisation à chaque séquence
        normalized_data_list = []
        # for seq in data_list:
        #     norm_seq = (seq - p5) / (p95 - p5 + 1e-8)
        #     norm_seq = np.clip(norm_seq, 0, 1)
        #     normalized_data_list.append(norm_seq.reshape(norm_seq.shape[0], norm_seq.shape[1]*norm_seq.shape[1]))
        for seq in data_list:
            norm_seq = (seq - p5) / (p95 - p5 + 1e-8)
            norm_seq = np.clip(norm_seq, 0, 1)
            normalized_data_list.append(norm_seq)   # ✅ garder la même shape
        return normalized_data_list, p5, p95

    def calculate_norm_images(self, sequence: str, image_sequence: list) -> tuple:
        """
        Computes normalization statistics for MFCC and articulatory contour sequences.

        This function calculates the mean and standard deviation of both MFCC and contour 
        sequences, as well as a moving average for contour normalization.

        Args:
            sequence (str): Identifier for the sequence, used in saved filenames.
            mfcc_sequence (list): List of MFCC feature arrays for each frame.
            image_sequence (list): List of articulatory contour arrays per frame.

        Returns:
            tuple: (std_mfcc, mean_mfcc, std_contour, mean_contour, moving_average)
                - std_mfcc (np.ndarray): Standard deviation of MFCC features.
                - mean_mfcc (np.ndarray): Mean of MFCC features.
                - std_contour (np.ndarray): Standard deviation of articulatory contour features.
                - mean_contour (np.ndarray): Mean of articulatory contour features.
                - moving_average (np.ndarray): Smoothed moving average of contour sequences.
        """
        pad = 30

        contour_sequence_reshaped = [
            traj.reshape(traj.shape[0], traj.shape[1] * traj.shape[2])
            for traj in image_sequence
        ] # [number of sequences, sequence_length, articulator*contours]
        
        
        all_mean_contour = np.array([np.mean(traj, axis=0) for traj in contour_sequence_reshaped])
        all_mean_contour = np.concatenate([np.expand_dims(np.pad(all_mean_contour[:, k], (pad, pad), "symmetric"), 1)
                                    for k in range(all_mean_contour.shape[1])], axis=1)
        moving_average = np.array(
        [np.mean(all_mean_contour[k - pad:k + pad], axis=0) for k in range(pad, len(all_mean_contour) - pad)])
        std_contour = np.mean(np.array([np.std(frame, axis=0) for frame in contour_sequence_reshaped]), axis=0)
        mean_contour = np.mean(np.array([np.mean(frame, axis=0) for frame in contour_sequence_reshaped]), axis=0)
        
        np.save(os.path.join("normalization_values", f"moving_average_image_{sequence}"), moving_average)
        np.save(os.path.join("normalization_values", f"std_image_{sequence}"), std_contour)
        np.save(os.path.join("normalization_values", f"mean_image_{sequence}"), mean_contour)
        
        return std_contour, mean_contour, moving_average
         
    def normalize_labels(self, single_contour: np.ndarray, std_contour: np.ndarray, single_moving_average: np.ndarray) -> np.ndarray:
        """
        Normalizes an articulatory contour using a precomputed moving average and standard deviation.

        Args:
            single_contour (np.ndarray): The contour data for a single sequence.
            std_contour (np.ndarray): Standard deviation of the contour sequence.
            single_moving_average (np.ndarray): Precomputed moving average for contour normalization.

        Returns:
            np.ndarray: The normalized contour.
        """
        epsilon = 1e-8  # Un petit nombre pour éviter la division par zéro
        std_contour = np.maximum(std_contour, epsilon)  # Remplace les zéros par epsilon
        normalized_contour = (single_contour - single_moving_average) / std_contour
        return normalized_contour
        return single_contour
    
    def detect_silence(self, mfcc: np.ndarray, tg: textgrid, sample_rate: int, threshold_ms: int = 200) -> np.ndarray:
        """
        Detects silence in an audio sequence using a TextGrid file.

        This function marks silence regions by analyzing phonetic annotations in the TextGrid file.

        Args:
            mfcc (np.ndarray): MFCC feature sequence.
            tg (TextGrid): Praat TextGrid object containing phonetic annotations.
            sample_rate (int): The sample rate of the audio signal.
            threshold_ms (int, optional): Minimum silence duration to consider, in milliseconds. Defaults to 200ms.

        Returns:
            np.ndarray: A binary mask (1 for speech, -1 for silence).
        """
        # Load the TextGrid
        silence_labels=["#", ""]
        # Get intervals and labels from the specified tier
        sentences = tg[0].intervals
        non_silence = np.ones(len(mfcc))
        # Filter intervals based on silence labels
        hop_length = 160
        i = 0
        for sentence in sentences :
            start = sentence.minTime
            end   = sentence.maxTime
            label = sentence.mark
            if label in silence_labels:
                index_start = int(np.floor(start *sample_rate / hop_length))
                index_end = int(np.floor(end*sample_rate/ hop_length))
                non_silence[index_start:index_end] = -1
        return non_silence
       
    def denormalize_labels(self, single_mfcc: np.ndarray, std_mfcc: np.ndarray, mean_mfcc: np.ndarray, index: int):
        """
        Denormalizes an MFCC sequence using the mean and standard deviation.

        Args:
            single_mfcc (np.ndarray): The normalized MFCC sequence.
            std_mfcc (np.ndarray): Standard deviation of the MFCC features.
            mean_mfcc (np.ndarray): Mean of the MFCC features.

        Returns:
            np.ndarray: The denormalized MFCC sequence.
        """
        denormalized_mfcc = single_mfcc * std_mfcc + mean_mfcc
        for i in range(denormalized_mfcc.shape[0]):
            image = denormalized_mfcc[i].reshape(136, 136) 
            plt.figure()
            plt.imshow(image, cmap='gray')
            plt.tight_layout()
            plt.xlim(0, 136)
            plt.ylim(0, 136)
            plt.gca().invert_yaxis()
            plt.legend()
            output_dir = os.path.join(self.config['data_save'], 'image_test')
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'image_{i}.png'))
            plt.close()
        return
        return denormalized_mfcc
    
    
    
    def normalize_inputs(self, single_mfcc: np.ndarray, std_mfcc: np.ndarray, mean_mfcc: np.ndarray) -> np.ndarray:
        """
        Normalizes MFCC input features using mean and standard deviation.

        Args:
            single_mfcc (np.ndarray): The MFCC feature sequence.
            std_mfcc (np.ndarray): Standard deviation of MFCC features.
            mean_mfcc (np.ndarray): Mean of MFCC features.

        Returns:
            np.ndarray: Normalized MFCC sequence.
        """
        epsilon = 1e-8  # Un petit nombre pour éviter la division par zéro
        std_mfcc = np.maximum(std_mfcc, epsilon)  # Remplace les zéros par epsilon
        normalized_mfcc = (single_mfcc - mean_mfcc) / std_mfcc
        return normalized_mfcc   

    
    def compute_mean_datas(self, concatenated_labels: list) -> torch.Tensor:
        """
        Computes the mean of the concatenated labels (articulatory trajectories) over all frames.

        This function iterates over the provided concatenated labels (articulatory trajectories) 
        to compute the mean of each articulator's features (coordinates).

        Args:
            concatenated_labels (list): List of tensors, where each tensor corresponds to a sequence 
                                        of articulator positions for a given frame.

        Returns:
            torch.Tensor: A tensor representing the mean position of each articulator across all frames.
        """

        
        nbr_coordinates = concatenated_labels[0][0].size(1)
        nbr_articulators = concatenated_labels[0][0].size(0)
        # Initialiser un tenseur pour accumuler les sommes, de dimensions [3, 100]
        total_sum = torch.zeros(nbr_articulators, nbr_coordinates)
        total_frames = 0  # Compteur pour le nombre total de frames

        # Parcourir chaque articulateur dans la liste
        for articulator in concatenated_labels:  # Chaque articulator est de dimension [X, 3, 100]
            total_sum += articulator.sum(dim=0)  # Somme sur la première dimension (X)
            total_frames += articulator.size(0)  # Ajouter le nombre de frames (X)

        # Calculer la moyenne en divisant par le nombre total de frames
        mean_tensor = total_sum / total_frames

        return mean_tensor

    def load_phonemes(self, tg: textgrid, mid_sec: float, all_phonemes: dict) -> tuple:
        """
        Loads the phonemes present in the specified time interval from a TextGrid.

        This function filters phonemes based on the given time `mid_sec` and returns their one-hot 
        encoded labels along with the corresponding phoneme marks.

        Args:
            tg (TextGrid): The TextGrid object containing phonetic annotations.
            mid_sec (float): The time (in seconds) at which to filter phonemes.
            all_phonemes (dict): A dictionary mapping phoneme labels to one-hot encoded vectors.

        Returns:
            tuple: 
                - phonemes_in_interval (list): List of phoneme marks in the interval.
                - phonemes_in_interval_one_hot (list): List of one-hot encoded phoneme labels.
        """

        phoneme_tier_index = int(self.config.get("phoneme_tier_index", 2))
        if phoneme_tier_index >= len(tg):
            phoneme_tier_index = len(tg) - 1
        phonemes = tg[phoneme_tier_index]
        phonemes_to_replace = ['','2h', 'eh', 'ih', 'uh', 'yh']
        short_distance = float('inf')
        best_mark = '#'
        for phoneme in phonemes.intervals:
            if phoneme.maxTime < mid_sec or phoneme.minTime > mid_sec:
                continue  # Skip intervals outside the range
            if phoneme.minTime <= mid_sec and phoneme.maxTime >= mid_sec:
                distance_to_start = abs(mid_sec - phoneme.minTime)
                distance_to_end = abs(mid_sec - phoneme.maxTime)
                # Find the shorter distance
                current_distance = min(distance_to_start, distance_to_end)
                if current_distance < short_distance:
                    short_distance = current_distance
                    mark = phoneme.mark
                    if not mark.strip():
                        mark = '#'
                    if mark in phonemes_to_replace:
                        mark = '#'
                    if '*' in mark:
                        mark = '#'
                    best_mark = mark
        if best_mark not in all_phonemes:
            best_mark = '#'
        return [best_mark], [all_phonemes[best_mark]]
    
    # Abstractmethod     

    @abstractmethod
    def compute_datas(self, sequences: list, audio_sequence: list, tg_sequence: list, contours_sequence: list):
        """
        Abstract method to compute data for a sequence. This method should be implemented 
        in a subclass.

        Args:
            sequences (list): A list of sequences for processing.
            audio_sequence (list): A list of audio features.
            tg_sequence (list): A list of TextGrid objects.
            contours_sequence (list): A list of contour data (articulatory positions).

        Example:
            # This method should be implemented in a subclass to process the specific data.
        """
        pass
    
    
    
    def load_labels(self, image_folder: str, list_image: list) -> list:
        """
        Loads and reconstructs articulatory images for a sequence of frames.

        This function loads `.npy` files for each articulator and constructs the complete sequence 
        of image features based on image numbers. If an image number has a fractional part, it 
        interpolates between the two nearest integer image numbers.

        Args:
            image_folder (str): Path to the folder containing the articulatory images.
            list_image (list): List of image numbers for the sequence. Can contain both integers and floats.

        Returns:
            list: A list of sequences containing the image data for each articulator in the sequence.

        Example:
            all_mris = load_labels("/path/to/images", image_numbers_list)
        """

        all_mris = []
        num_articulators = len(self.config["classes"])
        contour_file_cache = {}

        def load_contour_file(frame_number, articulator):
            image_filename = f"{frame_number:04d}_{articulator}.npy"
            image_path = os.path.join(image_folder, image_filename)
            if image_path not in contour_file_cache:
                contour_file_cache[image_path] = _canonicalize_contour_array(np.load(image_path))
            return contour_file_cache[image_path]

        total_chunks = len(list_image)
        progress_every = int(self.config.get("load_labels_progress_every", 10))
        started_at = time.time()
        for chunk_idx, image_numbers in enumerate(list_image, start=1):
            num_sequences = len(image_numbers)
            sequence_images = np.zeros((num_sequences, num_articulators, 100))
            for seq_idx, image_number in enumerate(image_numbers):
                int_image_number = int(image_number)
                for art_idx, articulator in enumerate(self.config["classes"]):
                    image_data = None
                    if  image_number == int_image_number:
                        image_data = load_contour_file(int_image_number, articulator)

                    elif isinstance(image_number, (float, np.floating)):
                        rounded_down = int(np.floor(image_number))
                        rounded_up = rounded_down + 1
                        image_down = load_contour_file(rounded_down, articulator)
                        image_up = load_contour_file(rounded_up, articulator)
                        # Original
                        image_data = (image_down + image_up) / 2
                    if image_data is not None:
                        sequence_images[seq_idx, art_idx, :] = image_data
            all_mris.append(sequence_images)
            if progress_every and (
                chunk_idx == 1 or chunk_idx == total_chunks or chunk_idx % progress_every == 0
            ):
                elapsed = time.time() - started_at
                print(
                    f"Loaded contour chunk {chunk_idx}/{total_chunks} "
                    f"from {image_folder} in {elapsed:.1f}s "
                    f"({len(contour_file_cache)} contour files cached)",
                    flush=True,
                )
        # Flatten the final array to match [articulator * 100]
        return all_mris

    def load_images(self, image_folder: str, list_image: list) -> list:
        """
        Loads and reconstructs articulatory images for a sequence of frames.

        This function loads `.npy` files for each articulator and constructs the complete sequence 
        of image features based on image numbers. If an image number has a fractional part, it 
        interpolates between the two nearest integer image numbers.

        Args:
            image_folder (str): Path to the folder containing the articulatory images.
            list_image (list): List of image numbers for the sequence. Can contain both integers and floats.

        Returns:
            list: A list of sequences containing the image data for each articulator in the sequence.

        Example:
            all_mris = load_labels("/path/to/images", image_numbers_list)
        """

        all_mris = []

        for image_numbers in list_image:
            num_sequences = len(image_numbers)
            sequence_images = np.zeros((num_sequences, 136,136), dtype=np.float32)
            for seq_idx, image_number in enumerate(image_numbers):
                int_image_number = int(image_number)
                image_data = None
                if  image_number == int_image_number:
                    image_filename  = f"{int_image_number:04d}.npy"
                    image_path = os.path.join(image_folder, image_filename)
                    image_data = np.load(image_path)        

                elif isinstance(image_number, float):
                    rounded_down = int(np.floor(image_number))
                    rounded_up = rounded_down + 1
                    down_filename  = f"{rounded_down:04d}.npy"
                    down_path = os.path.join(image_folder, down_filename)
                    image_down = np.load(down_path)
                    
                    # original 
                    up_filename  = f"{rounded_up:04d}.npy"
                    up_path = os.path.join(image_folder, up_filename)
                    image_up = np.load(up_path)
                    
                    # Original
                    image_data = (image_down + image_up) / 2
                    
                if image_data is not None:
                    sequence_images[seq_idx, :] = image_data
            all_mris.append(sequence_images)
        # Flatten the final array to match [articulator * 100]
        return all_mris

    def load_path(self, image_folder: str, list_image: list) -> list:
        """
        Loads image data for a sequence of articulatory images, extracting relevant information 
        from file paths and storing it in a specific format.

        This function constructs the file paths for each image based on the sequence of image numbers 
        and extracts specific details (such as folder numbers, subfolder names, and file numbers) 
        to create a structured representation for each image in the sequence.

        Args:
            image_folder (str): Path to the folder containing the image files.
            list_image (list): List of image numbers (either integers or floats) representing a sequence 
                            of images for each articulator.

        Returns:
            list: A list containing sequences of image data for each articulator, where each sequence 
                is represented as a 2D numpy array with shape `[num_sequences, 3]`, with the 
                3 columns corresponding to `[folder_number, subfolder_number, file_name_number]`.
        """

        all_mris = []
        articulator = self.config["classes"][0]
        for image_numbers in list_image:
            num_sequences = len(image_numbers)
            sequence_images = np.zeros((num_sequences, 3))
            for seq_idx, image_number in enumerate(image_numbers):
                int_image_number = int(image_number)
                image_data = None
                # Construct the prefix pattern
                if  image_number == int_image_number:
                    image_filename  = f"{int_image_number:04d}_{articulator}.npy"
                    image_path = os.path.join(image_folder, image_filename)
                    parts = image_path.split("/")
                    folder_number = parts[-4]  # "1775" or "P1"
                    subfolder = parts[-3]  # "S19"
                    subfolder_number = subfolder[1:]  # Remove the 'S' from "S19"
                    file_name = os.path.splitext(parts[-1])[0]  # "0190_tongue" without ".npy"
                    file_name_number = file_name.split('_')[0]
                    image_data = np.array([
                        _numeric_id(folder_number),
                        _numeric_id(subfolder_number),
                        float(file_name_number),
                    ], dtype=float)
                    
                    # image_data = np.concatenate((image_data, spm_data, ll_data, ul_data))
                elif isinstance(image_number, float):
                    rounded_down = int(np.floor(image_number))
                    image_filename  = f"{rounded_down:04d}.5_{articulator}.npy"
                    image_path = os.path.join(image_folder, image_filename)
                    parts = image_path.split("/")
                    folder_number = parts[-4]  # "1775" or "P1"
                    subfolder = parts[-3]  # "S19"
                    subfolder_number = subfolder[1:]  # Remove the 'S' from "S19"
                    file_name = os.path.splitext(parts[-1])[0]  # "0190_tongue" without ".npy"
                    file_name_number = file_name.split('_')[0]
                    image_data = np.array([
                        _numeric_id(folder_number),
                        _numeric_id(subfolder_number),
                        float(file_name_number.split(".")[0]) + 0.5,
                    ], dtype=float)
                if image_data is not None:
                    sequence_images[seq_idx,:] = image_data

            all_mris.append(sequence_images)
        return all_mris
    
    
