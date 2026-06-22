# project/preprocessing/main_preprocessing.py
from email.mime import image
import os
from cv2 import mean
from sympy import sequence
import torchaudio
import textgrid
import librosa
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import glob
from preprocessing.main_preprocessing import Corpus

class Corpus_contours(Corpus):
    def __init__(self, config, sequences, rank):
        super().__init__(config, sequences, rank)
        (
        self.concatenated_input,
        self.concatenated_labels,
        self.concatenated_frames,
        self.concatenated_phonemes_one_hot,
        self.concatenated_std,
        self.concatenated_mean,
        self.concatenated_images,
        self.concatenated_mean_datas,
        self.concatenated_length_datas,
        self.sequences_length,
        self.concatenated_p5_mris,
        self.concatenated_p95_mris
        ) = self.read_sequence()
        
    def compute_datas(self, sequences, audio_sessions, tg_sessions, contours_sessions, all_phonemes):
        mfccs_serie = None
        contours_serie = None
        path_serie = None
        mris_serie = None
        phonemes_serie_one_hot = None
        sequence_length_serie = []
        std_contours_serie = []
        mean_contours_serie = []
        for index, (audio_session, tg_session, contours_session) in enumerate(
            zip(audio_sessions, tg_sessions, contours_sessions)
        ):
            print(f"Processing session {index + 1}/{len(audio_sessions)}: {audio_session}", flush=True)
            if (self.config['input_type'] == 'mfcc'):
                # Compute MFCCs with specified parameters
                audio_signal, sample_rate = librosa.load(audio_session, sr=None)
                mfcc, window_length_samples, hop_length_samples = self.compute_mfcc(audio_signal, sample_rate)

            elif (self.config['input_type'] == 'cepstre'):
                # Compute linear cepstre
                audio_signal, sample_rate = torchaudio.load(audio_session, normalize=False)
                mfcc, window_length_samples, hop_length_samples = self.compute_cepstre(audio_signal, sample_rate)
            
            elif (self.config['input_type'] == 'wav2vec'):
                waveform, sample_rate = torchaudio.load(audio_session)
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                    waveform = resampler(waveform)
                waveform = waveform.squeeze(0)  # mono
                inputs = self.wav2vec_processor(waveform, sampling_rate=16000, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.wav2vec_model(inputs.input_values.to(self.rank))
                mfcc = outputs.last_hidden_state.squeeze(0).cpu().numpy()  # shape [seq_len, 768]
                # Wav2Vec n’a pas de fenêtre glissante explicite, fixe des valeurs compatibles
                window_length_samples = 400
                hop_length_samples = 320
                sample_rate = 16000
                self.hop_length_ratio = 20
                
            elif (self.config['input_type'] == 'hubert'):
                waveform, sample_rate = torchaudio.load(audio_session)
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                    waveform = resampler(waveform)
                waveform = waveform.squeeze(0)  # mono

                inputs = self.hubert_processor(waveform, sampling_rate=16000, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.hubert_model(inputs.input_values.to(self.rank), output_hidden_states=True)
                    # tu peux choisir le dernier hidden_state (par défaut) ou un autre layer
                    mfcc = outputs.last_hidden_state.squeeze(0).cpu().numpy()  # shape [seq_len, 768]

                # HuBERT a un stride de 20ms (comme Wav2Vec)
                window_length_samples = 400
                hop_length_samples = 320
                sample_rate = 16000
                self.hop_length_ratio = 20

            #duration = librosa.get_duration(y=audio_signal, sr=sample_rate)
            
            tg = textgrid.TextGrid.fromFile(tg_session)
            index_silence = self.detect_silence(mfcc, tg, sample_rate)
            list_mfcc, list_contour, list_phoneme, list_phoneme_one_hot = self.detect_sentences(
                mfcc, tg, all_phonemes, sample_rate, window_length_samples, hop_length_samples, index_silence
            )
            max_chunks = self.config.get("max_chunks_per_session")
            if max_chunks:
                list_mfcc = list_mfcc[:max_chunks]
                list_contour = list_contour[:max_chunks]
                list_phoneme = list_phoneme[:max_chunks]
                list_phoneme_one_hot = list_phoneme_one_hot[:max_chunks]
            print(f"Prepared {len(list_contour)} chunks for contour loading", flush=True)

            images_session = contours_session.replace("inference_contours", "NPY_MR")
            
            contour_loaded = self.load_labels(contours_session, list_contour)
            print(f"Loaded contours for {len(contour_loaded)} chunks", flush=True)
            image_loaded = self.load_images(images_session, list_contour) if self.config.get("use_mri", False) else []
            path_loaded = self.load_path(contours_session, list_contour)
            #image_loaded = []
            if mfccs_serie is None:
                contours_serie = contour_loaded
                mfccs_serie = list_mfcc
                path_serie = path_loaded
                mris_serie = image_loaded
                phonemes_serie_one_hot = list_phoneme_one_hot
            else:
                mfccs_serie += list_mfcc
                path_serie += path_loaded
                phonemes_serie_one_hot += list_phoneme_one_hot
                mris_serie += image_loaded
                contours_serie += contour_loaded

            sequence_length_serie.append(len(contour_loaded))


        std_mfcc, mean_mfcc, std_contour, mean_contour, moving_average = self.calculate_norm(
            sequences, mfccs_serie, contours_serie
        )
        # std_image, mean_image, moving_average_image = self.calculate_norm_images(sequences, mris_serie)
        mri = False
        if mri:
            mris_serie, p5, p95 = self.robust_global_minmax_normalize(mris_serie)
            p5_mris_serie = np.full((len(mris_serie),1), p5)
            p95_mris_serie = np.full((len(mris_serie),1), p95)
        else :
            mris_serie = []
            p5_mris_serie = np.array([])
            p95_mris_serie = np.array([])
        #std_image, mean_image = self.normalize_images(sequences, mris_serie)

        std_contour_np = np.stack(std_contour)  # Shape: (8, 100)
        std_contours_serie = [np.tile(std_contour_np, (1, 1, 1)) for arr in contours_serie]
        mean_contour_np = np.stack(mean_contour)  # Shape: (8, 100)
        mean_contours_serie = [np.tile(mean_contour_np, (1, 1, 1)) for arr in contours_serie]
         
        # std_image_np = np.stack(std_image)  
        # std_mris_serie = [np.tile(std_image_np, (1, 1)) for arr in mris_serie]
    
        # mean_image_np = np.stack(mean_image)  
        # mean_mris_serie = [np.tile(mean_image_np, (1, 1)) for arr in mris_serie]
        std_contour = std_contour[None, :, :]
        mean_contour = mean_contour[None, :, :]
        for index in range(len(mfccs_serie)):    
            #image_denoramlized = self.denormalize_labels(mris_serie[index], std_image, mean_image, index)
            mfccs_serie[index] = self.normalize_inputs(mfccs_serie[index], std_mfcc, mean_mfcc)
            contours_serie[index] = self.normalize_labels(contours_serie[index], std_contour, moving_average[index])
            #contours_serie[index] = self.normalize_labels(contours_serie[index], std_contour, mean_contour)
            #mris_serie[index] = self.normalize_inputs(mris_serie[index].reshape(mris_serie[index].shape[0],-1), std_image, mean_image)
        return (
            mfccs_serie,
            contours_serie,
            path_serie,
            phonemes_serie_one_hot,
            std_contours_serie,
            mean_contours_serie,
            sequence_length_serie,
            mris_serie,
            p5_mris_serie,
            p95_mris_serie,
        )    
    

    


    def detect_sentences(self, mfcc, tg, all_phonemes, sample_rate,window_length_samples, hop_length_samples, index_silence):
        #self.hop_length_ratio = 20
        total_images = 4000
        sentences = tg[0]
        silence_labels=["#", ""]
        sample_rate_ms = sample_rate/1000
        frame_duration = (window_length_samples/ sample_rate) * 1000
        decal_sample = sample_rate_ms * self.skip_ms
        all_sentences = []
        all_sentences_image = []
        all_sentences_phonemes = []
        all_sentences_phonemes_one_hot = []
        array_image = np.zeros(total_images)
        num_mfcc = len(mfcc[0])
        mfcc_final = np.zeros((total_images,num_mfcc))
        for sentence in sentences.intervals:
            start = sentence.minTime
            end   = sentence.maxTime
            label = sentence.mark
            if label not in silence_labels:
                start_sample = start * sample_rate
                end_sample = end * sample_rate
                index_begin = int(np.floor(start_sample/ hop_length_samples))
                index_end = int(np.ceil(end_sample / hop_length_samples))
                list_mfcc = []
                list_image = []
                list_phonemes = []
                list_phonemes_one_hot = []
                actual_num = 0
                for i in range(index_begin,index_end):
                    mid_trame_MFCC = int(i * sample_rate_ms * self.hop_length_ratio + ((frame_duration/2) * sample_rate_ms))
                    num_image = ((mid_trame_MFCC - decal_sample)/ (self.ms_image * sample_rate_ms))
                    rounded_down = int(np.floor(num_image))
                    mean_image = ((((num_image * self.ms_image) * sample_rate_ms) + decal_sample)+((((num_image+1) * self.ms_image)*sample_rate_ms)+ decal_sample))/2
                    begin_sec = i*(hop_length_samples/sample_rate)
                    end_sec = begin_sec+ (window_length_samples/ sample_rate)
                    mid_sec = (begin_sec+end_sec)/2
                    phoneme, phoneme_one_hot = self.load_phonemes(tg, mid_sec, all_phonemes)
                    if (rounded_down>=0 and rounded_down<total_images):
                        mean_image = ((rounded_down * self.ms_image + (rounded_down+1) * self.ms_image)/2)*sample_rate_ms + decal_sample

                        if len(list_mfcc) == self.config['sequence_length'] :
                            all_sentences.append(np.array(list_mfcc))
                            all_sentences_image.append(np.array(list_image))
                            all_sentences_phonemes.append(np.array(list_phonemes))
                            all_sentences_phonemes_one_hot.append(np.array(list_phonemes_one_hot))
                            list_mfcc = []
                            list_image = []
                            list_phonemes = []
                            list_phonemes_one_hot = []
                        elif(index_silence[i]==1) : 
                        #elif(index_silence[i]==1 and phoneme[0] not in silence_labels):
                            list_mfcc.append(np.array(mfcc[i], dtype=np.float32))
                            list_phonemes.append(phoneme)
                            list_phonemes_one_hot.append(phoneme_one_hot)
                            if actual_num == rounded_down:
                                list_image.append(num_image)
                            else : 
                                list_image.append(rounded_down)
                                
                            array_image[rounded_down] = i
                            mfcc_final[rounded_down] = mfcc[i]
                            actual_num = rounded_down
   
                if list_mfcc:
                    all_sentences.append(np.array(list_mfcc))
                    all_sentences_image.append(np.array(list_image))
                    all_sentences_phonemes.append(np.array(list_phonemes))
                    all_sentences_phonemes_one_hot.append(np.array(list_phonemes_one_hot))
        return all_sentences, all_sentences_image, all_sentences_phonemes, all_sentences_phonemes_one_hot
       
