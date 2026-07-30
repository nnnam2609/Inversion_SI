import os
import sys
#from preprocessing.main00_preprocessing import CustomDataset
#from preprocessing.ema00_preprocessing import EmaDataset
from src.preprocessing.contours_preprocessing import Corpus_contours
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DataLoader, DistributedSampler
from src.train.split_cache import ensure_split_caches, split_cache_path
from src.utils.normalization import (
    TRAINING_SPLIT_CACHE_KEYS,
    load_validated_split_cache_state,
    validate_contour_std_floor,
)


class CachedContourDataset(Dataset):
    def __init__(self, state):
        self.concatenated_input = state['features']
        self.concatenated_labels = state['labels']
        self.concatenated_frames = state['frames']
        self.concatenated_phonemes_one_hot = state['phonemes']
        self.concatenated_std = state['std']
        self.concatenated_mean = state['mean']
        self.concatenated_mean_datas = state['mean_datas']
        self.concatenated_length_datas = state['length_datas']
        self.sequences_length = state['sequences_length']

    def __len__(self):
        return len(self.concatenated_input)

    def __getitem__(self, idx):
        return {
            'features': self.concatenated_input[idx],
            'labels': self.concatenated_labels[idx],
            'frames': self.concatenated_frames[idx],
            'phonemes': self.concatenated_phonemes_one_hot[idx],
            'std': self.concatenated_std[idx],
            'mean': self.concatenated_mean[idx],
            'mean_datas': self.concatenated_mean_datas,
            'length_datas': self.concatenated_length_datas,
            'sequences_length': self.sequences_length[idx],
        }


def _dataset_state(dataset):
    return {
        'features': dataset.concatenated_input,
        'labels': dataset.concatenated_labels,
        'frames': dataset.concatenated_frames,
        'phonemes': dataset.concatenated_phonemes_one_hot,
        'std': dataset.concatenated_std,
        'mean': dataset.concatenated_mean,
        'mean_datas': dataset.concatenated_mean_datas,
        'length_datas': dataset.concatenated_length_datas,
        'sequences_length': dataset.sequences_length,
    }


def _state_from_assembled_payload(payload):
    labels = payload['concatenated_labels']
    std = payload['concatenated_std']
    mean = payload['concatenated_mean']
    features = payload['concatenated_input']
    if labels.dim() == 4 and labels.shape[0] < labels.shape[1]:
        labels = labels.permute(1, 2, 0, 3).contiguous()
    if std.dim() == 3 and std.shape[0] < std.shape[1]:
        std = std.permute(1, 0, 2).contiguous()
    if mean.dim() == 3 and mean.shape[0] < mean.shape[1]:
        mean = mean.permute(1, 0, 2).contiguous()
    if std.dim() == 3:
        std = std.unsqueeze(1)
    if mean.dim() == 3:
        mean = mean.unsqueeze(1)

    return {
        'features': features.float(),
        'labels': labels.float(),
        'frames': torch.zeros((features.shape[0], features.shape[1], 3), dtype=torch.float32),
        'phonemes': payload['concatenated_phonemes'].float(),
        'std': std.float(),
        'mean': mean.float(),
        'mean_datas': payload['concatenated_mean_datas'].float(),
        'length_datas': payload['concatenated_length_datas'],
        'sequences_length': payload['sequences_length'],
    }


def _load_assembled_cache_dataset(config, split_key):
    split_to_file = {
        'train_sequences': 'train.pt',
        'valid_sequences': 'valid.pt',
        'test_sequences': 'test.pt',
    }
    cache_dir = config['assembled_dataset_cache_dir']
    cache_path = os.path.join(cache_dir, split_to_file[split_key])
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Assembled dataset cache is missing: {cache_path}")
    print(f"Loading assembled dataset cache for {split_key}: {cache_path}", flush=True)
    assembled = torch.load(cache_path, map_location='cpu')
    state = _state_from_assembled_payload(assembled['payload'])
    validate_contour_std_floor(state, config, cache_path)
    return CachedContourDataset(state)


def _load_or_build_contour_dataset(config, split_key, rank, world_size):
    if config.get('assembled_dataset_cache_dir'):
        return _load_assembled_cache_dataset(config, split_key)

    if config.get('session_cache_dir') and config.get('split_cache_dir'):
        if not config.get('_split_cache_ready', False):
            if rank == 0:
                ensure_split_caches(config)
            if world_size > 1 and dist.is_initialized():
                dist.barrier()
            config['_split_cache_ready'] = True
        cache_path = split_cache_path(config, split_key)
        print(f"Loading split cache for {split_key}: {cache_path}", flush=True)
        state, _floor_summary = load_validated_split_cache_state(
            cache_path,
            config,
            required_keys=TRAINING_SPLIT_CACHE_KEYS,
        )
        return CachedContourDataset(state)

    if not config.get('cache_dataset', False):
        return Corpus_contours(config, split_key, rank)

    cache_dir = config.get(
        'dataset_cache_dir',
        os.path.join(config['data_save'], 'cache', config['folder_save'], 'datasets'),
    )
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'{split_key}.pt')
    rebuild = config.get('rebuild_dataset_cache', False)

    if rank == 0 and (rebuild or not os.path.exists(cache_path)):
        print(f"Building dataset cache for {split_key}: {cache_path}", flush=True)
        dataset = Corpus_contours(config, split_key, rank)
        torch.save(_dataset_state(dataset), cache_path)
        print(f"Saved dataset cache for {split_key}: {cache_path}", flush=True)

    if world_size > 1 and dist.is_initialized():
        dist.barrier()

    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Dataset cache was not created: {cache_path}")

    print(f"Loading dataset cache for {split_key}: {cache_path}", flush=True)
    state, _floor_summary = load_validated_split_cache_state(
        cache_path,
        config,
        required_keys=TRAINING_SPLIT_CACHE_KEYS,
    )
    return CachedContourDataset(state)


def _dataloader_kwargs(config):
    num_workers = int(config.get('num_workers', 0))
    kwargs = {
        'num_workers': num_workers,
        'pin_memory': bool(config.get('pin_memory', False)),
    }
    if num_workers > 0:
        kwargs['persistent_workers'] = bool(config.get('persistent_workers', False))
        if 'prefetch_factor' in config:
            kwargs['prefetch_factor'] = int(config['prefetch_factor'])
    return kwargs


def read_dataset_train(config: dict, world_size: int, rank: int):
    """
    Reads and prepares the dataset for training, validation, and testing in a distributed setup.

    This function loads the training, validation, and testing datasets from the `Corpus_contours` class based on 
    the provided configuration, and prepares them using distributed sampling. It returns the corresponding 
    DataLoader objects for each dataset, ensuring that data is correctly distributed across multiple processes.

    Args:
        config (dict): A dictionary containing configuration settings. It must include keys such as 'batch_size' 
                       and dataset-related parameters.
        world_size (int): The total number of processes participating in the distributed setup (number of GPUs or nodes).
        rank (int): The rank of the current process in the distributed setup. Each process handles a subset of the dataset.

    Returns:
        tuple: A tuple containing three DataLoader objects:
            - train_dataloader: The DataLoader for the training dataset.
            - validation_dataloader: The DataLoader for the validation dataset.
            - test_dataloader: The DataLoader for the test dataset.

    Example:
        config = {'batch_size': 32, 'train_sequences': 'path_to_train_data', 'valid_sequences': 'path_to_valid_data', 'test_sequences': 'path_to_test_data'}
        train_dl, val_dl, test_dl = read_dataset_baseline(config, world_size=4, rank=0)
    """
        
    train_sequences = 'train_sequences'
    valid_sequences = 'valid_sequences'
    test_sequences = 'test_sequences'
    
    
    train_dataset = _load_or_build_contour_dataset(config, train_sequences, rank, world_size)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    loader_kwargs = _dataloader_kwargs(config)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        sampler=train_sampler,
        **loader_kwargs,
    )
    
    
    validation_dataset = _load_or_build_contour_dataset(config, valid_sequences, rank, world_size)
    validation_sampler = DistributedSampler(validation_dataset, num_replicas=world_size, rank=rank)
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        sampler=validation_sampler,
        **loader_kwargs,
    )
    
    test_dataset = _load_or_build_contour_dataset(config, test_sequences, rank, world_size)
    batch_test = len(test_dataset)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_test,
        shuffle=False,
        sampler=test_sampler,
        **loader_kwargs,
    )
    
    return train_dataloader, validation_dataloader, test_dataloader
