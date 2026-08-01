import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler




def ddp_setup(rank: int, world_size: int) -> None:
    """
    Initializes the distributed training setup for PyTorch using NCCL backend.

    This function sets up the environment variables required for distributed training, initializes the process group 
    for communication across distributed processes using the NCCL backend, and assigns the GPU device based on the 
    given rank.

    Args:
        rank (int): The rank of the current process in the distributed setup. This is typically a unique identifier 
                    for each process in the distributed training.
        world_size (int): The total number of processes participating in the distributed training. It represents the 
                          number of GPUs or nodes being used in the training setup.

    Returns:
        None: This function does not return any value. It modifies the environment and sets up the training process 
              for distributed execution.

    Example:
        ddp_setup(rank=0, world_size=4)
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # NCCL is unavailable on native Windows.  Keep the server path unchanged,
    # but use PyTorch's Windows-supported backend for local one-GPU runs.
    backend = "nccl" if os.name != "nt" and dist.is_nccl_available() else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()
