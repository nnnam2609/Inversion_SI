import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.train.train_single import TrainSingle
from utils.read_yaml import read_datas
from utils.datasets import read_dataset_train
from utils.ddp import ddp_setup
import torch
import mlflow
import mlflow.pytorch
import torch.multiprocessing as mp
from utils.experiment_loader import  prepare_experiment, create_mlflow_experiment, get_run_id_by_name, extract_experiment_and_run_name
import numpy as np


def estimate_memory(obj):
    if isinstance(obj, torch.Tensor):
        return obj.element_size() * obj.nelement()
    elif isinstance(obj, np.ndarray):
        return obj.nbytes
    elif isinstance(obj, list):
        return sum(estimate_memory(x) for x in obj)
    elif isinstance(obj, dict):
        return sum(estimate_memory(v) for v in obj.values())
    else:
        return 0  # on ignore les objets non numériques

def print_batch_memory(batch):
    total_bytes = 0
    print("Détail mémoire (en Mo) :")
    for key, value in batch.items():
        mem_bytes = estimate_memory(value)
        mem_mb = mem_bytes / (1024 ** 2)
        total_bytes += mem_bytes
        print(f"  {key:<20}: {mem_mb:.2f} MB")
    total_mb = total_bytes / (1024 ** 2)
    print(f"---\nMémoire totale estimée : {total_mb:.2f} MB")
    return total_mb

def main(rank: int, world_size: int, config:dict, model_type: str, phonemes_arg: str= None, autoencoder_arg: str= None, checkpoint_path: str= None )-> None:
    """
    Main function to initialize the distributed training setup, load datasets, select the appropriate training class,
    and start the training process using MLflow for logging.

    This function sets up the distributed training environment, loads the dataset, and chooses the correct training 
    class based on the provided arguments. It also handles the training process, with support for resuming from a 
    checkpoint or starting a new experiment. MLflow is used for experiment tracking and logging.

    Args:
        rank (int): The rank of the current process in the distributed setup, used for distributed data parallel (DDP).
        world_size (int): The total number of processes in the distributed setup.
        config (dict): A dictionary containing configuration settings for training, including paths, model settings, etc.
        model_type (str): The type of model to use (e.g., 'baseline').
        phonemes_arg (str, optional): Whether to use phonemes in the model ('yes' or 'no'). Defaults to None.
        autoencoder_arg (str, optional): Whether to use an autoencoder in the model ('yes' or 'no'). Defaults to None.
        checkpoint_path (str, optional): The path to a checkpoint for resuming training. Defaults to None.

    Returns:
        None: This function does not return any value. It trains the model and logs the results using MLflow.

    This function does the following:
    - Initializes the DDP setup using `ddp_setup()`.
    - Loads datasets using `read_dataset_baseline()`.
    - Selects the appropriate training class based on `phonemes_arg` and `autoencoder_arg`.
    - If a checkpoint path is provided, it initializes the trainer with the checkpoint and resumes training.
    - If no checkpoint path is provided, it prepares for a new training run and logs experiment details with MLflow.
    
    Example:
        main(rank=0, world_size=4, config=config_dict, model_type="baseline", phonemes_arg="yes", autoencoder_arg="no", checkpoint_path=None)
    """
    
    ddp_setup(rank, world_size)           
    # Load dataset 
    train_dataloader, validation_dataloader, test_dataloader = read_dataset_train(config, world_size, rank)
    # calculate_pca(train_dataloader, validation_dataloader, test_dataloader, rank)
    # return
    TrainClass = TrainSingle
    
    #Train class 
    trainer = TrainClass(train_dataloader, validation_dataloader, test_dataloader, config, rank, checkpoint_path)
    # for batch in train_dataloader:
    #     print(f"image batch shape: {batch['images'].shape}")
    #     print(f"p5_mris batch shape: {batch['p5'].shape}")
    #     print(f"p95_mris batch shape: {batch['p95'].shape}")
    #     break
    # return
    mlflow.set_tracking_uri(f"{config['data_save']}/mlruns")
    # total = 0
    # for batch in train_dataloader:
    #     total_mb = print_batch_memory(batch)
    #     total += total_mb
    #     # pour ne faire le test que sur un batch
    # print(f"Total memory: {total:.2f} MB")
    # return
    if checkpoint_path:

        if rank == 0:
            run_name = extract_experiment_and_run_name(checkpoint_path)
            run_name_gpu = f"{run_name}_gpu_{rank}"
            run_id = get_run_id_by_name(config['experiment_name'], run_name_gpu)
            config['run_name'] = run_name
            with mlflow.start_run(run_id=run_id) :
                trainer.train(mlflow)
                mlflow.end_run()
        else :
            trainer.train()
    
    else :

        if rank == 0:
            folder_run, run_name, required_data, params  = prepare_experiment(config)
            print("Experiment mlflow : ")
            print(mlflow.get_tracking_uri())
            client = mlflow.tracking.MlflowClient()
            # for exp in client.list_experiments():
            #     print(f"ID: {exp.experiment_id} - Name: {exp.name}")
            with mlflow.start_run(run_name=run_name, experiment_id=config['experiment_id']) :
                mlflow.pytorch.autolog()
                params['run_id'] = mlflow.active_run().info.run_id
                config['run_id'] = params['run_id']
                mlflow.log_params(params)
                mlflow.log_dict(config, "config.yaml")
                mlflow.log_dict(required_data, "datasets.txt")
                trainer.train(mlflow)
                mlflow.end_run()
        else :
            trainer.train()
    
if __name__ == "__main__":
    
    config, phonemes_arg, autoencoder_arg, model_type, checkpoint_path = read_datas()
    if not checkpoint_path:
        create_mlflow_experiment(config)
    world_size = torch.cuda.device_count()
    if world_size <= 0:
        raise RuntimeError("No CUDA GPU is visible to PyTorch; run training inside an OAR GPU allocation with CUDA devices exposed.")
    mp.spawn(main, args=(world_size, config, model_type, phonemes_arg, autoencoder_arg, checkpoint_path), nprocs=world_size)
    #dist.destroy_process_group()
    
