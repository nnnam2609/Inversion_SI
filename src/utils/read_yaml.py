import yaml
import random
import argparse

def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1):
    """
    Splits the dataset into training, validation, and test sets based on provided ratios.

    Args:
        dataset (dict): The dataset containing sequences.
        train_ratio (float): The ratio of data to be used for training.
        val_ratio (float): The ratio of data to be used for validation.

    Returns:
        dict: A dictionary containing the split datasets.
    """
    datas = list(dataset['all_sequences'].items())
    all_datas = {}
    train_sessions = {}
    val_sessions = {}
    test_sessions = {}
    
    for session, sequences in datas:
        # Shuffle the sequences for randomness
        random.shuffle(sequences)
        total_length = len(sequences)
        train_length = int(total_length * train_ratio)
        val_length = int(total_length * val_ratio)
        
        # Split the sequences into training, validation, and test sets
        train_set = sequences[:train_length]
        val_set = sequences[train_length:train_length + val_length]
        test_set = sequences[train_length + val_length:]
        
        # Add the split sets to their respective dictionaries
        train_sessions[session] = train_set
        val_sessions[session] = val_set
        test_sessions[session] = test_set
        
    all_datas['train_sequences'] = train_sessions
    all_datas['valid_sequences'] = val_sessions
    all_datas['test_sequences'] = test_sessions
    
    return all_datas

def split_dataset_old(dataset, train_ratio=0.8, val_ratio=0.1):

    datas = list(dataset['all_sequences'].items())
    all_datas = {}
    train_sessions = {}
    val_sessions = {}
    test_sessions = {}
    for data in datas:
        session = data[0]
        sequences = data[1]
        random.shuffle(sequences)
        total_length = len(sequences)
        train_length = int(total_length * train_ratio)
        val_length = int(total_length * val_ratio)
        train_set = sequences[:train_length]
        val_set = sequences[train_length:train_length+val_length]
        test_set = sequences[train_length+val_length:]
        session_with_quotes = f'{session}'
        train_sessions[session_with_quotes] = train_set
        val_sessions[session_with_quotes] = val_set
        test_sessions[session_with_quotes] = test_set
        
    all_datas['train_sequences'] = train_sessions
    all_datas['valid_sequences'] = val_sessions
    all_datas['test_sequences'] = test_sessions
    return all_datas

def save_merged_dataset(dataset, output_file):
    """
    Saves the dataset to a YAML file, merging it with existing data if present.

    Args:
        dataset (dict): The dataset to be saved.
        output_file (str): The file path for the output YAML file.
    """
    existing_data = read_yaml(output_file)
    existing_data.update(dataset)
    
    with open(output_file, 'w') as file:
        yaml.dump(existing_data, file, default_flow_style=False)

def read_yaml(file_path):
    """
    Reads a YAML file and returns its contents.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        dict: The contents of the YAML file.
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def read_datas():
    """
    Reads configuration and checkpoint file paths from command line arguments, processes datasets if needed,
    and returns the configuration and checkpoint paths along with model-related parameters.

    This function uses argparse to parse command-line arguments for various options, including paths to configuration 
    and checkpoint files, as well as additional options like model type, phoneme usage, and autoencoder usage.
    If the 'generate_config' argument is provided, it processes datasets and generates a merged dataset from the default 
    configuration file before saving it.

    Arguments:
        --config (str): Path to the configuration YAML file.
        --checkpoint (str, optional): Path to the checkpoint file, or None if not provided.
        --generate_config (str, optional): If provided, generates a new configuration file based on the default configuration.
        --model_type (str): Specifies the model type, default is 'baseline'.
        --phonemes (str): Specifies whether to use phonemes ('yes' or 'no'), default is 'no'.
        --autoencoder (str): Specifies whether to use an autoencoder ('yes' or 'no'), default is 'no'.
        
    Returns:
        config (dict): The configuration data parsed from the YAML file specified in the --config argument.
        phonemes_arg (str): The value of the --phonemes argument, indicating whether phonemes are used ('yes' or 'no').
        autoencoder_arg (str): The value of the --autoencoder argument, indicating whether autoencoder is used ('yes' or 'no').
        model_type (str): The value of the --model_type argument, indicating the model type to be used (default: 'baseline').
        checkpoint_path (str or NoneType): The path to the checkpoint file, if provided, or None if not specified.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Read configuration and checkpoint file")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the checkpoint file (optional)")
    parser.add_argument("--generate_config", type=str, default=None, help="Generate a config file from the config_default and save it")
    
    # Add a new argument for model type with a default value
    parser.add_argument("--model_type", type=str, choices=['baseline'], default='baseline', help="Type of the model to use (default: baseline)")
    parser.add_argument("--phonemes", type=str, choices=['no', 'yes'], default='no', help="Use phonemes or not (default: no)")
    parser.add_argument("--autoencoder", type=str, choices=['no', 'yes'], default='no', help="Use Autoencoder or not (default: no)")
    args = parser.parse_args()

    config_path = args.config
    checkpoint_path = args.checkpoint
    generate_config = args.generate_config
    model_type = args.model_type
    phonemes_arg = args.phonemes
    autoencoder_arg = args.autoencoder
    
    if generate_config:
        # If generate_config is provided, read the dataset, split it, and save the merged dataset
        datasets = read_yaml(generate_config)
        all_data = split_dataset(datasets)
        save_merged_dataset(all_data, config_path)
    
    # Read and parse the YAML configuration file
    config = read_yaml(config_path)
    return config, phonemes_arg, autoencoder_arg, model_type, checkpoint_path
