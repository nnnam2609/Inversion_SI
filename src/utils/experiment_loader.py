import os
import mlflow
from typing import Any
from src.utils.tools import create_folder


def create_mlflow_experiment(config:dict):
    """
    Creates an MLflow experiment.

    Args:
        config (dict): A dictionary containing configuration settings for the experiment. The 'experiment_name' 
                       key should be present, and the function will modify this dictionary by adding 
                       'experiment_id' and 'run_name' keys.

    Returns:
        None: This function modifies the input `config` dictionary in place by adding the 'experiment_id' 
              and 'run_name' keys.

    Example:
        config = {'experiment_name': 'my_experiment'}
        create_mlflow_experiment(config)
        # config will be updated with 'experiment_id' and 'run_name'
    """
    
    experiment_name = config['experiment_name']
    experiment_id = create_experiment(experiment_name, config)
    run_name = create_run_name(config)
    config['experiment_id'] = experiment_id
    config['run_name'] = run_name
    
def create_experiment(experiment_name: str, config:dict) -> str:
    '''
     Creates an MLflow experiment with the given name, or retrieves the experiment ID if it already exists.
     
     Args:
        experiment_name (str): The name of the experiment to be created or retrieved.
        config (dict): A dictionary containing configuration settings. It should include keys like 'data_save' (the 
                       path where MLflow logs will be stored) and 'tag' (the environment tag to be applied to the experiment).

    Returns:
        str: The unique experiment ID of the created or retrieved experiment.

    Example:
        config = {'data_save': '/path/to/mlruns', 'tag': 'dev'}
        experiment_id = create_experiment('experiment_1', config)
    '''
    mlflow.set_tracking_uri(f"{config['data_save']}/mlruns")
    try:
        experiment_id = mlflow.create_experiment(name=experiment_name, tags={"env":config["tag"]}) 
    except:
        print(f"Experiment {experiment_name} already exists.")
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    return experiment_id

def create_run_name(config:str)-> str:
    """
    Generates a unique run name based on the configuration string provided.

    Args:
        config (str): A string used to access experiment-related configuration for generating a unique run name.

    Returns:
        str: A generated run name.

    Example:
        config = 'experiment_1'
        run_name = create_run_name(config)
    """
    run_name = get_name(config)
    return run_name

def get_name(config: dict) -> str:
    """
    Generates a unique experiment name based on the provided configuration and articulator classes.

    This function generates a name for the experiment by creating abbreviations for articulators, selecting the first 
    'n' articulators, and constructing an experiment name from them. The final name is returned as a string.

    Args:
        config (dict): A dictionary containing the configuration, which includes 'classes' (list of articulators) 
                       and 'model' (the model name).

    Returns:
        str: A generated experiment name based on the articulators and the model configuration.

    Example:
        config = {'classes': ['p', 'b', 'm'], 'model': 'my_model'}
        experiment_name = get_name(config)
    """
    # Get articulators from YAML
    articulators = config["classes"]
    # Generate abbreviations
    abbreviations = generate_abbreviations(articulators)
    # Example: Generate experiment names for 1 to 9 articulators
    for num in range(1, len(articulators) + 1):
        selected_articulators = articulators[:num]  # Choose the first 'num' articulators
        experiment_name = generate_experiment_name(selected_articulators, articulators, abbreviations)
    return (f"{config['model']}_{num}_articulators_{experiment_name}")

def generate_abbreviations(articulators: list) -> list:
    """
    Generates abbreviations for a list of articulators by taking the first letter of each word.

    This function takes a list of articulators (which may be compound words, separated by hyphens) and creates an abbreviation 
    for each articulator by taking the first letter of each word in the articulator. The abbreviation is then converted to lowercase.

    Args:
        articulators (list): A list of articulator names, which may contain compound words separated by hyphens.

    Returns:
        list: A list of abbreviations, where each abbreviation is a lowercase string formed by the first letter of each word 
              in the corresponding articulator name.

    Example:
        articulators = ['lip-tongue', 'glottis', 'teeth']
        abbreviations = generate_abbreviations(articulators)
        # abbreviations will be ['lt', 'g', 't']
    """
    abbreviations = []
    for articulator in articulators:
        words = articulator.split('-')
        abbreviation = ''.join(word[0] for word in words)  # Take the first letter of each word
        abbreviations.append(abbreviation.lower())
    return abbreviations

def generate_experiment_name(selected_articulators: list, articulators: list, abbreviations: list) -> str:
    """
    Generates a unique experiment name based on selected articulators and their abbreviations.

    This function takes a list of selected articulators, a list of all articulators, and their corresponding abbreviations. 
    It maps the selected articulators to their respective abbreviations and returns a string with the abbreviations joined by underscores.

    Args:
        selected_articulators (list): A list of selected articulators to be included in the experiment name.
        articulators (list): A list of all possible articulators from which the selected articulators are chosen.
        abbreviations (list): A list of abbreviations corresponding to each articulator in the `articulators` list.

    Returns:
        str: A string representing the experiment name, formed by joining the selected abbreviations with underscores.

    Example:
        selected_articulators = ['p', 'b']
        articulators = ['p', 'b', 'm']
        abbreviations = ['P', 'B', 'M']
        experiment_name = generate_experiment_name(selected_articulators, articulators, abbreviations)
        # experiment_name will be 'P_B'
    """
    selected_abbreviations = [abbreviations[articulators.index(a)] for a in selected_articulators]
    return "_".join(selected_abbreviations)

def get_run_id_by_name(experiment_name: str, run_name: str) -> str:
    """
    Retrieves the run ID for a given run name within a specific experiment.

    This function searches for runs within the specified experiment that match the given run name. If multiple runs 
    match the name, the function returns the ID of the first matching run. If no matching run is found, it returns None.

    Args:
        experiment_name (str): The name of the experiment in which to search for the run.
        run_name (str): The name of the run to search for.

    Returns:
        str or None: The run ID of the first matching run, or None if no run is found.

    Example:
        run_id = get_run_id_by_name('experiment_name', 'run_name')
        # run_id will be the ID of the first matching run, or None if not found.
    """
    # Search for runs with the given name in the specified experiment
    runs = mlflow.search_runs(experiment_ids=[mlflow.get_experiment_by_name(experiment_name).experiment_id],
                                filter_string=f"tags.mlflow.runName = '{run_name}'")
    
    # Check if any runs match the criteria
    if len(runs) == 0:
        print(f"No run found with name '{run_name}' in experiment '{experiment_name}'")
        return None
    elif len(runs) > 1:
        print(f"Multiple runs found with name '{run_name}' in experiment '{experiment_name}'. Returning the first one.")
    
    # Extract the run ID of the first matching run
    run_id = runs.iloc[0]["run_id"]
    return run_id

def prepare_experiment(config: dict) -> tuple:
    """
    Prepares the experiment by setting up necessary directories, saving required data, and returning experiment details.

    This function creates a folder for the experiment and the run, writes out the dataset information to a text file, 
    and returns the paths for the experiment folder, run folder, and relevant experiment data (parameters and required data).

    Args:
        config (dict): A dictionary containing configuration settings. It must include keys such as:
            - 'batch_size', 'n_epochs', 'learning_rate', etc. for model parameters.
            - 'train_sequences', 'valid_sequences', 'test_sequences', 'classes' for dataset information.
            - 'experiment_name' and 'run_name' for the experiment and run identifiers.

    Returns:
        tuple: A tuple containing:
            - folder_run (str): Path to the run folder.
            - run_name_gpu (str): The run name.
            - required_data (dict): Dictionary containing required dataset information.
            - params (dict): Dictionary of model parameters.

    Example:
        config = {
            'batch_size': 32, 'n_epochs': 10, 'learning_rate': 0.001, 'experiment_name': 'exp1',
            'run_name': 'run1', 'train_sequences': {'train_data': 'path_to_train_data'}}
        folder_run, run_name_gpu, required_data, params = prepare_experiment(config)
        # folder_run will be the path to the run folder, run_name_gpu is 'run1', and so on.
    """
    params = {    
    'batch_size': config['batch_size'],
    'n_epochs': config['n_epochs'],
    'patience': config['patience'],
    'learning_rate': config['learning_rate'],
    'weight_decay': config['weight_decay'],
    'loss_function': config['loss_function'],
    'optimizer': config['optimizer'],
    'input_layer': config['input_layer'],
    'hidden_layer': config['hidden_layer'],
    'output_layer': config['output_layer'],
    'num_layers': config['num_layers']
    }
    
    required_data = {
    "train_sequences": config.get("train_sequences", {}),
    "valid_sequences": config.get("valid_sequences", {}),
    "test_sequences": config.get("test_sequences", {}),
    "classes": config.get("classes", []),
    }
    

    
    folder_experiment = create_folder(f"results/{config['experiment_name']}")
    
    run_name_gpu = f"{config['run_name']}"
    folder_run = create_folder(f"{folder_experiment}/{config['run_name']}")
    dataset_file = f'{folder_run}/datasets.txt'
    with open(dataset_file, 'w') as f:
        for section_name, section_data in required_data.items():
            f.write(section_name + ":\n")
            if isinstance(section_data, list):
                for item in section_data:
                    f.write(f"    - {item}\n")
            else:
                for key, value in section_data.items():
                    f.write(f"    {key}:\n")
                    for item in value:
                        f.write(f"        - {item}\n")
    
    return folder_run, run_name_gpu, required_data, params  
    
def extract_experiment_and_run_name(path: str) -> str:
    """
    Extracts the run name from a given path.

    This function splits the provided path into components and extracts the run name, which is assumed to be 
    the third-to-last part of the path.

    Args:
        path (str): The path to the directory or file, typically containing experiment and run details.

    Returns:
        str: The extracted run name from the given path.

    Example:
        path = '/path/to/experiment_name/run_name/checkpoint'
        run_name = extract_experiment_and_run_name(path)
        # run_name will be 'experiment_name'
    """
    # Split the path by '/'
    parts = path.split('/')
    
    # Extract experiment name and run name
    run_name = parts[-3]

    return run_name

