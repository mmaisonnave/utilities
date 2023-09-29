"""
Module to handle the configuration yaml configuration file.
"""
import yaml
def get_config() -> dict:
    """
    Function to get the dictionary with all the configuration parameters.
    """
    with open('../config/paths.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config
