import yaml
def get_config():
    with open('../config/paths.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config