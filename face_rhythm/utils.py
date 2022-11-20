
class FR_Module:
    def __init__(self):
        print('hi')
        pass
    

def load_config_file(path_config):
    """
    loads config file

    Args:
        path_config (str): 
            path to config.yaml file

    Returns:
        config (dict): 
            config file as a dictionary

    """
    import yaml
    with open(path_config, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)