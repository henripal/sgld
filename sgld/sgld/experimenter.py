import yaml
import os

def get_params(config_name, filename = 'config.yaml'):
    """
    gets parameters corresponding to config_name in the file provided
    returns parameters, makes a directory for the experiment and returns both
    """

    try:
        with open(filename, 'r') as f:
            params = yaml.load(f)
            try:
                parameters = params[config_name]
            except KeyError:
                print("Params {} not found in config.yaml file".format(config_name))
    except OSError:
        print("config.yaml not  found.")

    try:
        i = 0 
        while os.path.exists(config_name + '_' + str(i)):
            i += 1
        directory = config_name + '_' + str(i)
        os.makedirs(directory)
    except OSError:
        print('unable to create directory')

    return parameters, directory
