import pdb

from src.models.deepcoxph import *
from src.models.deeppam import * 
from src.architectures.cnn import * 
from src.architectures.cnn3d import *


models = {'DeepCoxPH': DeepCoxPH,
          'DeepPAM': DeepPAM}

architectures = {'cnn': CNN,
                'cnn3d': CNN3d}


def parse_architecture_config(config):
    """
    """
    deep_params = config.get('model_params').get('deep_params')
    architecture = architectures[deep_params.get('architecture')]
    architecture_instance = architecture(**deep_params.get('architecture_params'))

    return architecture_instance

def parse_model_config(config):
    """
    """
    model = models[config.get('model_params').get('model_name')]
    deep = parse_architecture_config(config)
    model_instance = model(deep=deep, **config.get('model_params').get('structured_params'), **config.get('model_params').get('wide_deep_params'))
    return model_instance


def update_config(config, args):
    """
    """
    for name, value in vars(args).items():
        if value is None:
            continue

        for key in config.keys():
            if config[key].__contains__(name):
                config[key][name] = value

    return config
