import pdb

from src.models.deepcoxph import *
from src.models.deeppam import * 
from src.architectures.cnn import * 


models = {'DeepCoxPH': DeepCoxPH,
          'DeepPAM': DeepPAM}

architectures = {'cnn': CNN}


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


# def parse_model_config(config):
#     """
#     """
#     if 'deep_params' in config['model_params']:
#         deep = models['deep']
#         architecture = parse_architecture_config(config)
#         deep_instance = deep(net=architecture)

#     if 'wide_params' in config['model_params']:
#         wide = models['wide']
#         wide_instance = wide(**config['model_params']['wide_params'])
#     if 'wide_params' in config['model_params'] and 'deep_params' in config['model_params']:
#         wide_deep = models['widedeep']
#         wide_deep_instance = wide_deep(wide=wide_instance, 
#                              deep=deep_instance,
#                              **config.get('model_params').get('wide_deep_params'))

#         return wide_deep_instance
    
#     elif 'deep_params' in config['model_params']:
#         return deep_instance
#     else:
#         return wide_instance


