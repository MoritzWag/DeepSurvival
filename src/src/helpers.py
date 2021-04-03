
from src.models.deepcoxph import *
from src.models.baseline import *
from src.architectures.SIM.regressors import * 
from src.architectures.ADNI.regressors import *


models = {'DeepCoxPH': DeepCoxPH,
          'Baseline': Baseline,
          'Linear': Linear}

architectures = {'Classifier3d': Classifier3d,
                'Classifier2d': Classifier2d,
                'ResidualClassifier': ResidualClassifier}


def parse_architecture_config(config):
    """
    """
    deep_params = config.get('model_params').get('deep_params')
    architecture = architectures[deep_params.get('architecture')]
    architecture_instance = architecture(deep_params)

    return architecture_instance 

def parse_model_config(config):
    """
    """
    model = models[config.get('model_params').get('model_name')]
    if config.get('model_params').get('deep_params'):
        deep = parse_architecture_config(config)
        model_instance = model(deep=deep, **config.get('model_params').get('structured_params'), **config.get('model_params').get('wide_deep_params'),
                            **config.get('model_params').get('deep_params'))
    else:
        model_instance = model(**config.get('model_params').get('structured_params'), **config.get('model_params').get('wide_deep_params'))
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

        for key in config['baseline_params'].keys():
            try:
                if config['baseline_params'][key].__contains__(name):
                    config['baseline_params'][key][name] = value
            except:
                pass

    return config

def get_optimizer(model,
                 lr,
                 l2_penalty,
                 optimizer,
                 **kwargs):
    
    decay = []
    no_decay = []

    try:
        for name, param in model.deep.named_parameters():
            if not param.requires_grad:
                continue

            if 'bias' in name or 'bn' in name:
                no_decay.append(param)
            else:
                decay.append(param)
    except:
        pass
    
    for name, param in model.linear.named_parameters():
        if not param.requires_grad:
            continue
        no_decay.append(param)

    param_settings = [
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': decay, 'weight_decay': l2_penalty},
    ]

    try:
        return optimizer(params=param_settings, lr=lr, betas=(0.65, 0.99), **kwargs)
    except:
        return optimizer(params=param_settings, lr=lr, **kwargs)

