from planetAI.src.data.utils import PlanetConfig
from planetAI.src.data.map_paster import setup
from ...core.dataclass_argparser import CustomArgumentParser

from dataclasses import dataclass, field, fields, replace
from sys import exit

@dataclass
class BulkSetupConfig:
    return_num_params: bool = field(
        default=True,
        metadata={
            'help': 'Return the number of parameters in the sweep_parameters dictionary.'
        }
    )
    param_num: int = field(
        default=0,
        metadata={
            'help': 'The parameter number to do setup for.'
        }
    )
    force: bool = field(
        default=False,
        metadata={
            'help': 'Whether or not to force setup.'
        }
    )



def get_param_counts(all_params: dict) -> tuple:
    '''
    Returns the number of parameters in the sweep_parameters dictionary.
    Args:
        all_params (dict): The sweep_parameters dictionary.
    Returns:
        tuple: The number of sketch parameters, the number of river parameters.
    '''
    sketch_count = 1
    river_count = 1
    def_planet = PlanetConfig()
    for key in all_params:
        if key in def_planet.__dict__:
            if key in def_planet.sketch_keys():
                sketch_count *= len(all_params[key]['values'])
            if key in def_planet.river_keys():
                river_count *= len(all_params[key]['values'])
    return sketch_count, river_count

def get_params(all_params: dict, param_num: int, sketch: bool) -> dict:
    '''
    Get the parameters for the given parameter number.
    Args:
        all_params (dict): The sweep_parameters dictionary.
        param_num (int): The parameter number to get the parameters for.
        sketch (bool): Whether or not to get the sketch parameters.
    Returns:
        dict: The parameters for the given parameter number. 
    '''
    params = {}

    def_planet = PlanetConfig()

    keys = def_planet.sketch_keys() if sketch else def_planet.river_keys()

    for key in keys:
        if key in all_params:
            values = all_params[key]['values']
            params[key] = values[param_num % len(values)]
            param_num = param_num // len(values)

    
    return params


    


def main():
    '''
    This is a script that iterates through the sweep_parameters dictionary
    and does setup for the necessary images.
    '''
    parser = CustomArgumentParser(
        (PlanetConfig, BulkSetupConfig),
        description='''This is a script that iterates through the sweep_parameters dictionary 
        and does setup for the necessary images.'''
    )

    planet_cfg, bulk_cfg = parser.parse_args_into_dataclasses()

    h = 256*2**planet_cfg.size
    w = 2*h

    all_params = planet_cfg.get_wandb_dict()

    sketch_count, river_count = get_param_counts(all_params)

    param_count = river_count

    if bulk_cfg.return_num_params:
        print(param_count)
        exit(param_count)
    
    if bulk_cfg.param_num >= param_count:
        raise ValueError(f"param_num must be less than {param_count}")
    
    param_num = bulk_cfg.param_num

    sketch_params = get_params(all_params, min(param_num, sketch_count-1), True)
    river_params = get_params(all_params, min(param_num, river_count-1), False)
    params = {**sketch_params, **river_params}
    setup_cfg = replace(planet_cfg, **river_params)
    setup(setup_cfg, force=bulk_cfg.force)

if __name__ == '__main__':

    # num_params = get_param_count(sweep_parameters['512x256'])
    # print(num_params)
    # for i in range(num_params):
    #     print(get_params(sweep_parameters['512x256'], i))

    main()
