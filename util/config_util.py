import collections.abc
import yaml

def deep_merge(base, overrides):
    """
    Merge other config file with default config

    Parameters
    ----------
    base (dict) : Default config ./configs/default.yaml
    overrides (dict) : Config file read in by yaml, see ./configs/default.yaml for a more in-depth understanding
    
    Returns
    ----------
    base (dict) : Returned overwritten default config

    """
    for key, value in overrides.items():
        if (
            (isinstance(value, collections.abc.Mapping)) and 
            (key in base) and 
            (isinstance(base[key], collections.abc.Mapping))
        ):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base
