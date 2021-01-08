"""
Module of utility functions that are used across
different classes
"""
import os
import yaml



def find_by_key(data, target):
    for key, value in data.items():

        if key == target:
            yield value
        elif isinstance(value, dict):
            yield from find_by_key(value, target)


def get_sub_config(config: dict,sub_list: list) -> dict:
    """
    Take in config and return a dictionary with only key , values
    from sub_list
    :param config:
    :param sub_list:
    :return:
    """
    sub_config = {}
    for key in sub_list:
        res = [r for r in find_by_key(config,key)]
        if len(res) == 0: continue
        assert len(res) <= 1
        sub_config[key] = res[0]

    return sub_config


def load_agent(config: dict) -> object:
    """
    Load agent based on name passed from config file
    :param name: string
    :return: Object
    """
    from .trainer import BanditTrainer, SLTrainer

    name = config["agent"]
    del config["agent"]
    module_name = "janestreet"
    trainer = object
    if "bandit" in name.lower():
        module_name += ".bandits"
        trainer = BanditTrainer
    else:
        module_name += ".sl_agents"
        trainer = SLTrainer

    try:
        mod = __import__(module_name, fromlist=[name])
        klass = getattr(mod, name)

        return klass,config,trainer

    except Exception as e:
        print(f"Class {name} Does not exist")
        raise


def load_config(name: str) -> dict:

    if not os.path.isfile(name):
        try:
            os.chdir("../")
            path = os.getcwd()
            name = path + "/etc/" + name
            assert os.path.isfile(name)
        except Exception as e:
            print(f"Could not find the config file {name}")
            raise

    with open(name) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        return data


def load_sklearn(model: str,config: dict) -> object:
    from sklearn import cluster,decomposition
    cluster_algos = [name for name in dir(cluster) if name[0].isupper()]
    decomposition_algos = [name for name in dir(decomposition) if name[0].isupper()]

    module_name = None
    if model in cluster_algos:
        print(f"Loading {model} Clustering algorithm")
        module_name = "sklearn.cluster"

    elif model in decomposition_algos:
        print(f"Loading {model} Clustering algorithm")
        module_name = "sklearn.decomposition"

    else:
        raise Exception(f"Model {model} is not a known SKlearn model")

    mod = __import__(module_name, fromlist=[model])
    klass = getattr(mod, model)
    return (klass(**config) , module_name)