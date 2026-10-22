import json
import os
import yaml


class _bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"



def cprint(string, p=None):
    if not p:
        print(string)
        return
    pre = f"{_bcolors.ENDC}"

    if "bold" in p.lower():
        pre += _bcolors.BOLD
    elif "underline" in p.lower():
        pre += _bcolors.UNDERLINE
    elif "header" in p.lower():
        pre += _bcolors.HEADER

    if "warning" in p.lower():
        pre += _bcolors.WARNING
    elif "error" in p.lower():
        pre += _bcolors.FAIL
    elif "ok" in p.lower():
        pre += _bcolors.OKGREEN
    elif "info" in p.lower():
        if "blue" in p.lower():
            pre += _bcolors.OKBLUE
        else:
            pre += _bcolors.OKCYAN

    print(f"{pre}{string}{_bcolors.ENDC}")



def load_config(config_filepath):
    try:
        with open(config_filepath, "r") as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError:
        cprint(f"Config file not found! <{config_filepath}>", "error_bold")
        exit(1)



def get_model_path(name, dir="./", **kwargs):
    items = [
        name,
    ]
    
    file_name = "_".join(items)
    extension = "_last.pth"

    path = os.path.join(dir, file_name + extension)
    return path



def print_config(config, logger=None):
    conf_str = json.dumps(config, indent=2)
    if logger:
        logger.info(f"\n{' Config '.join(2*[10*'>>',])}\n{conf_str}\n{28*'>>'}")
    else:
        cprint("Config:", "info_underline")
        print(conf_str)
        print(30 * "~-", "\n")



def get_conf_name(config):
    items = []
    items.append(f'n-{config["model"]["name"]}')
    items.append(f's-{config["dataset"]["input_size"]}')
    items.append(f'b-{config["data_loader"]["train"]["batch_size"]}')
    items.append(f't-{config["diffusion"]["schedule"]["timesteps"]}')
    #     items.append(f'op-{config["training"]["optimizer"]["name"].lower()}')
    #     items.append(f'lr-{config["training"]["optimizer"]["params"]["lr"]}')
    #     items.append(f'ep-{config["training"]["epochs"]}')
    items.append(f'sc-{config["diffusion"]["schedule"]["mode"]}')
    return "_".join(items)
