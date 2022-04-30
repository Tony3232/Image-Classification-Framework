import os
import yaml
import random
import logging
import pandas as pd
import torchvision.transforms as transforms


def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def confirm_exp_name(exp_name, exp_root):
    # auto assign exp_name if it is not manually assigned
    make_dir(exp_root)

    if exp_name is None:
        exist_exp = os.listdir(exp_root)
        exp_nums = [int(name[3:]) for name in exist_exp if name[3:].isdigit()]

        if len(exp_nums) > 0:
            new_num = max(exp_nums) + 1
        else:
            new_num = 1

        exp_name = "exp" + str(new_num)

    return exp_name


def create_dirs(exp_name, exp_path, log_dir, model_dir, config_dir):
    make_dir(exp_path)
    make_dir(log_dir)
    make_dir(model_dir)
    make_dir(config_dir)


def save_config(config, config_path):
    data = {'mode': config.mode,
            'exp_name': config.exp_name,
            'exp_root': config.exp_root,
            'data_root': config.data_root,
            'num_epoch': config.num_epoch,
            'learning_rate': config.learning_rate,
            'weight_decay': config.weight_decay,
            'batch_size': config.batch_size,
            'image_size': config.image_size,
            'model_name': config.model_name,
            'device': config.device}

    with open(config_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False)


def load_data():
    data = []

    for i in range(1, 1001):
        path = f"dataset/cat/{i:04}.png"
        data.append((path, 0))
        
    for i in range(1, 1001):
        path = f"dataset/dog/{i:04}.png"
        data.append((path, 1))

    # shuffle data
    random.seed(0)
    random.shuffle(data)
    
    return data

def split_train_test_data(data):
    train = data[:1600]
    val = data[1600:1800]
    test = data[1800:]

    return train, val, test
    
def get_transforms(image_size, mean, std):
    transform_train = transforms.Compose(
        [
         transforms.Resize(image_size),
         transforms.RandomHorizontalFlip(),
         transforms.RandomRotation(10),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)
         ])
    transform_test = transforms.Compose(
        [
         transforms.Resize(image_size),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)
         ])

    return transform_train, transform_test


def get_logger(filename, verbosity=1, name=None):
    # logger
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}

    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s")

    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    
    if not logger.handlers:
        fh = logging.FileHandler(filename, "a")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
