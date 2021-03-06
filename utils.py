import os
import yaml
import random
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

    with open(config_path, 'a') as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False)


def load_data(data_root):
    data = []

    categories = os.listdir(data_root)

    label_dict = {}

    
    for i in range(len(categories)):
        label_dict[i] = categories[i]
        category_path = os.path.join(data_root, categories[i])
        image_names = os.listdir(category_path)
        num_images = len(image_names)
        print(f"class {i} has {num_images} images")
        for image_name in image_names:
            image_path = os.path.join(category_path, image_name)
            data.append((image_path, i))

    # shuffle data
    random.seed(0)
    random.shuffle(data)
    
    return data, label_dict

def split_train_test_data(data):
    n = len(data)
    sep1 = int(0.8*n)
    sep2 = int(0.9*n)
    train = data[:sep1]
    val = data[sep1:sep2]
    test = data[sep2:]

    print("train images: ", len(train))
    print("validation images: ", len(val))
    print("test images: ", len(test))

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
