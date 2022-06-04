import os
import torch
import random
import argparse

from log import Logger
from utils import load_data
from utils import create_dirs
from utils import save_config
from utils import get_transforms
from utils import confirm_exp_name
from utils import split_train_test_data
from dataset import CatDogDataset
from predictor import CatDogPredictor


def configuration():
    parser = argparse.ArgumentParser(description="Settings of NIR Treatment Response Prediction")
    parser.add_argument('--mode', type=str, default="train", help="train or evaluate")
    parser.add_argument('--train_flag', type=str, default="scratch", help="scratch, pretrained or self-trained, only in train mode")
    parser.add_argument('--exp_name', type=str, default=None, help="experiment name")
    parser.add_argument('--exp_root', type=str, default="./exp", help="experiment root dir")
    parser.add_argument('--data_root', type=str, default="./dataset", help="data root path")
    parser.add_argument('--num_epoch', type=int, default=20, help="train epochs")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.0001, help="weight decay")
    parser.add_argument('--batch_size', type=int, default=16, help="batch size")
    parser.add_argument('--image_size', type=int, default=(256, 256), help="input image size of neural network")
    parser.add_argument('--model_name', type=str, default="cat_dog_classification", help="select the model")
    parser.add_argument('--model_path', type=str, default=None, help="assign model path directly if you want")
    parser.add_argument('--device', type=str, default="cpu", help="choose to use cpu or gpu")
    args = parser.parse_args()
    return args


# parameters
config = configuration()
config.exp_name = confirm_exp_name(config.exp_name, config.exp_root)

device=config.device
train_flag = config.train_flag
batch_size = config.batch_size
image_size = config.image_size
exp_root = config.exp_root
exp_name = config.exp_name
data_root = config.data_root
model_name = config.model_name
exp_path = os.path.join(exp_root, exp_name)
log_dir = os.path.join(exp_path, 'log')
model_dir = os.path.join(exp_path, 'model')
config_dir = os.path.join(exp_path, 'config')
config_path = os.path.join(config_dir, 'config.yaml')

mean = (0.485, 0.456, 0.406)
std =  (0.229, 0.224, 0.225)

create_dirs(exp_name, exp_path, log_dir, model_dir, config_dir)

data, label_dict = load_data(data_root)

train, val, test = split_train_test_data(data)

transform_train, transform_test = get_transforms(image_size, mean, std)



# pytorch data loader
trainset = CatDogDataset(data=train, transform=transform_train)
valset = CatDogDataset(data=val, transform=transform_test)
testset = CatDogDataset(data=test, transform=transform_test)

# set num workers to 0 if your operating system is windows
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                           shuffle=True, num_workers=8)
val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                         shuffle=False, num_workers=8)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=1)

print("load data successfully")

if config.model_path:
    model_path = config.model_path
else:
    model_path = os.path.join(model_dir, f"{model_name}.pth")


if config.mode == "train":
    save_config(config, config_path)
    logger = Logger(log_dir, exp_name)  
    model = CatDogPredictor(flag=train_flag,
                            image_size=image_size,
                            labels=label_dict,
                            device=device,
                            exp_name=exp_name,
                            log_dir=log_dir,
                            logger=logger)

    print("start training")
    model_save_name = os.path.join(model_dir, f"{model_name}.pth")



    model.train_model(train_loader, val_loader, learning_rate=config.learning_rate,
                      num_epoch=config.num_epoch, weight_decay=config.weight_decay,
                      show_batch=10, model_save_path=model_save_name)

    model = CatDogPredictor(model_path=model_save_name,
                            flag="self-trained",
                            image_size=image_size,
                            labels=label_dict,
                            device=device,
                            exp_name=exp_name,
                            log_dir=log_dir)

    metric_test = model.evaluate(test_loader)

    logger.record_test_log(f"test result: " + f"accuracy={metric_test['accuracy']:.3f}, precision={metric_test['precision']:.3f}" \
                           f"recall={metric_test['recall']:.3f}, f1={metric_test['f1']:.3f}, roc auc={metric_test['roc_auc']:.3f}")

elif config.mode == "evaluate":
    model = CatDogPredictor(model_path=model_path,
                            flag="self-trained",
                            image_size=image_size,
                            labels=label_dict,
                            device=device,
                            exp_name=exp_name,
                            log_dir=log_dir)

    model.evaluate(test_loader)

