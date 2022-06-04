import os
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from torchvision import models
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

torch.backends.cudnn.enabled = False


class CatDogPredictor(object):
    def __init__(self,
                model_path="",
                flag="scratch",
                image_size=(256, 256),
                labels={0: "cat", 1: "dog"},
                exp_name="exp1",
                log_dir="./exp/exp1/log/",
                logger=None,
                device="cuda:0" if torch.cuda.is_available() else "cpu",
                pretrained=True):
        super(CatDogPredictor, self).__init__()
        self.path = model_path
        self.flag = flag
        self.device = torch.device(device)
        self.pretrained = pretrained
        self.image_size = image_size
        self.labels = labels
        self.num_classes = len(labels)
        self.transform = transforms.Compose(
                                            [
                                             transforms.Resize((image_size,image_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406),
                                                                  (0.229, 0.224, 0.225))
                                             ])
        self.log_dir = log_dir
        self.exp_name = exp_name
        self.writer = SummaryWriter(log_dir=self.log_dir)
        if logger:
            self.logger = logger
            self.has_logger = True
        else:
            self.has_logger = False

        self.model = self.load_model(path=self.path, flag=self.flag)


    def load_model(self, path="", flag="self-trained"):
        # flag: pretrained or self-trained or scratch

        if flag == "pretrained":
            model = models.resnet18(pretrained=True)
        else:
            model = models.resnet18()

        model.fc = nn.Linear(model.fc.in_features, self.num_classes)

        if flag == "self-trained":
            model.load_state_dict(torch.load(path, map_location=self.device))
        
        model.to(self.device)
        return model


    def train_model(self, train_loader, val_loader, learning_rate=0.01, num_epoch=3, show_batch=1,
                    weight_decay=0.0001, model_save_path="models/cat_dog_classifier.pth", 
                    is_write=True):

        average_mode = "binary" if self.num_classes == 2 else "macro"

        print(f"average_mode: {average_mode}")
        
        criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # train neural network
        best_f1 = 0
        y_true, y_pred, y_score = [], [], []

        if self.has_logger:
            self.logger.record_train_log("start training!")

        for epoch in range(num_epoch):
            self.model.train()
            
            running_loss = 0
            
            for i, data in enumerate(train_loader):
                # read data
                images, labels = data[0].to(self.device), data[1].to(self.device)
                optimizer.zero_grad()

                # predict
                outputs = self.model(images)
        
                # calculate loss
                loss = criterion(outputs, labels)

                # write tensorboard log
                if is_write:
                    self.visualize_loss(loss, epoch)

                loss.backward()
                optimizer.step()
                _, predicted = torch.max(outputs.data, 1)
                score = F.softmax(outputs.data, dim=1)[:,1]
                
                y_true.extend(list(labels.cpu().numpy()))
                y_pred.extend(list(predicted.cpu().numpy()))
                y_score.extend(list(score.cpu().numpy()))
                
                running_loss += loss.item()

                if i % show_batch == (show_batch-1):    # print every 100 mini-batches
                    y_true = np.array(y_true)
                    y_pred = np.array(y_pred)
                    y_score = np.array(y_score)
                    metrics_train = self.compute_metrics(y_true, y_pred, average_mode=average_mode)

                    print(f"Epoch:[{epoch+1}/{num_epoch}, {i+1}]], loss:{running_loss/show_batch:.3f}, accuracy={metrics_train['accuracy']:.3f},", end=" ")
                    print(f"precision={metrics_train['precision']:.3f}, recall={metrics_train['recall']:.3f}, f1={metrics_train['f1']:.3f}")
                    if self.has_logger:
                        self.logger.record_train_log(f"[Train] Epoch:[{epoch+1}/{num_epoch}, {i+1}], total loss:{running_loss/show_batch:.3f}, " + \
                        f"accuracy={metrics_train['accuracy']:.3f}, precision={metrics_train['precision']:.3f}, " + \
                        f"recall={metrics_train['recall']:.3f}, f1={metrics_train['f1']:.3f}")

                    if is_write:
                        self.visualize_metrics(mode="train", metrics=metrics_train, epoch=epoch)

                    y_true = []
                    y_pred = []
                    y_score = []
                    running_loss = 0

            # evaluate
            metrics_eval = self.evaluate(val_loader, show=True, mode="val")
            if self.has_logger:
                self.logger.record_val_log(f"[Validation] Epoch:[{epoch+1}/{num_epoch}], " + \
                                           f"accuracy={metrics_eval['accuracy']:.3f}, precision={metrics_eval['precision']:.3f}, " + \
                                           f"recall={metrics_eval['recall']:.3f}, f1={metrics_eval['f1']:.3f}")
                                           #f"roc auc={metrics_eval['roc_auc']:.3f}")
            if is_write:
                self.visualize_metrics(mode="val", metrics=metrics_eval, epoch=epoch)

            if metrics_eval["f1"] > best_f1:
                best_f1 = metrics_eval["f1"]
                torch.save(self.model.state_dict(), model_save_path)
                print(f"new best f1: {metrics_eval['f1']}")
                print("model saved")

        if self.has_logger:
            self.logger.record_val_log(f"best validation score: f1={best_f1:.3f}")
            self.logger.record_train_log("finish training!")

        return self.model

    def visualize_loss(self, loss, epoch):
        self.writer.add_scalar("loss/train", loss, epoch)
        self.writer.flush()

    def visualize_metrics(self, mode, metrics, epoch):
        # visualize metrics using tensorboard
        # mode: train/val
        self.writer.add_scalar(f"accuracy/{mode}", metrics['accuracy'], epoch)
        self.writer.add_scalar(f"precision/{mode}", metrics['precision'], epoch)
        self.writer.add_scalar(f"recall/{mode}", metrics['recall'], epoch)
        self.writer.add_scalar(f"f1/{mode}", metrics['f1'], epoch)
        self.writer.flush()

    def compute_metrics(self, y_true, y_pred, y_score=None, average_mode='binary'):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=average_mode, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average_mode, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average_mode, zero_division=0)

        metrics = {"accuracy": accuracy,
                   "precision": precision,
                   "recall": recall,
                   "f1": f1}
        """
        if y_score is not None:
            if self.num_classes == 2:
                metrics["roc_auc"] = roc_auc_score(y_true, y_score)
            else:
                metrics["roc_auc"] = roc_auc_score(y_true, y_score, average=average_mode, multi_class='ovo')
        """
        
        return metrics

    def evaluate(self, test_loader, average_mode="binary", show=True, mode="test"):
        self.model.eval()
        y_true = []
        y_pred = []
        y_score = []

        average_mode = "binary" if self.num_classes == 2 else "macro"
        
        with torch.no_grad():
            for data in test_loader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                score = F.softmax(outputs.data, dim=1)[:,1]
                y_true.extend(list(labels.cpu().numpy()))
                y_pred.extend(list(predicted.cpu().numpy()))
                y_score.extend(list(score.cpu().numpy()))

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_score = np.array(y_score)

        
        metrics = self.compute_metrics(y_true, y_pred, y_score, average_mode=average_mode)
        
        if show:
            print(f"----------------------------------------------------------")
            print(f"{mode} results:")
            #print(f"ROC AUC: {metrics['roc_auc']}")
            print(f"Accuracy: {metrics['accuracy']}")
            print(f"Precision: {metrics['precision']}")
            print(f"Recall: {metrics['recall']}")
            print(f"f1: {metrics['f1']}")
            print(f"----------------------------------------------------------")        
            
        return metrics

    
    def predict(self, image_path):
        self.model.eval()
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img).unsqueeze(0)
        img = img.to(self.device)
        output = self.model(img)
        output = torch.softmax(output, 1)
        _, predicted = torch.max(output.data, 1)
        #p = self.sigmoid(_[0])
        p = _[0]
        result = self.labels[int(predicted[0])]
        
        print(f"predicted result: {result}")
        print(f"probability: {p}")
        
        return p, result
    
