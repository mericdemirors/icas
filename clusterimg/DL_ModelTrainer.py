import os
import datetime

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Subset

from helper_functions import print_verbose
from helper_exceptions import *

class ModelTrainer():
    def __init__(self, num_of_epochs: int, lr: float, batch_size: int, loss_type: str, dataset, model, ckpt_path: str=None, verbose: int=0):
        """class to capsulate pytorch model and dataset

        Args:
            num_of_epochs (int): number of epochs for model training
            lr (float): learning rate
            batch_size (int): batch size for dataloader
            loss_type (str): loss function type to pass to get_criterion()
            dataset (pytorch dataset): pytorch dataset
            model (pytorch model): pytorch model
            ckpt_path (str, optional): path to load model checkpoint, None means model is not trained. Defaults to None.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_of_epochs = num_of_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.loss_type = loss_type
        self.dataset = dataset
        self.verbose = verbose

        self.model = model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = self.get_criterion(loss_type)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        
        self.ckpt_path = ckpt_path
        if self.ckpt_path is None:
            # if checkpoint is not given, create a new serial number
            self.model_serial_path = f"{self.dataset.root_dir}_{type(self.model).__name__}_{self.loss_type}_{datetime.datetime.now().strftime('%m:%d:%H:%M:%S')}"
            os.makedirs(self.model_serial_path)
        else:
            # else copy the given serial number
            self.model_serial_path = os.path.abspath(os.path.split(self.ckpt_path)[0])

    def get_criterion(self, loss_type: str="mse", model=None):
        """function to set loss function

        Args:
            loss_type (str, optional): indicates los function. Defaults to "mse".
            model (pytorch model, optional): model to use at 'perceptual' loss type, None results in vgg19. Defaults to None.

        Returns:
            function: loss function
        """
        if loss_type == "mse":
            loss = nn.MSELoss()
        elif loss_type == "mae":
            loss = nn.L1Loss()
        elif loss_type == "perceptual":
            if model is None:
                model = models.vgg19(pretrained=True)
            
            model = model.eval()

            def perceptual_loss(x, y, model=model):
                """perceptual loss function, calculates mean absolute difference between 
                2 tensors feature outputs from a model

                Args:
                    x (torch.tensor): first tensor
                    y (torch.tensor): second tensor
                    model (pytorch model, optional): model to pass parameter tensors None results in vgg19. Defaults to model.

                Returns:
                    torch.tensor: loss
                """
                x_out = model(x)
                y_out = model(y)
                return (x_out-y_out).mean()
            loss = perceptual_loss
        else:
            raise(InvalidLossException("Invalid loss type: " + loss_type))
        return loss

    def train(self):
        """trains model with dataset
        """
        torch.cuda.empty_cache()
        min_loss = None
        old_min_save = None
        early_stop_step = 0
        self.model = self.model.train()
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(self.num_of_epochs):
            for (paths, images) in tqdm(dataloader, desc=f"Training {self.model_serial_path}", leave=False):
                images = images.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(images, outputs)
                
                self.optimizer.zero_grad()
                loss.backward()    
                self.optimizer.step()
            
            # save the lowest loss each batch
            if min_loss is None or loss < min_loss:
                early_stop_step = 0
                min_loss = loss
                if old_min_save is not None:
                    os.remove(os.path.join(self.model_serial_path, old_min_save))
                save_name = "min_loss:"+str(min_loss.item()) + "_epoch:" + str(epoch) + ".pth"
                old_min_save = save_name
                torch.save(self.model.state_dict(), os.path.join(self.model_serial_path, save_name))

            print_verbose("v", f"Epoch: {epoch+1} | epoch loss: {loss.item()} | min loss: {min_loss.item()}", verbose=self.verbose-1)
            self.scheduler.step()
            
            early_stop_step = early_stop_step + 1
            if early_stop_step == 10:
                print_verbose("v", "early stopping", verbose=self.verbose-1)
                break
        
        torch.cuda.empty_cache()
        # load the last saved checkpoint
        self.model.load_state_dict(torch.load(os.path.join(self.model_serial_path, save_name)))
        self.ckpt_path = os.path.join(self.model_serial_path, save_name)

    def get_features(self, start: int, end: int):
        """function to get image features

        Args:
            start (int): index of first image in batch
            end (int): index of last image in batch

        Returns:
            dict: dictionary of image paths and corresponding features
        """
        self.model = self.model.eval()
        torch.cuda.empty_cache()
        features = dict()
        subset = Subset(self.dataset, list(range(start, end)))
        dataloader = DataLoader(subset, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():
            for (paths, images) in tqdm(dataloader, desc=f"Getting image features from {start} to {end}", leave=False):
                images = images.to(self.device)
                
                embeds = self.model.embed(images)
                embeds = embeds.cpu().numpy()
                features = features | {paths[i]:embeds[i] for i in range(len(paths))}

        torch.cuda.empty_cache()
        return features