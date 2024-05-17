import os
from datetime import datetime

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Subset
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

class ModelTrainer():
    def __init__(self, num_of_epochs, lr, batch_size, loss_type, dataset, model, ckpt_path=None, verbose=0):
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
        self.criterion = self.get_criterion(loss_type, self.verbose-1)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        self.ckpt_path = ckpt_path
        if self.ckpt_path is None:
            self.model_serial_path = f"{self.dataset.root_dir}_{type(self.model).__name__}_{self.loss_type}_{datetime.now().strftime('%m:%d:%H:%M:%S')}"
            os.makedirs(self.model_serial_path)
        else:
            self.model_serial_path = self.ckpt_path.split(os.sep)[-2]

    def get_criterion(self, loss_type="mse", model=None, verbose=0):
        """function to set loss function

        Args:
            loss_type (str, optional): indicates los function. Defaults to "mse".
            model (pytorch model, optional): model to use at 'perceptual' loss type, None results in vgg19. Defaults to None.

        Returns:
            function: loss function
        """
        if loss_type == "mse":
            loss = nn.MSELoss()
        if loss_type == "mae":
            loss = nn.L1Loss()
        if loss_type == "perceptual":
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
        return loss

    def train(self, verbose=0):
        torch.cuda.empty_cache()
        min_loss = None
        old_min_save = None
        early_stop_step = 0
        self.model = self.model.train()
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(self.num_of_epochs):
            for (paths, images) in tqdm(dataloader, leave=False):
                images = images.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(images, outputs)
                
                self.optimizer.zero_grad()
                loss.backward()    
                self.optimizer.step()
            
            if min_loss is None or loss < min_loss:
                early_stop_step = 0
                min_loss = loss
                if old_min_save is not None:
                    os.remove(os.path.join(self.model_serial_path, old_min_save))
                save_name = "min_loss:"+str(min_loss.item()) + "_epoch:" + str(epoch) + ".pth"
                old_min_save = save_name
                torch.save(self.model.state_dict(), os.path.join(self.model_serial_path, save_name))

            print(f"Epoch: {epoch+1} | epoch loss: {loss.item()} | min loss: {min_loss.item()}", flush=True)
            self.scheduler.step()
            
            early_stop_step = early_stop_step + 1
            if early_stop_step == 10:
                print("early stopping...")
                break
        
        torch.cuda.empty_cache()
        self.model.load_state_dict(torch.load(os.path.join(self.model_serial_path, save_name)))

    def get_features(self, start, end, verbose=0):
        """function to get image features

        Returns:
            dict: dictionary of image paths and corresponding features
        """
        self.model = self.model.eval()
        torch.cuda.empty_cache()
        features = dict()
        subset = Subset(self.dataset, list(range(start, end)))
        dataloader = DataLoader(subset, batch_size=self.batch_size, shuffle=False)


        with torch.no_grad():
            for (paths, images) in dataloader:
                images = images.to(self.device)
                
                embeds = self.model.embed(images)
                embeds = embeds.cpu().numpy()
                features = features | {paths[i]:embeds[i] for i in range(len(paths))}

        torch.cuda.empty_cache()
        return features
    
    def __call__(self, verbose=0):
        """call function to capsulate all pipeline in one call

        Returns:
            dict: dictionary of image paths and corresponding features
        """
