import os
import shutil
import time

from helper_functions import generate_test_dataset

import torch

from DL_Datasets import *
from DL_Models import *
from DL_ModelTrainer import ModelTrainer
from DL_Clusteror import DL_Clusteror

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cpu" # for tesy purposes
    test_path = os.path.join(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0], "test_images")

    for loss in ["perceptual", "mae", "mse"]:
        for method in ["kmeans", "hierarchy", "DBSCAN", "gaussian", "HDBSCAN"]:
            generate_test_dataset(test_path, 4, size=64)

            time.sleep(1) # sleeping for 1 second to cause a model_serial_path change
            mt = ModelTrainer(num_of_epochs=1, lr=0.001, batch_size=2, loss_type=loss, dataset=ImageDataset(test_path),
                            model=PowerOf2s32to128(), verbose=0, device=device)
            dlc = DL_Clusteror(model_trainer=mt, method=method, batch_size=100, overwrite=True, verbose=0)
            try:
                dlc()
            except:
                pass

            time.sleep(1) # sleeping for 1 second to cause a model_serial_path change
            mt = ModelTrainer(num_of_epochs=1, lr=0.001, batch_size=2, loss_type=loss, dataset=ImageDataset(test_path),
                            model=PowerOf2s32to128(), verbose=0, device=device)
            dlc = DL_Clusteror(model_trainer=mt, method=method, batch_size=100, overwrite=True, verbose=0)
            dlc.model_trainer.train()
            try:
                dlc()
            except:
                pass

            old_ckpt = mt.ckpt_path
            mt = ModelTrainer(num_of_epochs=1, lr=0.001, batch_size=2, loss_type=loss, dataset=ImageDataset(test_path),
                            model=PowerOf2s32to128(), verbose=0, device=device, ckpt_path=old_ckpt)
            dlc = DL_Clusteror(model_trainer=mt, method=method, batch_size=100, overwrite=True, verbose=0)
            dlc.model_trainer.train()
            try:
                dlc()
            except:
                pass

            old_ckpt = mt.ckpt_path
            mt = ModelTrainer(num_of_epochs=1, lr=0.001, batch_size=2, loss_type=loss, dataset=ImageDataset(test_path),
                            model=PowerOf2s32to128(), verbose=0, device=device, ckpt_path=old_ckpt)
            dlc = DL_Clusteror(model_trainer=mt, method=method, batch_size=100, overwrite=True, verbose=0)
            try:
                dlc()
            except:
                pass

            shutil.rmtree(test_path)
            shutil.rmtree(test_path + "_clustered")
            print(f"Test passed for {method} with {loss}.")

    for loss in ["perceptual", "mae", "mse"]:
        for method in ["kmeans", "hierarchy", "DBSCAN", "gaussian", "HDBSCAN"]:
            generate_test_dataset(test_path, 4, size=256)

            time.sleep(1) # sleeping for 1 second to cause a model_serial_path change
            mt = ModelTrainer(num_of_epochs=1, lr=0.001, batch_size=2, loss_type=loss, dataset=ImageDataset(test_path),
                            model=PowerOf2s256andAbove(), verbose=0, device=device)
            dlc = DL_Clusteror(model_trainer=mt, method=method, batch_size=100, overwrite=True, verbose=0)
            try:
                dlc()
            except:
                pass

            time.sleep(1) # sleeping for 1 second to cause a model_serial_path change
            mt = ModelTrainer(num_of_epochs=1, lr=0.001, batch_size=2, loss_type=loss, dataset=ImageDataset(test_path),
                            model=PowerOf2s256andAbove(), verbose=0, device=device)
            dlc = DL_Clusteror(model_trainer=mt, method=method, batch_size=100, overwrite=True, verbose=0)
            dlc.model_trainer.train()
            try:
                dlc()
            except:
                pass

            old_ckpt = mt.ckpt_path
            mt = ModelTrainer(num_of_epochs=1, lr=0.001, batch_size=2, loss_type=loss, dataset=ImageDataset(test_path),
                            model=PowerOf2s256andAbove(), verbose=0, device=device, ckpt_path=old_ckpt)
            dlc = DL_Clusteror(model_trainer=mt, method=method, batch_size=100, overwrite=True, verbose=0)
            dlc.model_trainer.train()
            try:
                dlc()
            except:
                pass

            old_ckpt = mt.ckpt_path
            mt = ModelTrainer(num_of_epochs=1, lr=0.001, batch_size=2, loss_type=loss, dataset=ImageDataset(test_path),
                            model=PowerOf2s256andAbove(), verbose=0, device=device, ckpt_path=old_ckpt)
            dlc = DL_Clusteror(model_trainer=mt, method=method, batch_size=100, overwrite=True, verbose=0)
            try:
                dlc()
            except:
                pass

            shutil.rmtree(test_path)
            shutil.rmtree(test_path + "_clustered")
            print(f"Test passed for {method} with {loss}.")

if __name__ == "__main__":
    test()