from DL_Datasets import *
from DL_Models import *
from DL_ModelTrainer import ModelTrainer
from DL_clustering import DL_Clustering

mt = ModelTrainer(num_of_epochs=50, lr=0.001,
                  batch_size=16, loss_type="mse",
                  dataset=ImageDataset("/home/mericdemirors/labelimg/clusterimg/heredenene"),
                  model=PowerOf2s32to128(), verbose=10,
                  ckpt_path="/home/mericdemirors/labelimg/clusterimg/heredenene_PowerOf2s32to128_mse_05:18:10:39:14/min_loss:0.01934790425002575_epoch:14.pth")

dlc = DL_Clustering(model_trainer=mt, method="kmeans", batch_size=100, overwrite=True, verbose=10)
#dlc.model_trainer.train()
dlc()