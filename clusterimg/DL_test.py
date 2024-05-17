from DL_Datasets import *
from DL_Models import *
from DL_ModelTrainer import ModelTrainer
from DL_clustering import DL_Clustering

dataset = ImageDataset("/home/mericdemirors/labelimg/clusterimg/heredenene")
model = PowerOf2s256andAbove()
mt = ModelTrainer(num_of_epochs=1, lr=0.001,
                  batch_size=16, loss_type="mse",
                  dataset=dataset, model=model,
                  ckpt_path="/home/mericdemirors/labelimg/clusterimg/heredenene_PowerOf2s256andAbove_mse_05:16:19:01:53/min_loss:0.06544473022222519_epoch:19.pth", verbose=10)

dlc = DL_Clustering(model_trainer=mt, method="kmeans", batch_size=100, overwrite=True, verbose=0)
dlc.model_trainer.train()
dlc()