
from DL_Datasets import *
from DL_Models import *
from DL_ModelTrainer import ModelTrainer

dataset = ImageDataset("here256")
model = PowerOf2s256andAbove()
mt = ModelTrainer(num_of_epochs=50, lr=0.001,
                  batch_size=2, loss_type="mse",
                  dataset=dataset, model=model)

features = features = mt()

print(features)

dataset = ImageDataset("here128")
model = PowerOf2s32to128()
mt = ModelTrainer(num_of_epochs=50, lr=0.001,
                  batch_size=2, loss_type="mse",
                  dataset=dataset, model=model)

features = features = mt()

print(features)