# Clusteror

## Functions

## Attributes

<br/><br/>
<br/><br/>

# DL_Datasets
Dataset to train deep learning models. Custom datasets can be used as long as they contain below functions and attributes with same uses.

## Functions

### * *__init__(self, root_dir)*
creates a dataset object
* root_dir: path to image folder

### * *__len__(self)*
length of the dataset

returns length of dataset

### * *read_image(self, image_path)*
reads image from given path and creates instance suitable for passing the model
* image_path: path to image  

returns readed image

### * *__getitem__(self, idx)*
gets the item
idx: index of the image

returns image path and readed image

## Attributes
self.root_dir: path to image folder
self.x: image file names
self.num_samples: length of dataset

<br/><br/>
<br/><br/>

# DL_Models
Deep learning model to train. Custom models can be used as long as they contain below functions and attributes with same uses.

## Functions

### *__init__(self)*
creates a deep learning model

### *forward(self, x)*
forward pass function
* x: batch to pass  

returns model output

### *embed(self, x)*
function to extract image features
* x: image to extract features  

returns extracted features

## Attributes

<br/><br/>
<br/><br/>

# DL_ModelTrainer
Deep Learning model trainer object.

## Functions

### * *__init__(self, num_of_epochs: int, lr: float, batch_size: int, loss_type: str, dataset, model, loss_model=None, device: str="cpu", ckpt_path: str=None, verbose: int=0)*
creates a DL_ModelTrainer object
* num_of_epochs: number of epochs to train
* lr: learning rate
* batch_size: training batch size
* loss_type: selection of training loss between three loss:
  * mse: [PyTorch MSELoss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html) 
  * mae: [PyTorch L1Loss](https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html) 
  * perceptual: passing both the deep learning models input and output to another feature extractor model. Then calculating the mean of features difference.

* dataset: dataset to train model on
* model: model to train
* loss_model: model to use at perceptual loss, default is [torchvision VGG19](https://pytorch.org/vision/main/models/generated/torchvision.models.vgg19.html)
* device: device to train on
* ckpt_path: path to model checkpoint

### * *get_criterion(self, loss_type: str="mse", model=None)*
creates loss function
* loss_type: type of loss function
* model: loss model for perceptual loss  

returns loss function

### * *train(self)*
trains model forcefully, overwrites checkpoint path.

### * *get_features(self, start: int, end: int)*
gets features of images in batch
* start: index of first image in batch
* end: index of last image in batch  

returns features

## Attributes
* device: device to train on
* num_of_epochs: number of epochs
* lr: learning rate
* batch_size: batch size
* loss_type: loss type
* dataset: dataset to train on
* model: model to train
* optimizer: optimizer to use
* criterion: criterion to use
* scheduler: scheduler to use
* ckpt_path: checkpoint path to model
* model_serial_path: model signature


<br/><br/>
<br/><br/>

# DL_Clusteror

## Functions

## Attributes

<br/><br/>
<br/><br/>
