# Clusteror

## Attributes
* images_folder_path: path to image folder
* method: selected method to use at clustering
* threshold: selected similarity threshold
* batch_size: batch size
* num_of_threads: number of threads to run
* scale: scale of image
* size: size of image
* option: selected process option
* transfer: selected file transfer
* overwrite: permission to overwrite
* chunk_time_threshold: time limit for similarity calculations
* lock: lock for thread accessed variabales
* result_container_folder: path to result folder

## Functions

### * *__init__(self, images_folder_path: str, method: str, batch_size: int, threshold: float=None, num_of_threads: int=2, size: tuple=(0, 0), scale: tuple=(1.0, 1.0), option: str="", transfer: str="copy", overwrite: bool=False, chunk_time_threshold: int=60, verbose: int=0)*
initializing Clusteror object
* images_folder_path: path to image folder
* method: method to use at clustering
  * SSIM: [structural_similarity from scikit-image](https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity) is used
  * minhash: [jaccard from datasketch](http://ekzhu.com/datasketch/minhash.html) is used
  * imagehash: difference between [ImageHash library](https://pypi.org/project/ImageHash/) hashes are used
  * ORB: ratio of good matches at [ORB class from opencv](https://docs.opencv.org/4.x/db/d95/classcv_1_1ORB.html) features over all matches are used
  * TM: minimum error at [template matching from opencv](https://docs.opencv.org/4.x/df/dfb/group__imgproc__object.html#ga586ebfb0a7fb604b35a23d85391329be) is used
* batch_size: batch size at similarity check
* threshold: threshold to consider a pair similar
* num_of_threads: threads to run similarity calculations
* size: size of image to read
* scale: scale of image to read
* option: process option type:
  * merge: merges batch folders
  * dontmerge: clusters the batch folders than terminates
  * "": runs the whole process
* transfer: image transfer type:
  * copy: copies image files to result folder
  * move: moves the image files to result folder
* overwrite: permission to overwrite old clustered result folder
* chunk_time_threshold: time limit for checkpoint saving and early stopping

### * *__str__(self)*
to string method  

returns explanatory string

### * *arguman_check(self, verbose: int=0)*
checks arguman validity. Ensures passed method, option and transfer are valid

### * *interactive_threshold_selection(self, num_of_files: int=1000, verbose: int=0)*
lets user select the threshold from a sample of imagaes
* num_of_files: number of files to include in sample

### * *calculate_batch_similarity(self, batch_idx: int, image_paths: list, verbose: int=0)*
calculates image pair similarities in a batch
* batch_idx: index of batch in data
* image_paths: image paths in batch  

returns dictionary of image pairs and similarities

### * *calculate_representative_similarities(self, representative_paths: list, verbose: int=0)*
calculates representative pair similarities
* representative_paths: representative paths in batch  

returns dictionary of representative pairs and similarities

### * *merge_clusters_by_representatives(self, batch_folder_paths: list, verbose: int=0)*
merges batch clusters according to representative similarities
batch_folder_paths: path to each batch folder

returns list of folders to merge together

### * *create_clusters(self, batch_idx: int, image_files: list, verbose: int=0)*
function to encapsulate clustering process for a batch
batch_idx: index of batch in data
image_files: images in the batch

### * *process(self, verbose: int=0)*
function to encapsulate all clustering pipeline

### * *__call__(self)*
calls process() function

<br/><br/>
<br/><br/>

# DL_Datasets
Dataset to train deep learning models. Custom datasets can be used as long as they contain below functions and attributes with same uses.

## Attributes
* root_dir: path to image folder
* x: image file names
* num_samples: length of dataset

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

<br/><br/>
<br/><br/>

# DL_Models
Deep learning model to train. Custom models can be used as long as they contain below functions and attributes with same uses. Any type of model can be used, autoencoders are recommended for their unsupervised training benefit. Also pretrained feature extractor models as ResNets, GoogLeNets, VGGs and such can be used (without training) to teafure extraction if they are modified or encapulated with another model to have below functions and attributes with same uses. two autoencoder model has been provided for square images with edge length of power of twos

## Attributes
No attribute is required

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

<br/><br/>
<br/><br/>

# DL_ModelTrainer
Deep Learning model trainer object.

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

<br/><br/>
<br/><br/>

# DL_Clusteror

## Attributes
* model_trainer: DL_ModelTrainer object
* method: method to use at clustering
* batch_size: batch size at similarity check
* number_of_clusters: grid search parameter list
* max_iter: grid search parameter list
* DBSCAN_eps: grid search parameter list
* DBSCAN_min_samples: grid search parameter list
* HDBSCAN_min_cluster_size: grid search parameter list
* HDBSCAN_max_cluster_size: grid search parameter list
* option: selected process option
* transfer: selected file transfer
* overwrite: permission to overwrite
* result_container_folder: path to result folder

## Functions

### * *__init__(self, model_trainer, method: str, batch_size:int, number_of_clusters: list=[10], max_iter: list=[200], DBSCAN_eps: list=[0.5], DBSCAN_min_samples: list=[5], HDBSCAN_min_cluster_size: list=[5], HDBSCAN_max_cluster_size: list=[None], option: str="", transfer: str="copy", overwrite: bool=False, verbose: int=0)*
initializing DL_Clusteror object
* model_trainer: DL_ModelTrainer object
* method: clustering method
  * [Kmeans from scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
  * [Agglomerative Clustering from scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html), which is hierarchy
  * [DBSCAN from scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
  * [Gaussian Mixture from scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)
  * [HDBSCAN from scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html)
* batch_size: clustering batch size
* number_of_clusters: parameter to pass at kmeans, hierarchy and gaussian methods for parameter grid search
* max_iter: parameter to pass at kmeans and gaussian for parameter grid search
* DBSCAN_eps: parameter to pass at DBSCAN for parameter grid search
* DBSCAN_min_samples: parameter to pass at DBSCAN for parameter grid search
* HDBSCAN_min_cluster_size: parameter to pass at HDBSCAN for parameter grid search
* HDBSCAN_max_cluster_size: parameter to pass at HDBSCAN for parameter grid search
* option: selected process option
* transfer: selected file transfer
* overwrite: permission to overwrite

### * *__str__(self, verbose: int=0)*
to string method  

returns explanatory string

### * *arguman_check(self, verbose: int=0)*
checks arguman validity. Ensures passed method, option and transfer are valid

### * *get_models(self, verbose: int=0)*
gets clustering models in parameter grid search space

returns initialized models

### * *find_best_model(self, models: list, image_embeds, verbose: int=0)*
find best clustering model
* models: clustering models
* image_embeds: image features to cluster  

returns best model  

all models are evaluated after clustering image features and best model is selected according to three metrics(maximizing the silhouette and calinski_harabasz scores, minimizing the davies_bouldin score):
  * [silhouette_score from scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
  * [davies_bouldin_score from scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html)
  * [calinski_harabasz_score from scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html)
  
  ### * *calculate_batch_clusters(self, start: int, end: int, verbose: int=0)*
calculates clusters in a batch
* start: index of first image in batch
* end: index of last image in batch  

returns calculated clusters

### * *calculate_representative_clusters(self, representative_paths: list, verbose: int=0)*
calculates clusters for representatives
* representative_paths: path to representatives  

returns calculated clusters

### * *merge_clusters_by_representatives(self, batch_folder_paths: list, verbose: int=0)*
merges batch clusters according to representative clusters
batch_folder_paths: path to each batch folder

returns list of folders to merge together

### * *create_clusters(self, batch_idx: int, start: int, end: int, verbose: int=0)*
function to encapsulate clustering process for a batch
batch_idx: index of batch in data
* start: index of first image in batch
* end: index of last image in batch  

### * *process(self, verbose: int=0)*
function to encapsulate all clustering pipeline

### * *__call__(self)*
calls process() function