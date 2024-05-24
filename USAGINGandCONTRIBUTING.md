# Test the whole system pipeline with:

### * Clusterimg test
```python
import labelimg.clusterimg.clusterimg_test as ct
ct()
```

### * Deep Learning Clusterimg test
```python
import labelimg.clusterimg.DL_clusterimg_test as dl_ct
dl_ct()
```

### * Segmentatimg test
```python
import labelimg.segmentatimg.segmentatimg_test as st
st()
```

# Use the labelimg systems with:

### * Clusterimg 
```python
from labelimg.clusterimg.Clusteror import  Clusteror

test_path = "path/to/images"
method = "TM"
batch_size = 500
threshold = 0.5

cl = Clusteror(test_path, method, method, threshold=threshold, overwrite=True)
cl()
```

### * Deep Learning Clusterimg 
```python
from labelimg.clusterimg.DL_ModelTrainer import ModelTrainer
from labelimg.clusterimg.DL_Datasets import ImageDataset
from labelimg.clusterimg.DL_Models import PowerOf2s32to128
from labelimg.clusterimg.DL_Clusteror import DL_Clusteror

device = "cpu"
method = "kmeans"
loss = "mse"
test_path = "path/to/images"

mt = ModelTrainer(num_of_epochs=1, lr=0.001, batch_size=2, loss_type=loss, dataset=ImageDataset(test_path), model=PowerOf2s32to128(), verbose=0, device=device)

dlc = DL_Clusteror(model_trainer=mt, method=method, batch_size=100, overwrite=True, verbose=0)
dlc()
```

### * Segmentatimg
```python
from labelimg.segmentatimg.Segmentator import  Segmentator

test_path = "path/to/images"
method = "graph"

sg = Segmentator(test_path, method=method)
sg()
```

# Contribute to the whole system pipeline with:

### * adding new method to Clusterimg:

* 1- add needed parameters to Clusteror.\_\_init\_\_()
* 2- add methods name to valid_methods in Clusteror.agruamn_check()
* 3- add similarity calculation to helper_functions.similarity_methods()
* 4- if threading is needed, use helper_functions.thread_this() function with format: helper_functions.thread_this(function_to_pass_to_threads, list_of_parameters_to_pass_that_function), it will return the returned values in a list with same order of list_of_parameters_to_pass_that_function
* 5- add method to helper_functions.get_image_features() function


### * adding new deep learning model/dataset or clustering method to Deep Learning Clusterimg

#### deep learning model/dataset:
* 1- you can add your own deep learning model and dataset with attributes and functions explained in Clusterimg README. There wont be and incode additions for that, just pass the your objects to DL_Clusteror.\_\_init\_\_()

#### clustering method:
* 1- add needed parameters to DL_Clusteror.\_\_init\_\_()
* 2- add clustering method to valid_methods in DL_Clusteror.agruamn_check()
* 3- add your models parameter grid search to DL_Clusteror.calculate_grid_search() function.
* 4- add your model initialization to get_models() function inside DL_Clusteror.calculate_grid_search()
* 5- write a fit_predict() function to your model for fitting the passed data and returning the predicted labels for them. Pay attention to match your clustering models fit_predict() output with current models output format: numpy.ndarray of shape (n_samples,)


### * adding new method to Segmentatimg

#### automatic method:
* 1- add needed parameters to Segmentator.\_\_init\_\_()
* 2- add methods name to valid_methods in Segmentator.agruamn_check()
* 3- add your segmentation function above the helper_functions.segment_image() function. Pay attention to match your segmentation functions output with current methods output format: two dimensional numpy.ndarray with shape equal to original images shape[:2]. edges are indicated with value 0 and segments are labeled starting from 1
* 4- add your function to helper_functions.segment_image() function with needed parameters

#### interactive method:
* 1- these methods require user input beforehand to segment an image
* 2- add methods name to valid_methods in Segmentator.agruamn_check()
* 3- we recommend you to create a new class .py file to seperate annotation part with segmentation painting part. Than pass your annotation object to Segmentator.\_\_init\_\_() in a new suitable parameter field
* 4- Than add a new Segmentator.process() function since methods requiring user input doesn't run on threads
* 5- add your segmentation function above the helper_functions.segment_image() function. Pay attention to match your segmentation functions output with current methods output format: two dimensional numpy.ndarray with shape equal to original images shape[:2]. edges are indicated with value 0 and segments are labeled starting from 1
* 6- add your function to helper_functions.segment_image() function with needed parameters

# To user attention:
Below libraries are not installed with labelimg since aim for __labelimg__ is to be lightweight. Advanced usage such as Deep Learning Clustering and SAM segmentation requires below installations:
```python
pip install torch
pip install torchvision
pip install scikit-learn
pip install git+https://github.com/facebookresearch/segment-anything.git
```