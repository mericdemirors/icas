# Labelimg
Tool for clustering and segmenting image datasets. Detailed descriptions for packages and pipelines can be found in [githup repo](https://github.com/mericdemirors/labelimg). Below is just basic usage sytle.

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

### * Segmentimg test
```python
import labelimg.segmentimg.segmentimg_test as st
st()
```

<br/><br/>
<br/><br/>

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

### * Segmentimg
```python
from labelimg.segmentimg.Segmentator import  Segmentator

test_path = "path/to/images"
method = "graph"

sg = Segmentator(test_path, method=method)
sg()
```

<br/><br/>
<br/><br/>

# To user attention:
We insist you to add comments, descriptions and example usage to your contributions  

Below libraries are not installed with labelimg since aim for __labelimg__ is to be lightweight. Advanced usage such as Deep Learning Clustering and SAM segmentation requires below installations:
```python
pip install torch
pip install torchvision
pip install scikit-learn
pip install git+https://github.com/facebookresearch/segment-anything.git
```