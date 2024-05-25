# Image Clustering and Segmenting: icas
Tool for clustering and segmenting image datasets. Detailed descriptions for packages and pipelines can be found in [githup repo](https://github.com/mericdemirors/icas). Below is just basic usage sytle.

# Test the whole system pipeline with:

### * clusterimg test
```python
import icas.clusterimg.clusterimg_test as ct
ct()
```

### * Deep Learning clusterimg test
```python
import icas.clusterimg.DL_clusterimg_test as dl_ct
dl_ct()
```

### * segmentimg test
```python
import icas.segmentimg.segmentimg_test as st
st()
```

<br/><br/>
<br/><br/>

# Use the icas systems with:

### * clusterimg 
```python
from icas.clusterimg.Clusteror import  Clusteror

test_path = "path/to/images"
method = "TM"
batch_size = 500
threshold = 0.5

cl = Clusteror(test_path, method, method, threshold=threshold, overwrite=True)
cl()
```

### * Deep Learning clusterimg 
```python
from icas.clusterimg.DL_ModelTrainer import ModelTrainer
from icas.clusterimg.DL_Datasets import ImageDataset
from icas.clusterimg.DL_Models import PowerOf2s32to128
from icas.clusterimg.DL_Clusteror import DL_Clusteror

device = "cpu"
method = "kmeans"
loss = "mse"
test_path = "path/to/images"

mt = ModelTrainer(num_of_epochs=1, lr=0.001, batch_size=2, loss_type=loss, dataset=ImageDataset(test_path), model=PowerOf2s32to128(), verbose=0, device=device)

dlc = DL_Clusteror(model_trainer=mt, method=method, batch_size=100, overwrite=True, verbose=0)
dlc()
```

### * segmentimg
```python
from icas.segmentimg.Segmentator import  Segmentator

test_path = "path/to/images"
method = "graph"

sg = Segmentator(test_path, method=method)
sg()
```

<br/><br/>
<br/><br/>

# To user attention:
We insist you to add comments, descriptions and example usage to your contributions  

Below libraries are not installed with icas since aim for __icas__ is to be lightweight. Advanced usage such as Deep Learning Clustering and SAM segmentation requires below installations:
```python
pip install torch
pip install torchvision
pip install scikit-learn
pip install git+https://github.com/facebookresearch/segment-anything.git
```