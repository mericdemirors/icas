# Clusterimg
Tool for clustering mixed images.

## Main image clustering pipeline flow operates as follows:  

### 1- For each batch:  
* image features are obtained with one of these five methods:
  * SSIM: images itself is used as feature
  * minhash: vector of most obvious corners pixel locations are used as feature. Corners are detected with [cornerHarris from opencv](https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#gac1fc3598018010880e370e2f709b4345)
  * imagehash: perceptual hash of image is used as feature, phash is calculated with [ImageHash library](https://pypi.org/project/ImageHash/)
  * ORB: images ORB features are used as feature. features are calculated with [ORB class from opencv](https://docs.opencv.org/4.x/db/d95/classcv_1_1ORB.html)
  * TM: images itself is used as feature
* similarities intra-batch are calculated with selected methods similarity calculation:  
  * SSIM: [structural_similarity from skimage](https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity) is used
  * minhash: [jaccard from datasketch](http://ekzhu.com/datasketch/minhash.html) is used
  * imagehash: difference between 2 hash values are used.
  * ORB: ratio of good matches over all matches are used
  * TM: minimum matching error is used

  Two image with similarity score bigger than threshold is considered similar, threshold can be given as parameter or interactively selected from set of computations over small data sample. Y axis is threshold value and X axis is the number of expected similar pairs at corresponding threshold value. Approximate number of clustered image is calculated and displayed when a threshold value is hovered:    
  ![th_selection](images/th_select.png)
* similar images are clustered with "If and X and Y image are similar, they are putted into same cluster. If any X and Y image has an image chain X-A-B-...-N-Y that has consecutive pair similarities, they are putted into same cluster." logic
* all clusters + outliers are writed into batch's folder  

### 2- Merging batch folders  
* first image from all cluster folders at every batch folder is selected as a "representative" for that cluster  
* representative features are obtained
* similarities between these representatives are calculated
* similar representatives are clustered
* cluster folders are merged according to their representatives belonging cluster



## Deep Learning supported image clustering pipeline flow operates as follows:  

### 1- For whole dataset:  
* Autoencoder model is trained(if there ins't already a trained model)

### 2- For each batch:  
* image features are obtained
* clustering models are created according to given parameters
* all models are evaluated and best model is selected
* best model is used to cluster the images
* all clusters + outliers(there wont be any outlier but outlier folder is kept to match the format of main pipeline) are writed into batch's folder  

### 3- Merging batch folders  
* first image from all cluster folders at every batch folder is selected as a "representative" for that cluster  
* representative features are obtained
* similarities between these representatives are calculated
* similar representatives are clustered
* cluster folders are merged according to their representatives belonging cluster
* clustering models are created according to given parameters
* all models are evaluated and best model is selected
* best model is used to cluster the representatives
* cluster folders are merged according to their representatives belonging cluster


## Further possible optimizations:  
* openmp optimizations
* C/C++ optimizations
* CUDA optimizations
