# Labelimg
Tool for processing images


## Clusterimg
Tool for clustering mixed images.

### Main image clustering pipeline flow operates as follows:  

Process starts with folder full of mixed images:  
![whole dataset](images/mixed_images.png)  

#### 1- For each batch:  
* image features are obtained with one of these five methods:
  * SSIM: images itself is used as feature
  * minhash: vector of most obvious corners pixel locations are used as feature. Corners are detected with [cornerHarris from opencv](https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#gac1fc3598018010880e370e2f709b4345)
  * imagehash: perceptual hash of image is used as feature, phash is calculated with [ImageHash library](https://pypi.org/project/ImageHash/)
  * ORB: images ORB features are used as feature. features are calculated with [ORB class from opencv](https://docs.opencv.org/4.x/db/d95/classcv_1_1ORB.html)
  * TM: images itself is used as feature
* similarities are calculated with selected methods similarity calculation:  
  * SSIM: [structural_similarity from scikit-image](https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity) is used
  * minhash: [jaccard from datasketch](http://ekzhu.com/datasketch/minhash.html) is used
  * imagehash: difference between 2 hash values are used.
  * ORB: ratio of good matches over all matches are used
  * TM: minimum matching error is used

  Two image with similarity score bigger than threshold is considered similar, threshold can be given as parameter or interactively selected from set of computations over small data sample. Y axis is threshold value and X axis is the number of expected similar pairs at corresponding threshold value. Approximate number of clustered image is calculated and displayed when a threshold value is hovered:    
  ![th_selection](images/th_select.png)
* similar images are clustered with "If and X and Y image are similar, they are putted into same cluster. If any X and Y image has an image chain X-A-B-...-N-Y that has consecutive pair similarities, they are putted into same cluster." logic
* all clusters + outliers are writed into batch's folder  

A folder full of batch folders and computed image similarities are created after first step:  
![mid result](images/mid_result_folder.png)
Sample batch folder content(each cluster folder has similar found images inside it):  
![mid result](images/batch_folder.png)

#### 2- Merging batch folders  
* first image from all cluster folders at every batch folder is selected as a "representative" for that cluster  
* representative features are obtained
* similarities between these representatives are calculated
* similar representatives are clustered
* cluster folders are merged according to their representatives belonging cluster

All batch folders are merged in one resutl folder after second step:  
![result folder](images/final_result.png)
Sample cluster folder content:  
![mid result](images/result_cluster.png)

### Deep Learning supported image clustering pipeline flow operates as follows:  

#### 1- For whole dataset:  
* If there ins't already a trained model deep learning model is trained with one of the following loss functions:  
  * [PyTorch MSELoss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html)  
  * [PyTorch L1Loss](https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html)  
  * perceptual loss, which is obtained by passing both the deep learning models input and output to another feature extractor model(default is [torchvision VGG19](https://pytorch.org/vision/main/models/generated/torchvision.models.vgg19.html)). Then calculating the mean of features difference.

#### 2- For each batch:  
* image features are obtained from deep learning model
* selected clustering model are created according to given parameters:
  * [Kmeans from scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
  * [Agglomerative Clustering from scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html), which is hierarchy
  * [DBSCAN from scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
  * [Gaussian Mixture from scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)
  * [HDBSCAN from scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html)
* all models are evaluated after clustering image features and best model is selected according to three metrics(maximizing the silhouette and calinski_harabasz scores, minimizing the davies_bouldin score):
  * [silhouette_score from scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
  * [davies_bouldin_score from scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html)
  * [calinski_harabasz_score from scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html)
* best model is used to cluster the images
* all clusters + outliers(there wont be any outlier but outlier folder is kept to match the format of main pipeline) are writed into batch's folder  

#### 3- Merging batch folders  
* first image from all cluster folder at every batch folder is selected as a "representative" for that cluster  
* representative features are obtained
* selected clustering model are created according to given parameters
* all models are evaluated and best model is selected
* representatives clustered with best model
* cluster folders are merged according to their representatives belonging cluster

In deep learning pipeline, main flow is preserved. Only the underlying structure for computations such as image feature extraction(done by feature extractor deep learning models) and similarity calculation(done by clustering models) are changed.  

Main Flow(hard cornered item means a folder in computer, soft cornered item means a variable held storage during run time):  
![main_flow](images/main_flow.png)

### Further possible optimizations:  
* openmp optimizations
* C/C++ optimizations
* CUDA optimizations


## Segmentatimg 
Tool for interactively segmentating images. Main image segmenting pipeline flow operates as follows:  

### 1- image is divided into segments with one of these methods. Segmented image will have labeled segments starting from 1(also edges with value of 0 if any):
* edge: image is divided with edges using opencv's operations
* superpixel: [opencv's superpixel](https://docs.opencv.org/4.x/df/d6c/group__ximgproc__superpixel.html#ga503d462461962668b3bffbf2d7b72038) is used
* kmeans: [opencv's kmeans](https://docs.opencv.org/4.x/d5/d38/group__core__cluster.html#ga9a34dc06c6ec9460e90860f15bcd2f88) is used
* slickmeans: first opencv's superpixel, than opencv's kmeans is applied
* chanvase: [scikit-image's chan vese](https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.chan_vese) is used
* felzenszwalb: [scikit-image's felzenszwalb](https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.felzenszwalb) is used
* quickshift: [scikit-image's quickshift](https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.quickshift) is used
* graph: [opencv's graph segmentation](https://docs.opencv.org/4.x/dd/d19/classcv_1_1ximgproc_1_1segmentation_1_1GraphSegmentation.html) is used
* grabcut: [opencv's grabcut](https://docs.opencv.org/4.x/d3/d47/group__imgproc__segmentation.html#ga909c1dda50efcbeaa3ce126be862b37f) is used. Segmentation is done manually on two window with five annotation types:  
  * Segments window: displays the current segments of image  
  * Annotations window: displays the current annotations on image  
  * rectangle annotation: annotated with mouse middle button, indicates the attention area of the grabcut  
  * foreground and background annotation: annotated with left and right click, indicates the pixels that are definitely foreground or background  
  * possible foreground and background annotation: annotated with ctrl + left and right click, indicates the pixels that may be foreground or background  
    
    Also keyboard inputs are listened for various actions other than painting:  
  * q: quits the segmentation  
  * f: finishes the image segmentation and passes image to interactive painting  
  * r: resets the annotations  
  * space: runs grabcut once(multiple presses are needed for convergence)  
  Annotations of a sample grabcut:  
  ![Annotations of a sample grabcut](images/grabcut/annots.png)  
  selected foreground:  
  ![selected foreground](images/grabcut/mask.png)  

* SAM: [Meta's Segment Anything Model](https://github.com/facebookresearch/segment-anything) is used. Segmentation is done by one of two SAM models: SamAutomaticMaskGenerator(doesnt require any annotation, all processes are automatic) or SamPredictor(prompt must be generated on a window with three annotation types):  
  * Annotations window: displays the current segments of image  
  * rectangle annotation: annotated with mouse middle button, indicates the attention area  
  * foreground and background annotation: annotated with left and right click, indicates the pixels that are definitely foreground or background  

    Also keyboard inputs are listened for various actions other than painting:  
  * q: quits the segmentation  
  * r: resets the annotations  
  * space: ends segmenting and passes prompt to prediction function  
  * f: finishes the segmentation and passes image to interactive painting  
  * z: reverses the last annotation  
    Annotations of a sample SAM:  
    ![Annotations of a sample SAM](images/SAM/annots.png)  
    generated mask:  
    ![generated mask](images/SAM/mask.png)  

### 2- Two window is showed to user, one for color selecting other for painting segments.  
* Color selecting window is used for selecting the segmentation color and displaying the painting mode. There are two paint modes other than default clicking actions. One is for continuously filling and other is unfilling. Both of them are activated and deactivated with double click on related mouse button.  
  Sample image "jet1.jpg":  
  ![Sample image "jet1.jpg"](images/jet1.jpg)  
  Segments for "jet1.jpg" using superpixel(selected method and its parameters should be selected for better segments, this is only for explanatory purposes[black lines around red painted area are edge annotations, originally not included in segments]):  
  ![Segments for "jet1.jpg"](images/normal_segmentation/seg.png)  
  Painted image:  
  ![Painted image](images/normal_segmentation/res.png)  
  Generated Mask "jet1_mask_(R:204,G:0,B:0).png":  
  ![Generated Mask "jet1_mask_(R:204,G:0,B:0).png"](images/normal_segmentation/jet1_mask_(R:204,G:0,B:0).png)  

* Painting are done in segmenting window. Left click fills the segment and right click unfills, Rapid filling and unfilling can be done with continuous modes. Middle button is used to make a cut, a line is cutted between consecutive middle button clicked points and cutted pixels are assigned to be an edge. Also keyboard inputs are listened for various actions other than painting:  
  * q: quits the segmentation  
  * n: goes to next image in folder(no save)  
  * p: goes to previous image in folder(no save)  
  * space: saves the current image masks with "original_image_name\_mask\_(R:value,G:value,B:value).png" format and goes to next image  
  * z: reverses the last action  
  * r: resets the segmentation  
  * d: displays the image segmentation and painted pixels for debug purposes  
  * t: applies template painting. Painting is done with four base image type template, attention(optional), segment and mask(optional). Attention and mask images can generated from template and segment images if not provided.  
    * template: template to look for a match in image  
    * attention: masks that indicates which parts of the templates are considered while looking for a match  
    * segment: paint to put over found match  
    * mask: indicates which pixels on the segment image will painted on the image  
      Sample template(means we will search for a plane in this pose):  
      ![Sample template](images/template/template.png)  
      Sample attention(means that we will ignore the sky and only focus on plane similarity):  
      ![Sample attention](images/template/attention.png)  
      Sample segment(means these pixels will be painted):  
      ![Sample segment](images/template/segment.png)  
      Sample mask(means only white pixels will be painted):  
      ![Sample mask](images/template/mask.png)  