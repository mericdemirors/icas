# Segmentatimg 
Tool for interactively segmentating images. Main image segmenting pipeline flow operates as follows:  

## 1- image is divided into segments with one of these methods. Segmented image will have labeled segments starting from 1(also edges with value of 0 if any):
* edge: image is divided with edges using opencv's operations
* superpixel: opencv's superpixel algorithm is used
* kmeans: opencv's kmeans algorithm is used
* slickmeans: first opencv's superpixel, than opencv's kmeans is applied
* chanvase: chan vese from skimage algorithm used
* felzenszwalb: felzenszwalb from skimage algorithm used
* quickshift: quickshift from skimage algorithm used
* graph: opencv's graph segmentation is used
* grabcut: opencv's grabcut algorithm is used. Segmentation is done manually on two window with five annotation types:  
  * Segments window: displays the current segments of image  
  * Annotations window: displays the current annotations on image  
  * rectangle annotation: annotated with mouse middle button, indicates the attention area of the algorithm  
  * foreground and background annotation: annotated with left and right click, indicates the pixels that are definitely foreground or background  
  * possible foreground and background annotation: annotated with ctrl + left and right click, indicates the pixels that may be foreground or background  

  Also keyboard inputs are listened for various actions other than painting:  
  * q: quits the segmentation  
  * f: finishes the image segmentation and passes image to interactive painting  
  * r: resets the annotations  
  * space: runs grabcut algorithm once(multiple presses are needed for convergence)  
* SAM: Meta's Segment Anything Model is used. Segmentation is done by one of two SAM models: SamAutomaticMaskGenerator(doesnt require any annotation, all processes are automatic) or SamPredictor(prompt must be generated on a window with three annotation types):  
  * Annotations window: displays the current segments of image  
  * rectangle annotation: annotated with mouse middle button, indicates the attention area of the algorithm  
  * foreground and background annotation: annotated with left and right click, indicates the pixels that are definitely foreground or background  

  Also keyboard inputs are listened for various actions other than painting:  
  * q: quits the segmentation  
  * r: resets the annotations  
  * space: ends segmenting and passes prompt to prediction function  

## 2- Two window is showed to user, one for color selecting other for painting segments.  
* Color selecting window is used for selecting the segmentation color and displaying the painting mode. There are 2 paint modes other than default clicking actions. One is for continuously filling and other is unfilling. Both of them are activated and deactivated with double click on related mouse button.  
* Painting are done in segmenting window. Left click fills the segment and right click unfills, Rapid filling and unfilling can be done with continuous modes. Middle button is used to make a cut, a line is cutted between consecutive middle button clicked points and cutted pixels are assigned to be new segments with new label. Also keyboard inputs are listened for various actions other than painting:  
  * q: quits the segmentation  
  * n: goes to next image in folder(no save)  
  * p: goes to previous image in folder(no save)  
  * s: saves the current image masks with "original_image_name\_mask\_(R:value,G:value,B:value).png" format and goes to next image  
  * z: reverses the last action  
  * r: resets the segmentation  
  * d: displays the image segmentation and painted pixels for debug purposes  
  * t: applies template painting. Painting is done with 4 base image type template, attention(optional), segment and mask(optional). Attention and mask images can generated from template and segment images if not provided.  
    * template: template to look for a match in image  
    * attention: masks that indicates which parts of the templates are considered while looking for a match  
    * segment: paint to put over found match  
    * mask: indicates which pixels on the segment image will painted on the image  
