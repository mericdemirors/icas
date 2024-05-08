# Segmentatimg
---
Tool for segmentating images

TO-ADD
- DL segmenting
- secondary segmenting(first apply kmeans then superpixel to output of kmeans)

TO-TRY
GVF Snake
(https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/AV1011/cartas.pdf)
(https://github.com/t-suzuki/gradient_vector_flow_test/blob/master/gvf.py)
(https://github.com/anlumo/gvf_snakes)

grabcut
(https://www.geeksforgeeks.org/python-foreground-extraction-in-an-image-using-grabcut-algorithm/)
(https://docs.opencv.org/3.4/d8/d83/tutorial_py_grabcut.html)
(https://pyimagesearch.com/2020/07/27/opencv-grabcut-foreground-segmentation-and-extraction/)

selective search
(https://pyimagesearch.com/2020/06/29/opencv-selective-search-for-object-detection/)
(https://www.geeksforgeeks.org/opencv-selective-search-for-object-detection/)
(https://docs.opencv.org/4.x/d5/df0/group__ximgproc__segmentation.html)

skimage.segmentation.felzenszwalb
skimage.segmentation.active_contour
skimage.segmentation.inverse_gaussian_gradient
skimage.segmentation.morphological_chan_vese
skimage.segmentation.morphological_geodesic_active_contour
skimage.segmentation.quickshift
skimage.segmentation.random_walker

TO-TIDY
optimize flood/fill/erode/dilate processes with skimage/opencv functions

TO-FIX