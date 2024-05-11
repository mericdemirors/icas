# Segmentatimg
---
Tool for segmentating images

TO-ADD
- DL segmenting
- secondary segmenting(first apply kmeans then superpixel to output of kmeans)

TO-TRY
grabcut
(https://github.com/symao/InteractiveImageSegmentation?tab=readme-ov-file)
(https://github.com/jasonyzhang/interactive_grabcut)
(https://www.youtube.com/watch?v=kAwxLTDDAwU)

TO-TIDY
optimize flood/fill/erode/dilate processes with skimage/opencv functions

TO-FIX
if grabcut is used, make thread range = 0
optimize grabcut imshows, add comments and add a reverse key
functionalize the repeating codes(get_segments in GrabcutSegmentor)