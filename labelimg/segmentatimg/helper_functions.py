import time

import cv2
import numpy as np
from skimage.morphology import flood_fill, flood
from skimage.segmentation import chan_vese, felzenszwalb, quickshift

from .GrabcutSegmentator import GrabcutSegmentator
from .helper_exceptions import *

# edge segmentation
def edge_segmentation(image_path:str, edge_th:int, bilateral_d:int, sigmaColor:int, sigmaSpace:int, templateWindowSize:int,
                      searchWindowSize:int, h:int, hColor:int, verbose:int=0):
    """edge segmentation

    Args:
        image_path (str): path to image to segment
        edge_th (int, optional): threshold to consider a pixel as edge
        bilateral_d (int, optional): window size for cv2.bilatera
        sigmaColor (int, optional): color strength for cv2.bilateral
        sigmaSpace (int, optional): distance strength for cv2.bilateral
        templateWindowSize (int, optional): window size for cv2.fastNlMeansDenoisingColore
        searchWindowSize (int, optional): window size for cv2.fastNlMeansDenoisingColored
        h (int, optional): noise remove strenght for cv2.fastNlMeansDenoisingColored
        hColor (int, optional): color noise remove strenght for cv2.fastNlMeansDenoisingColored
        verbose (int, optional): verbose level. Defaults to 0.

    Returns:
        numpy.ndarray: segmented image
    """
    image = cv2.imread(image_path)
    bilateral = cv2.bilateralFilter(image, d=bilateral_d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    preprocessed = cv2.fastNlMeansDenoisingColored(bilateral, templateWindowSize=templateWindowSize, searchWindowSize=searchWindowSize, h=h, hColor=hColor)
    
    gradient_x = cv2.Sobel(preprocessed, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(preprocessed, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.int16)
    edge_image = (gradient_magnitude[:,:,0] + gradient_magnitude[:,:,1] + gradient_magnitude[:,:,2])/3

    edge_image[edge_image > edge_th] = 255
    edge_image[edge_image < edge_th] = 0

    edge_image = cv2.dilate(edge_image, np.ones((3,3)), iterations=1)
    edge_image = cv2.erode(edge_image, np.ones((3,3)), iterations=1)
    
    edge_image[edge_image > edge_th] = -1
    segment_pixels = np.where(edge_image == 0)
    segment_id = 1

    while len(segment_pixels[0]) != 0: # while image has pixels with value 0 which means non-labeled segment
        ri, ci = segment_pixels[0][0], segment_pixels[1][0] # get a segment pixel
        
        edge_image = flood_fill(edge_image, (ri, ci), segment_id, connectivity=1, in_place=True) # floodfill segment
        extracted_segment = np.array(edge_image == edge_image[ri][ci]).astype(np.int16) # extract only segment as binary
        extracted_segment = cv2.dilate(extracted_segment, np.ones((3,3)), iterations=1) # expand segment borders by one pixel to remove edges
        np.putmask(edge_image, extracted_segment != 0, segment_id) # overwrite expanded segment to edge_image

        segment_id = segment_id + 1
        segment_pixels = np.where(edge_image == 0)
    
    edge_image[edge_image == -1] = 0 # now 0 means edge
    edge_image = edge_image.astype(np.int16)
    
    return edge_image

# superpixel segmentation
def superpixel_segmentation(image_path:str, region_size:int, ruler:int, verbose:int=0):
    """segments image with opencv superpixel

    Args:
        image_path (str): path to image to segment
        region_size (int): region_size parameter for superpixel
        ruler (int): ruler parameter for superpixel
        verbose (int, optional): verbose level. Defaults to 0.

    Returns:
        numpy.ndarray: segmented image
    """
    image = cv2.imread(image_path)
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.int16)

    # Create a SLIC superpixel object
    slic = cv2.ximgproc.createSuperpixelSLIC(image_lab, algorithm=cv2.ximgproc.MSLIC, region_size=region_size, ruler=ruler)

    # Perform the segmentation
    slic.iterate()

    # Get the mask of superpixel segments
    superpixel_mask = slic.getLabels()
    
    return superpixel_mask + 1 

# kmeans segmentation
def kmeans_segmentation(image_path:str, k:int, color_importance:int, verbose:int=0):
    """segments image with opencv kmeans

    Args:
        image_path (str): path to image to segment
        k (int): k parameter for opencv kmeans
        color_importance (int): importance of pixel colors proportional to pixels coordinates
        verbose (int, optional): verbose level. Defaults to 0.

    Returns:
        numpy.ndarray: segmented image
    """
    image = cv2.imread(image_path)

    # numpy matrix that holds pixel coordinates
    xy_image = np.array([[r,c] for r in range(image.shape[0]) for c in range(image.shape[1])]
                        ).reshape((image.shape[0], image.shape[1], 2)) / color_importance
    pixel_data = np.concatenate([image, xy_image], axis=2)    

    # Convert the image to the required format for K-means (flatten to 2D array), pixels are represented as: [X, Y, COLOR_VALUES]
    if image.shape[-1] == 3:
        pixels = pixel_data.reshape((-1, 5))
    else:
        pixels = pixel_data.reshape((-1, 3))

    # Convert to float32 for kmeans function
    pixels = np.float32(pixels)

    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, _ = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    labels = labels + 1
    labels = np.reshape(labels, (image.shape[0], image.shape[1]))

    return labels

# slickmeans segmentation
def slickmeans_segmentation(image_path:str, region_size:int, ruler:int, k:int, verbose:int=0):
    """segmentation with first slic then kmeans to slic centers

    Args:
        image_path (str): path to image to segment
        region_size (int): region_size parameter for superpixel
        ruler (int): ruler parameter for superpixel
        k (int): k parameter for opencv kmeans
        verbose (int, optional): verbose level. Defaults to 0.

    Returns:
        numpy.ndarray: segmented image
    """
    image = cv2.imread(image_path)
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    slic = cv2.ximgproc.createSuperpixelSLIC(image_lab, algorithm=cv2.ximgproc.MSLIC, region_size=region_size, ruler=ruler)
    slic.iterate()
    
    num_superpixels = slic.getNumberOfSuperpixels()
    superpixel_features = np.zeros((num_superpixels, 5), dtype=np.float32)
    slic_labels = slic.getLabels()

    for i in range(num_superpixels):
        locs = np.where(slic_labels == i)
        center = np.mean(locs, axis=1)
        values = np.mean(image_lab[locs], axis=0)
        superpixel_features[i] = np.hstack((center, values))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, _ = cv2.kmeans(superpixel_features, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    segmented_image = np.zeros_like(image).astype(np.int16)
    for e,l in enumerate(labels):
        segmented_image[np.where(slic_labels==e)] = l

    return segmented_image[:,:,0]

# chan_vase segmentation
def chan_vase_segmentation(image_path:str, number_of_bins:int, verbose:int=0):
    """segmenting image with chan vase segmenting

    Args:
        image_path (str): path to image to segment
        number_of_bins (int): number of segments to extract from chan vase method output
        verbose (int, optional): verbose level. Defaults to 0.

    Returns:
        numpy.ndarray: segmented image
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    cv_results = chan_vese(image, mu=0.25, lambda1=2, lambda2=1, tol=1e-3, max_num_iter=100, dt=0.5, init_level_set="checkerboard", extended_output=True)
    
    # transforming from float image to segment labels
    divider = (256/number_of_bins)
    result = cv_results[1]
    result = (result-result.min())/(result.max()-result.min())
    result = (result*255).astype(np.uint8)
    result = (result//divider)+1

    return result

# felzenszwalb segmentation
def felzenszwalb_segmentation(image_path:str, segment_scale:int, sigma:float, min_segment_size:int, verbose:int=0):
    """segmenting image with felzenszwalb segmentation

    Args:
        image_path (str): path to image to segment
        segment_scale (int): higher value means larger segments
        sigma (float): standard deviation of Gaussian kernel used in preprocessing
        min_segment_size (int): min size of a segment
        verbose (int, optional): verbose level. Defaults to 0.

    Returns:
        numpy.ndarray: segmented image
    """
    image = cv2.imread(image_path)
    segmented_image = felzenszwalb(image, scale=segment_scale, sigma=sigma, min_size=min_segment_size)
    
    return segmented_image + 1

# quickshift segmentation
def quickshift_segmentation(image_path:str, segment_size:int, color_weight:float, verbose:int=0):
    """segmenting image with quickshift segmentation

    Args:
        image_path (str): path to image to segment
        segment_size (int): size of segments
        color_weight (float): between 0-1 higher value means higher color importance
        verbose (int, optional): verbose level. Defaults to 0.

    Returns:
        numpy.ndarray: segmented image
    """
    image = cv2.imread(image_path)
    segmented_image = quickshift(image, kernel_size=3, max_dist=segment_size, ratio=color_weight)
    
    return segmented_image + 1

# graph segmentation
def graph_segmentation(image_path:str, k:int, min_segment_size:int, sigma:float, verbose:int=0):
    """segmenting image with graph segmentation

    Args:
        image_path (str): path to image to segment
        k (int): high values mean smooth few big segments, low values mean detailed finer segments
        min_segment_size (int): min size of a segment
        sigma (float): higher sigma means smoother segment edges, lower values preserve finer details and edges
        verbose (int, optional): verbose level. Defaults to 0.

    Returns:
        numpy.ndarray: segmented image
    """
    image = cv2.imread(image_path)
    
    graph_segmentor = cv2.ximgproc.segmentation.createGraphSegmentation(sigma=sigma, k=k, min_size=min_segment_size)
    labels = graph_segmentor.processImage(image)
    labels = labels + 1

    return labels

# grabcut segmentation
def grabcut_segmentation(image_path:str, verbose:int=0):
    """segmenting image with interactive grabcut segmentation 

    Args:
        image_path (str): path to image to segment
        verbose (int, optional): verbose level. Defaults to 0.

    Returns:
        numpy.ndarray: segmented image
    """
    gb = GrabcutSegmentator()
    labels = gb(image_path)
    return labels

# SAM segmentation
def SAM_segmentation(image_path:str, SAMSegmentator, verbose:int=0):
    """segmenting image with SAM segmentation 

    Args:
        image_path (str): path to image to segment
        verbose (int, optional): verbose level. Defaults to 0.

    Returns:
        numpy.ndarray: segmented image
    """
    labels = SAMSegmentator(image_path)
    return labels

# capsulates all segmentation techniques in one function
def segment_image(image_path:str="", method:str="", edge_th:int=60, bilateral_d:int=7, sigmaColor:int=100, sigmaSpace:int=100,
                  templateWindowSize:int=7, searchWindowSize:int=21, h:int=10, hColor:int=10, region_size:int=40, ruler:int=30,
                  k:int=15, color_importance:int=5, number_of_bins:int=20, segment_scale:int=100, sigma:float=0.5,
                  min_segment_size:int=100, segment_size:int=100, color_weight:float=0.5, SAMSegmentator=None, verbose:int=0):
    """segments image with selected segmentation process. Multiple same/similar meaning carrying parameters
    has used to create a clear distinction between different segmentation techniques. Further inforamtion could
    be obtained from directly each techniques descrtion.

    Args:
        image_path (str, optional): path to image to segment. Defaults to "".
        method (str): type of segmentation proces
        edge_th (int, optional): threshold to consider a pixel as edge. Defaults to 60.
        bilateral_d (int, optional): window size for cv2.bilateral. Defaults to 7.
        sigmaColor (int, optional): color strength for cv2.bilateral. Defaults to 100.
        sigmaSpace (int, optional): distance strength for cv2.bilateral. Defaults to 100.
        templateWindowSize (int, optional): window size for cv2.fastNlMeansDenoisingColored. Defaults to 7.
        searchWindowSize (int, optional): window size for cv2.fastNlMeansDenoisingColored. Defaults to 21.
        h (int, optional): noise remove strenght for cv2.fastNlMeansDenoisingColored. Defaults to 10.
        hColor (int, optional): color noise remove strenght for cv2.fastNlMeansDenoisingColored. Defaults to 10.
        region_size (int, optional): region_size parameter for superpixel. Defaults to 40.
        ruler (int, optional): ruler parameter for superpixel. Defaults to 30.
        k (int, optional): k parameter for opencv kmeans or graph segmentation. Defaults to 15.
        color_importance (int, optional): importance of pixel colors proportional to pixels coordinates for kmeans. Defaults to 5.
        number_of_bins (int): number of segments to extract from chan vase method output
        segment_scale (int): segment scale parameter for felzenszwalb. Defaults to 100.
        sigma (float): standard deviation of Gaussian kernel in felzenszwalb or sigma parameter for graph segmentation. Defaults to 0.5.
        min_segment_size (int): min size of a segment for felzenszwalb or graph. Defaults to 100.
        segment_size (int): size of segments for felzenszwalb, quickshift or graph. Defaults to 100.
        color_weight (float): weight of color to space in quickshift. Defaults to 0.5.
        SAMSegmentator(SAMSegmentator): SAMSegmentator object. Defaults to None.
        verbose (int, optional): verbose level. Defaults to 0.

    Returns:
        numpy.ndarray: segmented image, segment ids start from 1, edges between segments are 0 if exist
    """
    if method == "edge":
        result_image = edge_segmentation(image_path, edge_th, bilateral_d, sigmaColor, sigmaSpace, templateWindowSize, searchWindowSize, h, hColor, verbose=verbose-1)
    elif method == "superpixel":
        result_image = superpixel_segmentation(image_path, region_size=region_size, ruler=ruler, verbose=verbose-1)
    elif method == "kmeans":
        result_image = kmeans_segmentation(image_path, k=k, color_importance=color_importance, verbose=verbose-1)
    elif method == "slickmeans":
        result_image = slickmeans_segmentation(image_path, region_size=region_size, ruler=ruler, k=k, verbose=verbose-1)
    elif method == "chanvase":
        result_image = chan_vase_segmentation(image_path, number_of_bins=number_of_bins, verbose=verbose-1)
    elif method == "felzenszwalb":
        result_image = felzenszwalb_segmentation(image_path, segment_scale=segment_scale, sigma=sigma, min_segment_size=min_segment_size, verbose=verbose-1)
    elif method == "quickshift":
        result_image = quickshift_segmentation(image_path, segment_size=segment_size, color_weight=color_weight, verbose=verbose-1)
    elif method == "graph":
        result_image = graph_segmentation(image_path, k, min_segment_size, sigma, verbose=verbose-1)
    elif method == "grabcut":
        result_image = grabcut_segmentation(image_path, verbose=verbose-1)
    elif method == "SAM":
        result_image = SAM_segmentation(image_path, SAMSegmentator, verbose=verbose-1)

    # Below methods are not implemented because they are not suited for multiclass image segmentation tasks
    # But they could be use for singleclass similar object detection tasks
    # watershed
    # contours

    return result_image

# displays all segmentation method outputs
def preview_methods(image_path:str="", edge_th:int=60, bilateral_d:int=7, sigmaColor:int=100, sigmaSpace:int=100,
                  templateWindowSize:int=7, searchWindowSize:int=21, h:int=10, hColor:int=10, region_size:int=40, ruler:int=30,
                  k:int=15, color_importance:int=5, number_of_bins:int=20, segment_scale:int=100, sigma:float=0.5,
                  min_segment_size:int=100, segment_size:int=100, color_weight:float=0.5, verbose:int=0):
    """previews all segmentation methods on image. Multiple same/similar meaning carrying parameters
    has used to create a clear distinction between different segmentation techniques. Further inforamtion could
    be obtained from directly each techniques descrtion.

    Args:
        image_path (str, optional): path to image to segment. Defaults to "".
        edge_th (int, optional): threshold to consider a pixel as edge. Defaults to 60.
        bilateral_d (int, optional): window size for cv2.bilateral. Defaults to 7.
        sigmaColor (int, optional): color strength for cv2.bilateral. Defaults to 100.
        sigmaSpace (int, optional): distance strength for cv2.bilateral. Defaults to 100.
        templateWindowSize (int, optional): window size for cv2.fastNlMeansDenoisingColored. Defaults to 7.
        searchWindowSize (int, optional): window size for cv2.fastNlMeansDenoisingColored. Defaults to 21.
        h (int, optional): noise remove strenght for cv2.fastNlMeansDenoisingColored. Defaults to 10.
        hColor (int, optional): color noise remove strenght for cv2.fastNlMeansDenoisingColored. Defaults to 10.
        region_size (int, optional): region_size parameter for superpixel. Defaults to 40.
        ruler (int, optional): ruler parameter for superpixel. Defaults to 30.
        k (int, optional): k parameter for opencv kmeans or graph segmentation. Defaults to 15.
        color_importance (int, optional): importance of pixel colors proportional to pixels coordinates for kmeans. Defaults to 5.
        number_of_bins (int): number of segments to extract from chan vase method output
        segment_scale (int): segment scale parameter for felzenszwalb. Defaults to 100.
        sigma (float): standard deviation of Gaussian kernel in felzenszwalb or sigma parameter for graph segmentation. Defaults to 0.5.
        min_segment_size (int): min size of a segment for felzenszwalb or graph. Defaults to 100.
        segment_size (int): size of segments for felzenszwalb, quickshift or graph. Defaults to 100.
        color_weight (float): weight of color to space in quickshift. Defaults to 0.5.
        verbose (int, optional): verbose level. Defaults to 0.
    """

    edge_image = edge_segmentation(image_path, edge_th, bilateral_d, sigmaColor, sigmaSpace, templateWindowSize, searchWindowSize, h, hColor, verbose=verbose-1)
    cv2.imshow("edge", (edge_image * (255 // edge_image.max()) if edge_image.max() < 255 else edge_image).astype(np.uint8))
    superpixel_image = superpixel_segmentation(image_path, region_size=region_size, ruler=ruler, verbose=verbose-1)
    cv2.imshow("superpixel", (superpixel_image * (255 // superpixel_image.max()) if superpixel_image.max() < 255 else superpixel_image).astype(np.uint8))
    kmeans_image = kmeans_segmentation(image_path, k=k, color_importance=color_importance, verbose=verbose-1)
    cv2.imshow("kmeans", (kmeans_image * (255 // kmeans_image.max()) if kmeans_image.max() < 255 else kmeans_image).astype(np.uint8))
    slickmeans_image = slickmeans_segmentation(image_path, region_size=region_size, ruler=ruler, k=k, verbose=verbose-1)
    cv2.imshow("slickmeans", (slickmeans_image * (255 // slickmeans_image.max()) if slickmeans_image.max() < 255 else slickmeans_image).astype(np.uint8))
    chan_vase_image = chan_vase_segmentation(image_path, number_of_bins=number_of_bins, verbose=verbose-1)
    cv2.imshow("chanvase", (chan_vase_image * (255 // chan_vase_image.max()) if chan_vase_image.max() < 255 else chan_vase_image).astype(np.uint8))
    felzenszwalb_image = felzenszwalb_segmentation(image_path, segment_scale=segment_scale, sigma=sigma, min_segment_size=min_segment_size, verbose=verbose-1)
    cv2.imshow("felzenszwalb", (felzenszwalb_image * (255 // felzenszwalb_image.max()) if felzenszwalb_image.max() < 255 else felzenszwalb_image).astype(np.uint8))
    quickshift_image = quickshift_segmentation(image_path, segment_size=segment_size, color_weight=color_weight, verbose=verbose-1)
    cv2.imshow("quickshift", (quickshift_image * (255 // quickshift_image.max()) if quickshift_image.max() < 255 else quickshift_image).astype(np.uint8))
    graph_image = graph_segmentation(image_path, k, min_segment_size, sigma, verbose=verbose-1)
    cv2.imshow("graph", (graph_image * (255 // graph_image.max()) if graph_image.max() < 255 else graph_image).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# fills selected segments with color
def fill(result_image, segmented_image, painted_pixels, click_row:int, click_column:int, color, verbose:int=0):
    """fills segment that selected pixel belongs in result image according to segmented_image and painted_pixels

    Args:
        result_image (numpy.ndarray): image to fill
        segmented_image (numpy.ndarray): processed segments of result_image
        painted_pixels (numpy.ndarray): numpy matrix same size as result_image that indicates which pixels are filled
        click_row (int): row index of selected pixel
        click_column (int): column index of selected pixel
        color (list): BGR values of color that is being filled
        verbose (int, optional): verbose level. Defaults to 0.
    """
    # get selected segment pixels on all layers at image being segmented
    if painted_pixels[click_row, click_column] == 1: # if this pixel is previously painted, so we should overpaint it on the result image
        selected_segment_B = flood(result_image[:,:,0], (click_row, click_column), connectivity=1).astype(np.uint8)
        selected_segment_G = flood(result_image[:,:,1], (click_row, click_column), connectivity=1).astype(np.uint8)
        selected_segment_R = flood(result_image[:,:,2], (click_row, click_column), connectivity=1).astype(np.uint8)
        selected_segment = np.logical_and(selected_segment_B, selected_segment_G, selected_segment_R)
    else: # get selected segment pixels on segmented image
        selected_segment = flood(segmented_image, (click_row, click_column), connectivity=1).astype(np.uint8)

    # fill segment
    result_image[:,:,0][selected_segment==1] = color[0]
    result_image[:,:,1][selected_segment==1] = color[1]
    result_image[:,:,2][selected_segment==1] = color[2]
    
    # mark as painted
    painted_pixels[selected_segment==1] = 1

# unfills selected segment
def unfill(result_image, painted_pixels, raw_image:int, click_row:int, click_column, verbose:int=0):
    """unfills segment that selected pixel belongs in result image according to segmented_image and painted_pixels

    Args:
        result_image (numpy.ndarray): image to unfill
        painted_pixels (numpy.ndarray): numpy matrix same size as result_image that indicates which pixels are filled
        raw_image (numpy.ndarray): non-processed original image
        click_row (int): row index of selected pixel
        click_column (int): column index of selected pixel
        verbose (int, optional): verbose level. Defaults to 0.
    """
    # get selected segment pixels on all layers at image being segmented
    selected_segment_B = flood(result_image[:,:,0], (click_row, click_column), connectivity=1).astype(np.uint8)
    selected_segment_G = flood(result_image[:,:,1], (click_row, click_column), connectivity=1).astype(np.uint8)
    selected_segment_R = flood(result_image[:,:,2], (click_row, click_column), connectivity=1).astype(np.uint8)
    selected_segment = np.logical_and(selected_segment_B, selected_segment_G, selected_segment_R)

    # unfill segmenting
    result_image[:,:,0][selected_segment==1] = raw_image[:,:,0][selected_segment==1]
    result_image[:,:,1][selected_segment==1] = raw_image[:,:,1][selected_segment==1]
    result_image[:,:,2][selected_segment==1] = raw_image[:,:,2][selected_segment==1]
    
    # mark as not painted
    painted_pixels[selected_segment==1] = 0

# places templates and corresponding paints
def put_template_segments(raw_image, result_image, painted_pixels, temp_att_seg_mask:list, threshold:float=None, verbose:int=0):
    """automatically segments given templates

    Args:
        raw_image (numpy.ndarray): raw image to search templates on
        result_image (numpy.ndarray): processing image
        painted_pixels (numpy.ndarray): numpy matrix same size as result_image that indicates which pixels are filled
        temp_att_seg_mask (list): list of templates, attentions(template masks), segments and masks(segment masks)
        threshold (float, optional): max error rate to consider a template as matched, if None, best match is considered. Defaults to None.
        verbose (int, optional): verbose level. Defaults to 0.
    """
    for template, attention, segment, mask in temp_att_seg_mask:
        # attention is non zero where we want to compare with
        matches = cv2.matchTemplate(raw_image, template, mask=attention, method=cv2.TM_SQDIFF_NORMED)
        
        if threshold:
            loc = np.where(matches <= threshold)
            loc = list(zip(loc[0], loc[1]))
        else:
            loc = [np.unravel_index(np.argmin(matches, axis=None), matches.shape)]

        for r,c in loc:
            # segment with expanded borders to match image size
            scaled_segment = np.zeros_like(result_image)
            scaled_segment[r:r+template.shape[0], c:c+template.shape[1]] = segment
            # segments mask with expanded borders to match image size
            scaled_segment_mask = np.zeros_like(result_image)
            scaled_segment_mask[r:r+mask.shape[0], c:c+mask.shape[1]] = mask

            # newly painted pixels with expanded borders to match image size
            painted_pixel_mask = np.zeros_like(painted_pixels)
            for i in range(3):
                painted_pixel_mask[scaled_segment_mask[:,:,i]!=0] = 1

            np.putmask(result_image, mask=scaled_segment_mask, values=scaled_segment)
            np.putmask(painted_pixels, mask=painted_pixel_mask, values=np.ones_like(painted_pixels))

# prints verboses in a format
def print_verbose(verbose_type, message:str, verbose:int=0):
    """Prints verbose messages

    Args:
        verbose_type (int or str): int for indicating batch_idx or string for result/error
        message (str): message to print
        verbose (int, optional): verbose level. Defaults to 0.
    """
    output = "[" + time.strftime("%H:%M:%S") + "] - " 
    if verbose_type == "q":
        output = output + "[quit]  | " + message
    elif verbose_type == "n":
        output = output + "[next]  | " + message
    elif verbose_type == "p":
        output = output + "[prev]  | " + message
    elif verbose_type == "s":
        output = output + "[save]  | " + message
    elif verbose_type == "e":
        output = output + "[error] | " + message
        raise(ErrorException(output))
    else:
        output = output + "[vt ex] | wrong verbose type"
        raise(WrongTypeException(output))

    print(output)
