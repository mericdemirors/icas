import cv2
import time
import numpy as np
from skimage.morphology import flood_fill, flood
from skimage.segmentation import chan_vese
from skimage.segmentation import felzenszwalb
from skimage.segmentation import quickshift


from helper_exceptions import *

# ! Not finished
def edge_segmentation(image_path, verbose=0):
    """segments image with opencv canny edge detection

    Args:
        image_path (str): path to image to segment
        verbose (int, optional): verbose level. Defaults to 0.

    Returns:
        numpy.ndarray: segmented image
    """
    image_to_process = cv2.imread(image_path)
    edge_image = cv2.Canny(image_to_process, 0, 50, 150)

    # individual edge islands
    kernels = [ # -1 means it should be background, 1 means it should be edge, 0 means ignore
    np.array(([-1, -1, -1],[-1, 0, -1],[-1, -1, -1]), dtype="int"),
    np.array([[-1, -1, -1, -1],[-1, 0, 0, -1],[-1, 0, 0, -1],[-1, -1, -1, -1]], dtype="int"),
    np.array([[-1, -1, -1, -1, -1],[-1, 0, 0, 0, -1],[-1, 0, 0, 0, -1],[-1, 0, 0, 0, -1],[-1, -1, -1, -1, -1]], dtype="int"),
    np.array([[-1, -1, -1, -1, -1, -1],[-1, 0, 0, 0, 0, -1],[-1, 0, 0, 0, 0, -1],[-1, 0, 0, 0, 0, -1],[-1, 0, 0, 0, 0, -1],[-1, -1, -1, -1, -1, -1]], dtype="int"),
    np.array([[-1, -1, -1],[-1,  0, -1],[-1,  0, -1],[-1, -1, -1]], dtype="int"),
    np.array([[-1, -1, -1, -1],[-1,  0,  0, -1],[-1, -1, -1, -1]], dtype="int"),
    np.array([[-1, -1, -1, -1, -1],[-1,  0,  0,  0, -1],[-1, -1, -1, -1, -1]], dtype="int"),
    np.array([[-1, -1, -1],[-1,  0, -1],[-1,  0, -1],[-1,  0, -1],[-1, -1, -1]], dtype="int"),
    np.array([[-1, -1, -1, -1, -1],[-1,  0,  0,  0, -1],[-1,  0,  0,  0, -1],[-1, -1, -1, -1, -1]], dtype="int"),
    np.array([[-1, -1, -1, -1],[-1,  0,  0, -1],[-1,  0,  0, -1],[-1,  0,  0, -1],[-1, -1, -1, -1]], dtype="int"),
    np.array([[-1, -1, -1, -1, -1, -1],[-1,  0,  0,  0,  0, -1],[-1, -1, -1, -1, -1, -1]], dtype="int"),
    np.array([[-1, -1, -1],[-1,  0, -1],[-1,  0, -1],[-1,  0, -1],[-1,  0, -1],[-1, -1, -1]], dtype="int"),
    np.array([[-1, -1, -1, -1, -1, -1],[-1,  0,  0,  0,  0, -1],[-1,  0,  0,  0,  0, -1],[-1, -1, -1, -1, -1, -1]], dtype="int"),
    np.array([[-1, -1, -1, -1],[-1,  0,  0, -1],[-1,  0,  0, -1],[-1,  0,  0, -1],[-1,  0,  0, -1],[-1, -1, -1, -1]], dtype="int"),
    np.array([[-1, -1, -1, -1, -1, -1],[-1,  0,  0,  0,  0, -1],[-1,  0,  0,  0,  0, -1],[-1,  0,  0,  0,  0, -1],[-1, -1, -1, -1, -1, -1]], dtype="int"),
    np.array([[-1, -1, -1, -1, -1],[-1,  0,  0,  0, -1],[-1,  0,  0,  0, -1],[-1,  0,  0,  0, -1],[-1,  0,  0,  0, -1],[-1, -1, -1, -1, -1]], dtype="int"),
    ]

    # surround image with edge so that each background is enclosed with edges
    edge_image = cv2.copyMakeBorder(edge_image, 1,1,1,1, cv2.BORDER_CONSTANT, value=255)

    # hit or miss kernels over detected edges to remove alone edge islands
    for ke in kernels:
        detected_islands = cv2.morphologyEx(edge_image, cv2.MORPH_HITMISS, ke, anchor=(0,0), iterations=1)     
        detected_islands_xys = np.where(detected_islands == 255)
        for (x,y) in list(zip(detected_islands_xys[0], detected_islands_xys[1])):
            edge_image[x:x+ke.shape[0], y:y+ke.shape[1]] = 0

    
    # now -1 means edge (needed for segmenting backgrounds)
    edge_image = edge_image.astype(np.int16)
    edge_image[edge_image == 255] = -1

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
    edge_image = edge_image[1:edge_image.shape[0]-1, 1:edge_image.shape[1]-1] # remove the added border
    return edge_image.astype(np.int16)

def superpixel_segmentation(image_path, region_size, ruler, verbose=0):
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

def kmeans_segmentation(image_path, k, color_importance, verbose=0):
    """segments image with opencv kmeans

    Args:
        image_path (str): path to image to segment
        k (int): k parameter for opencv kmeans
        color_importance (int): importance of pixel colors proportional to pixels coordinates
        verbose (int, optional): verbose level. Defaults to 0.

    Returns:
        numpy.ndarray: segmented image
    """
    image_to_process = cv2.imread(image_path)

    # numpy matrix that holds pixel coordinates
    xy_image = np.array([[r,c] for r in range(image_to_process.shape[0]) for c in range(image_to_process.shape[1])]).reshape((image_to_process.shape[0], image_to_process.shape[1], 2)) / color_importance
    pixel_data = np.concatenate([image_to_process, xy_image], axis=2)    

    # Convert the image to the required format for K-means (flatten to 2D array), pixels are represented as: [X, Y, COLOR_VALUES]
    if image_to_process.shape[-1] == 3:
        pixels = pixel_data.reshape((-1, 5))
    else:
        pixels = pixel_data.reshape((-1, 3))

    # Convert to float32 for kmeans function
    pixels = np.float32(pixels)

    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, _ = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    labels = labels + 1
    labels = np.reshape(labels, (image_to_process.shape[0], image_to_process.shape[1]))

    return labels

def slickmeans_segmentation(image_path, region_size, ruler, k, verbose=0):
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

def chan_vase_segmentation(image_path, number_of_bins, verbose=0):
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

def felzenszwalb_segmentation(image_path, segment_scale, sigma, min_segment_size, verbose=0):
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

def quickshift_segmentation(image_path, segment_size, color_weight, verbose=0):
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

def graph_segmentation(image_path, k, min_segment_size, sigma, verbose=0):
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

def segment_image(method, image_path="", region_size=40, ruler=30, k=15, color_importance=5,
                  number_of_bins=20, segment_scale=100, sigma=0.5, min_segment_size=100,
                  segment_size=100, color_weight=0.5, 
                  verbose=0):
    """segments image with selected segmentation process. Multiple same/similar meaning carrying parameters
    has used to create a clear distinction between different segmentation techniques. Further inforamtion could
    be obtained from directly each techniques descrtion.

    Args:
        method (str): type of segmentation proces
        image_path (str, optional): path to image to segment. Defaults to "".
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

    Returns:
        numpy.ndarray: segmented image, segment ids start from 1, edges between segments are 0 if exist
    """
    if method == "edge":
        result_image = edge_segmentation(image_path, verbose=verbose-1)
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

    # Below methods are not implemented because they are not suited for multiclass image segmentation tasks
    # But they could be use for singleclass similar object detection tasks
    # graph cut (https://github.com/opencv/opencv/blob/master/samples/python/grabcut.py)
    # watershed
    # contours

    return result_image

def fill(result_image, segmented_image, painted_pixels, click_row, click_column, color, verbose=0):
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

def unfill(result_image, painted_pixels, raw_image, click_row, click_column, verbose=0):
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

def put_template_segments(raw_image, result_image, painted_pixels, temp_att_seg_mask, threshold=None, verbose=0):
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

def print_verbose(verbose_type, message, verbose=0):
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
        output = output + "[error]    | " + message
        raise(ErrorException(output))
    else:
        output = output + "[wrong vt] | wrong verbose type"
        raise(WrongTypeException(output))

    print(output)
