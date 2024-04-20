import cv2
import time
import numpy as np
from skimage.morphology import flood_fill, flood

from helper_exceptions import *

def edge_segmentation(img_path):
    """segments image with opencv canny edge detection

    Args:
        img_path (str): path to image to segment

    Returns:
        numpy.ndarray: segmented image
    """
    img_to_process = cv2.imread(img_path)
    edge_img = cv2.Canny(img_to_process, 0, 50, 150)

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
    edge_img = cv2.copyMakeBorder(edge_img, 1,1,1,1, cv2.BORDER_CONSTANT, value=255)
    for ke in kernels: # pass hit or miss kernels over detected edge images to remove alone edge islands
        detected_islands = cv2.morphologyEx(edge_img, cv2.MORPH_HITMISS, ke, anchor=(0,0), iterations=1)     
        detected_islands_xys = np.where(detected_islands == 255)
        for (x,y) in list(zip(detected_islands_xys[0], detected_islands_xys[1])):
            edge_img[x:x+ke.shape[0], y:y+ke.shape[1]] = 0

    edge_img = edge_img.astype(np.int16)
    edge_img[edge_img == 255] = -1 # now -1 means edge (needed for segmenting backgrounds)

    segment_id = 1
    segment_pixels = np.where(edge_img == 0)

    while len(segment_pixels[0]) != 0: # while image has pixels with value 0 which means non-labeled segment
        ri, ci = segment_pixels[0][0], cluster_pixels[1][0] # get a cluster pixel
        
        edge_img = flood_fill(edge_img, (ri, ci), cluster_id, connectivity=1, in_place=True) # floodfill cluster
        extracted_cluster = np.array(edge_img == edge_img[ri][ci]).astype(np.int16) # extract only cluster as binary
        extracted_cluster = cv2.dilate(extracted_cluster, np.ones((3,3)), iterations=1) # expand cluster borders by one pixel to remove edges
        np.putmask(edge_img, extracted_cluster != 0, cluster_id) # overwrite expanded cluster to edge_img

        cluster_id = cluster_id + 1
        cluster_pixels = np.where(edge_img == 0)

    edge_img[edge_img == -1] = 0 # now 0 means edge
    edge_img = edge_img[1:edge_img.shape[0]-1, 1:edge_img.shape[1]-1] # remove the added border
    return edge_img.astype(np.int16)

def superpixel_segmentation(img_path, region_size, ruler):
    """segments image with opencv superpixel

    Args:
        img_path (str): path to image to segment
        region_size (int): region_size parameter for superpixel
        ruler (int): ruler parameter for superpixel

    Returns:
        numpy.ndarray: segmented image
    """
    img = cv2.imread(img_path)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.int16)

    # Create a SLIC superpixel object
    slic = cv2.ximgproc.createSuperpixelSLIC(img_lab, algorithm=cv2.ximgproc.MSLIC, region_size=region_size, ruler=ruler)

    # Perform the segmentation
    slic.iterate()

    # Get the mask of superpixel segments
    superpixel_mask = slic.getLabels()
    
    return superpixel_mask + 1 

def kmeans_segmentation(img_path, k, color_importance):
    """segments image with opencv kmeans

    Args:
        img_path (str): path to image to segment
        k (int): k parameter for opencv kmeans
        color_importance (int): importance of pixel colors proportional to pixels coordinates

    Returns:
        numpy.ndarray: segmented image
    """
    img_to_process = cv2.imread(img_path)

    num_of_rows, num_of_columns = img_to_process.shape[:2]
    # numpy matrix that holds image coordinates
    xy_img = np.array([[r,c] for r in range(num_of_rows) for c in range(num_of_columns)]).reshape((img_to_process.shape[0], img_to_process.shape[1], 2)) / color_importance
    pixel_data = np.concatenate([img_to_process, xy_img], axis=2)    

    # Convert the img to the required format for K-means (flatten to 2D array), pixels are represented as: [X, Y, COLOR_VALUES]
    if img_to_process.shape[-1] == 3:
        pixels = pixel_data.reshape((-1, 5))
    else:
        pixels = pixel_data.reshape((-1, 3))

    # Convert to float32 for kmeans function
    pixels = np.float32(pixels)

    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, _ = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    labels = labels + 1
    labels = np.reshape(labels, (img_to_process.shape[0], img_to_process.shape[1]))

    return labels

def segment_image(method, img_path="", region_size=40, ruler=30, k=15, color_importance=5):
    """segments image with selected segmentation process

    Args:
        method (str): type of segmentation proces
        img_path (str, optional): path to image to segment. Defaults to "".
        region_size (int, optional): region_size parameter for superpixel. Defaults to 40.
        ruler (int, optional): ruler parameter for superpixel. Defaults to 30.
        k (int, optional): k parameter for opencv kmeans. Defaults to 15.
        color_importance (int, optional): importance of pixel colors proportional to pixels coordinates: _description_. Defaults to 5.

    Returns:
        numpy.ndarray: segmented image, segment ids start from 1, edges between segments are 0 if exist
    """
    if method == "edge":
        result_img = edge_segmentation(img_path)
    elif method == "superpixel":
        result_img = superpixel_segmentation(img_path, region_size=region_size, ruler=ruler)
    elif method == "kmeans":
        result_img = kmeans_segmentation(img_path, k=k, color_importance=color_importance)

    return result_img

def fill(result_img, segmented_img, painted_pixels, click_row, click_column, color):
    """fills segment that selected pixel belongs in result image according to segmented_image and painted_pixels

    Args:
        result_img (numpy.ndarray): image to fill
        segmented_img (numpy.ndarray): processed segments of result_img
        painted_pixels (numpy.ndarray): numpy matrix same size as result_img that indicates which pixels are filled
        click_row (int): row index of selected pixel
        click_column (int): column index of selected pixel
        color (list): BGR values of color that is being filled
    """
    # get selected segment pixels on all layers at image being segmented
    if painted_pixels[click_row, click_column] == 1: # if this pixel is previously painted, so we should overpaint it on the result image
        selected_segment_B = flood(result_img[:,:,0], (click_row, click_column), connectivity=1).astype(np.uint8)
        selected_segment_G = flood(result_img[:,:,1], (click_row, click_column), connectivity=1).astype(np.uint8)
        selected_segment_R = flood(result_img[:,:,2], (click_row, click_column), connectivity=1).astype(np.uint8)
        selected_segment = np.logical_and(selected_segment_B, selected_segment_G, selected_segment_R)
    else: # get selected segment pixels on segmented image
        selected_segment = flood(segmented_img, (click_row, click_column), connectivity=1).astype(np.uint8)

    # fill segment
    result_img[:,:,0][selected_segment==1] = color[0]
    result_img[:,:,1][selected_segment==1] = color[1]
    result_img[:,:,2][selected_segment==1] = color[2]
    
    # mark as painted
    painted_pixels[selected_segment==1] = 1

def unfill(result_img, painted_pixels, raw_img, click_row, click_column):
    """unfills segment that selected pixel belongs in result image according to segmented_image and painted_pixels

    Args:
        result_img (numpy.ndarray): image to unfill
        painted_pixels (numpy.ndarray): numpy matrix same size as result_img that indicates which pixels are filled
        raw_img (numpy.ndarray): non-processed original image
        click_row (int): row index of selected pixel
        click_column (int): column index of selected pixel
    """
    # get selected segment pixels on all layers at image being segmented
    selected_segment_B = flood(result_img[:,:,0], (click_row, click_column), connectivity=1).astype(np.uint8)
    selected_segment_G = flood(result_img[:,:,1], (click_row, click_column), connectivity=1).astype(np.uint8)
    selected_segment_R = flood(result_img[:,:,2], (click_row, click_column), connectivity=1).astype(np.uint8)
    selected_segment = np.logical_and(selected_segment_B, selected_segment_G, selected_segment_R)

    # unfill segmenting
    result_img[:,:,0][selected_segment==1] = raw_img[:,:,0][selected_segment==1]
    result_img[:,:,1][selected_segment==1] = raw_img[:,:,1][selected_segment==1]
    result_img[:,:,2][selected_segment==1] = raw_img[:,:,2][selected_segment==1]
    
    # mark as not painted
    painted_pixels[selected_segment==1] = 0

def print_verbose(verbose_type, message):
    """Prints verbose messages

    Args:
        verbose_type (int or str): int for indicating batch_idx or string for result/error
        message (str): message to print
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
    elif verbose_type == "r":
        output = output + "[reset] | " + message
    elif verbose_type == "e":
        output = output + "[error]    | " + message
        raise(ErrorException(output))
    else:
        output = output + "[wrong vt] | wrong verbose type"
        raise(WrongTypeException(output))

    print(output)
