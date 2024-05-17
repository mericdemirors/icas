import os
import json
import time
import shutil
import random
import datetime
import concurrent.futures

import cv2
import numpy as np
from tqdm.auto import tqdm

import imagehash
from PIL import Image
from datasketch import MinHash
from skimage.metrics import structural_similarity

from helper_exceptions import *
from global_variables import GLOBAL_THREADS

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Clustering images
def cluster(sorted_similarities, clustering_threshold, verbose=0):
    """clusters list of similarities of item pairs, if and X and Y image are more similar then clustering threshold, they are putted into same cluster, if any X and Y image has an image chain X-A-B-...-M-Y that has consecutive pair similarities bigger than threshold, they are putted into same cluster

    Args:
        sorted_similarities (list): list of item pairs and their similarities in form [((X, Y), similarity_point), ...]
        clustering_threshold (float): decides if X and Y are close or not
        verbose (int, optional): verbose level. Defaults to 0.
        
    Returns:
        tuple: clusters -> list of clusters, clustered_images -> list of items that are clustered
    """
    clustered_images = []
    clusters = []

    for (image1, image2), sim in sorted_similarities:
        if sim < clustering_threshold:
            continue

        # if both are not added: new class found
        if (image1 not in clustered_images) and (image2 not in clustered_images):
            clustered_images.append(image1)
            clustered_images.append(image2)
            clusters.append([image1, image2])

        # if both are added in different cluster: merge 2 different parts of the same cluster
        elif (image1 in clustered_images) and (image2 in clustered_images):
            for c in clusters:
                if image1 in c:
                    image1_c = c
                if image2 in c:
                    image2_c = c
            if image1_c != image2_c:
                clusters.append(image1_c + image2_c)
                clusters.remove(image1_c)
                clusters.remove(image2_c)

        # if only one is added: add non-added one to others cluster
        elif (image1 not in clustered_images) and image2 in clustered_images:
            for c in clusters:
                if image2 in c:
                    c.append(image1)
                    clustered_images.append(image1)
        elif (image2 not in clustered_images) and image1 in clustered_images:
            for c in clusters:
                if image1 in c:
                    c.append(image2)
                    clustered_images.append(image2)

    return clusters, clustered_images

def similarity_methods(method, images, image1_file, image2_file, verbose=0):
    """similarity calculation between 2 image for each method

    Args:
        method (str): similarity method
        images (dictionary): dictionary of image paths and image feeatues
        image1_file (str): first images path
        image2_file (str): second images path
        verbose (int, optional): verbose level. Defaults to 0.

    Returns:
        float: similarity value
    """
    if method == "SSIM":
        sim = structural_similarity(images[image1_file], images[image2_file], full=True)[0]
    
    elif method == "minhash":
        sim = images[image1_file].jaccard(images[image2_file])
    
    elif method == "imagehash":
        sim = 1 - (images[image1_file] - images[image2_file]) / len(images[image1_file].hash)
    
    elif method == "ORB":
        matches = bf.match(images[image1_file], images[image2_file])
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = [match for match in matches if match.distance < 0.75 * max(matches, key=lambda x: x.distance).distance]
        sim = len(good_matches) / len(matches)
    
    elif method == "TM":
        sim = 1 - np.float64(np.min(cv2.matchTemplate(images[image1_file], images[image2_file], cv2.TM_SQDIFF_NORMED)))

    return sim

# Calculating similarities of 2 images (needs to be faster, needs to have early stopping)
def calculate_similarity(
    tpl,
    im1_idx,
    im2_idx,
    chunk_idx,
    lock,
    similarity_threshold,
    image_similarities,
    images,
    bools,
    last_checkpoint_time,
    last_verbose_time,
    chunk_last_work_time_dict,
    batch_idx,
    chunk_time_threshold,
    destination_container_folder_base,
    method,
    verbose=0
):
    """calculates 2 images similarity based on structural_similarity, this function is runned by threads. If any X and Y are found to be similar, any pair that has Y in it is discarded from calculation queue because if X and Y are similar and Y and Z are similar, then X and Z should be similar too.

    Args:
        tpl (tuple): tuple of image paths pair
        im1_idx (int): index of first image in images list
        im2_idx (int): index of second image in images list
        chunk_idx (int): chunks id between all chunks
        lock (Lock from threading library): for locking the threads
        similarity_threshold (float): decides if 2 image is similar or not
        image_similarities (dictionary): dictionary in form ((image1, image2): similarity_between_image1_and_image2)
        images (dictionary): dictionary of image path:image feature
        bools (numpy.ndarray): index [i, j] indicates whether to calculate the similarity of image i and j or not
        last_checkpoint_time (list): last time of checkpoint, when did image_similarities last saved into a file
        last_verbose_time (list): last time of verbose
        chunk_last_work_time_dict (dictionary): stores when did a chunk last found a similar pair
        batch_idx (int): indicates the index of batch
        chunk_time_threshold (int): inactivation time limit for checkpoint saving
        destination_container_folder_base (string): path to write checkpoints inside
        method (string): method type for similarity calculation
        verbose (int, optional): whether to print progress or not. Defaults to 1.
    """
    now = datetime.datetime.now()

    image1_file, image2_file = tpl

    if bools[im1_idx][im2_idx]:
        # trying to calculate similarity
        try:
            sim = similarity_methods(method, images, image1_file, image2_file, verbose=verbose-1)
        except Exception as e:
            sim = -np.inf
            print_verbose("w", "error while similarity calculation(setting image similarity to -np.inf):\n" + str(e), verbose=verbose-1)

        # if similar
        if sim > similarity_threshold:
            lock.acquire()
            chunk_last_work_time_dict[chunk_idx] = now

            # store the similarity and disable indices for computation efficiency
            image_similarities[(image1_file, image2_file)] = sim
            bools[im2_idx, :] = 0
            bools[:, im2_idx] = 0

            # if it is time to verbose
            if verbose and (now - last_verbose_time[0]).total_seconds() > 60:
                last_verbose_time[0] = now
                print_verbose(batch_idx, "remaining combinations to check: " + str(int(bools.sum() / 2)), verbose=verbose-1)
            lock.release()

    # if it is time to save a checkpoint
    if (now - last_checkpoint_time[0]).total_seconds() > chunk_time_threshold:
        lock.acquire()
        last_checkpoint_time[0] = now
        with open(os.path.join(destination_container_folder_base, "image_similarities_batch_" + str(batch_idx) + "_checkpoint.json"), "w") as json_file:
            json.dump({str(k): v for (k, v) in list(image_similarities.items())}, json_file, indent="")
        print_verbose(batch_idx, "image_similarities saved(checkpoint)", verbose=verbose-1)
        lock.release()

# saves similarities into a json
def save_checkpoint(batch_idx, path_to_write, image_similarities, verbose=0):
    """saves checkpoint files of image similarities

    Args:
        batch_idx (int): index of batch
        path_to_write (str): path to write checkpoint
        image_similarities (dictionary): dictionary of image pairs similarity in that batch
        verbose (int, optional): verbose level. Defaults to 0.
    """
    # last checkpoint at the end of process
    dict_to_save = {str(k): v for (k, v) in list(image_similarities.items())}
    with open(os.path.join(path_to_write, "image_similarities_batch_" + str(batch_idx) + ".json"), "w") as json_file:
        json.dump(dict_to_save, json_file, indent="")
    print_verbose(batch_idx, "image_similarities saved", verbose=verbose-1)

# How to read checkpoint dictionary
def load_checkpoint(path, verbose=0):
    """Loads a saved checkpoint file

    Args:
        path (string): path of checkpoint file
        verbose (int, optional): verbose level. Defaults to 0.

    Returns:
        dictionary: loaded save
    """
    with open(path, 'r') as file:
        loaded_dict = json.load(file)
    loaded = {(k.split(", ")[0][2:-1], k.split(", ")[1][1:-2]):v for (k,v) in loaded_dict.items()}
    return loaded

# transfers image from folder to folder
def image_transfer(transfer, source_path, destination_path):
    """encapsulation of image transfering

    Args:
        transfer (str): type of transfer
        source_path (str): source path
        destination_path (str): destination path
    """
    if transfer == "copy":
        shutil.copy(source_path, destination_path)
    if transfer == "move":
        shutil.move(source_path, destination_path)

# writes clusters into destination
def write_clusters(clusters, batch_idx, destination_container_folder, outliers, transfer, verbose=0):
    """writes image clusters to a folder

    Args:
        clusters (list): list of clusters
        batch_idx (int): index of batch
        destination_container_folder (str): path to folder to write into
        outliers (list): list of outlier images
        transfer (str): transfer type of images
        verbose (int, optional): verbose level. Defaults to 0.
    """
    # Loop through each cluster, create a folder and transfer images in that cluster to folder
    for i, image_list in enumerate([c for c in clusters if len(c) > 1]):
        destination_folder_path = os.path.join(destination_container_folder, "batch_" + str(batch_idx), "cluster_" + str(i))

        os.makedirs(destination_folder_path)
        [image_transfer(transfer, image_filename, destination_folder_path) for image_filename in image_list]

    # write all non-clustered image into outliers folder
    destination_folder_path = os.path.join(destination_container_folder, "batch_" + str(batch_idx), "outliers")
    os.makedirs(destination_folder_path)
        
    [image_transfer(transfer, image_filename, destination_folder_path) for image_filename in outliers]

    print_verbose(batch_idx, str(len(clusters)) + " cluster found", verbose=verbose-1)

# prints verboses in a format
def print_verbose(verbose_type, message, verbose=0):
    """Prints verbose messages

    Args:
        verbose_type (int or str): int for indicating batch_idx or string for result/error
        message (str): message to print
        verbose (int, optional): verbose level. Defaults to 0.
    """
    output = "[" + time.strftime("%H:%M:%S") + "] - " 
    if isinstance(verbose_type, int):
        output = output + "[batch " + str(verbose_type) + "]  | " + message
    elif verbose_type == "r":
        output = output + "[result]   | " + message
    elif verbose_type == "v":
        output = output + "[verbose]  | " + message
    elif verbose_type == "m":
        output = output + "[merge]    | " + message
    elif verbose_type == "w":
        output = output + "[warning]  | " + message
    elif verbose_type == "f":
        output = output + "[finish]   | " + message
        raise(FinishException(output))
    elif verbose_type == "e":
        output = output + "[error]    | " + message
        raise(ErrorException(output))
    else:
        output = output + "[wrong vt] | wrong verbose type"
        raise(WrongTypeException(output))

    if verbose > 0:
        print(output)

# threads a given function with given parameters to given number of threads
def thread_this(func, params):
    """Treads given function to speed it up

    Args:
        func (python function): function to speed up
        params (list): list of parameters to run function on

    Returns:
        list: list of parallel execution results
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=GLOBAL_THREADS) as executor:
        results = list(executor.map(func, params))
    return results

# returns images and related features dict
def get_image_features(method, image_paths, size, scale, verbose=0):
    """returns images and related features dict

    Args:
        method (str): method to calculate similarity, decides features
        image_paths (list): list of image paths
        size (tuple): dsize parameters for cv2.resize
        scale (tuple): fx and fy parameters for cv2.resize
        verbose (int, optional): verbose level. Defaults to 0.

        Returns:
        dictionary: dictionary of images and related features
    """

    image_features = {}

    if method == "SSIM":
        image_features = {image_file:read_and_resize(image_file, size, scale) for image_file in tqdm(image_paths, desc="Reading images for SSIM, may take a while", leave=False)}

    elif method == "minhash":
        # returns most distinct n corners coordinates of image
        def get_corner_features(gray_image, blockSize=2, ksize=3, k=0.04, top_n_corners=100, verbose=0):
            """Extracts most distinct top_n_corners location as an array

            Args:
                gray_image (numpy.ndarray): grayscale image to extarct feature from
                blockSize (int, optional): cv2.cornerHarris parameter. Defaults to 2.
                ksize (int, optional): cv2.cornerHarris parameter. Defaults to 3.
                k (float, optional): cv2.cornerHarris parameter. Defaults to 0.04.
                top_n_corners (int, optional): number of corners to extract from image. Defaults to 100.
                verbose (int, optional): verbose level. Defaults to 0.
                
            Returns:
                list: list of corner coordinates
            """
            # returns most obvious n corner points indices in list
            corners = cv2.cornerHarris(gray_image, blockSize, ksize, k)
            flattened_corners = corners.flatten()

            largest_corner_indices = np.argpartition(flattened_corners, max(-top_n_corners, len(flattened_corners)-1))[-top_n_corners:]
            corner_rows, corner_cols = np.unravel_index(largest_corner_indices, corners.shape)
            corner_rows = np.concatenate((sorted(corner_rows), np.zeros(100 - len(corner_rows))))/corners.shape[0]
            corner_cols = np.concatenate((sorted(corner_cols), np.zeros(100 - len(corner_cols))))/corners.shape[1]
            corner_features = np.concatenate((corner_rows, corner_cols))
            return corner_features
        
        def get_image_corners(image_file):
            """gets given images corner features, this method is writed to suit to thread_this() call

            Args:
                image_file (str): name of image file

            Returns:
                list: list of corner features of given image
            """
            return get_corner_features(read_and_resize(image_file, size, scale), verbose=verbose-1)

        results = thread_this(get_image_corners, image_paths)
        image_corners_dict = {image_file:results[e] for e, image_file in enumerate(image_paths)}

        image_mh = MinHash()
        for (file, corners) in tqdm(list(image_corners_dict.items()), desc="Minhashing features", leave=False):
            image_mh.update_batch(corners)
            image_features[file] = image_mh.copy()
            image_mh.clear()

    elif method == "imagehash":
        def get_image_hash(image_file):
            """gets given images perceptual hash, this method is writed to suit to thread_this() call

            Args:
                image_file (str): name of image file

            Returns:
                <class 'imagehash.ImageHash'>: hash of given image
            """
            image = Image.open(image_file)
            resized_image = image.resize((int(image.size[0] * scale[0]), int(image.size[1] * scale[1])))
            return imagehash.phash(resized_image, hash_size=64, highfreq_factor=16)
        
        results = thread_this(get_image_hash, image_paths)
        image_features = {image_file:results[e] for e, image_file in enumerate(image_paths)}

    elif method == "ORB":
        orb = cv2.ORB_create()
        def get_image_fetaures(image_file):
            """gets given images ORB features, this method is writed to suit to thread_this() call

            Args:
                image_file (str): name of image file

            Returns:
                numpy.ndarray: features of image
            """
            image_keypoints, image_descriptors = orb.detectAndCompute(read_and_resize(image_file, size, scale), None)
            return image_descriptors

        results = thread_this(get_image_fetaures, image_paths)
        image_features = {image_file:results[e] for e, image_file in enumerate(image_paths)}

    elif method == "TM":
        image_features = {image_file:read_and_resize(image_file, size, scale, gray=False) for image_file in tqdm(image_paths, desc="Reading images for TM, may take a while", leave=False)}

    return image_features

# reads and resizes images
def read_and_resize(path, size=(0,0), scale=(1.0, 1.0), gray=True):
    """reads and resizes image with opencv

    Args:
        path (str): path to image file
        size (tuple, optional): dsize parameters for cv2.resize
        scale (tuple, optional): fx and fy parameters for cv2.resize
        gray (bool, optional): indicates whether to read image grayscale or RGB. Defaults to True.

    Returns:
        numpy.ndarray: readed and resized image
    """
    if gray:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(path)

    # if no output size is provided, scale the image
    if size == (0,0):
        image = cv2.resize(image, dsize=size, fx=scale[0], fy=scale[1])
    else: # if output size is provided resize it
        image = cv2.resize(image, dsize=size)

    return image



# generates test dataset
def generate_test_dataset(path, count, size=256, x=40, y=220, rand_RGB_value=0, rand_xy_value=5, font_scale=9, font_thickness=45):
    """function to generate test dataset

    Args:
        path (str): folder path to generate images in
        count (int): number of images
        size (int, optional): size of image. Defaults to 256.
        x (int, optional): x coordinate of character. Defaults to 40.
        y (int, optional): y coordinate of character. Defaults to 220.
        rand_RGB_value (int, optional): random RGB shift. Defaults to 0.
        rand_xy_value (int, optional): random coordinate shift. Defaults to 5.
        font_scale (int, optional): font scale. Defaults to 9.
        font_thickness (int, optional): font thickness. Defaults to 45.

    """
    # generates test image
    def generate_image(character_to_put_on):
        """function to generate test dataset images

        Args:
            character_to_put_on (str): character to write on image

        Returns:
            numpy.ndarray: prepared image
        """
        bg = (220 + random.randint(-rand_RGB_value, rand_RGB_value),
            245 + random.randint(-rand_RGB_value, rand_RGB_value),
            245 + random.randint(-rand_RGB_value, rand_RGB_value))
        background = np.full((size, size, 3), bg, dtype=np.uint8)
        
        # put given character text over background
        background = cv2.putText(background, character_to_put_on,
                                (x + random.randint(-rand_xy_value, rand_xy_value),
                                y + random.randint(-rand_xy_value, rand_xy_value)), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), font_thickness, cv2.LINE_AA) 

        return background

    os.makedirs(path, exist_ok=True)
    for i in range(count):
        character = random.choice("0123456789")
        image = generate_image(character)
        cv2.imwrite(os.path.join(path, str(i) + character + ".png"), image)