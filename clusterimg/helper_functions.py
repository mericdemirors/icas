import os
import shutil
import json
import time
import datetime
import concurrent.futures

import cv2
import numpy as np
from tqdm.auto import tqdm
import imagehash
from PIL import Image
from datasketch import MinHash
from skimage.metrics import structural_similarity

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Clustering images
def cluster(sorted_similarities, clustering_threshold):
    """clusters list of similarities of item pairs, if and X and Y image are more similar then clustering threshold, they are putted into same cluster, if any X and Y image has an image chain X-A-B-...-M-Y that has consecutive pair similarities bigger than threshold, they are putted into same cluster

    Args:
        sorted_similarities (list): list of item pairs and their similarities in form [((X, Y), similarity_point), ...]
        clustering_threshold (float): decides if X and Y are close or not

    Returns:
        tuple: clusters -> list of clusters, added_img -> list of items that are clustered
    """
    added_img = []
    clusters = []

    for (img1, img2), s in sorted_similarities:
        if s < clustering_threshold:
            continue

        # if both are not added: new class found
        if (img1 not in added_img) and (img2 not in added_img):
            added_img.append(img1)
            added_img.append(img2)
            clusters.append([img1, img2])

        # if both are added in different cluster: merge 2 different parts of the same cluster
        elif (img1 in added_img) and (img2 in added_img):
            for c in clusters:
                if img1 in c:
                    img1_c = c
                if img2 in c:
                    img2_c = c
            if img1_c != img2_c:
                clusters.append(img1_c + img2_c)
                clusters.remove(img1_c)
                clusters.remove(img2_c)

        # if only one is added: add non-added one to others cluster
        elif (img1 not in added_img) and img2 in added_img:
            for c in clusters:
                if img2 in c:
                    c.append(img1)
                    added_img.append(img1)
        elif (img2 not in added_img) and img1 in added_img:
            for c in clusters:
                if img1 in c:
                    c.append(img2)
                    added_img.append(img2)

    return clusters, added_img

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
    verbose=1
):
    """calculates 2 images similarity based on structural_similarity, this function is runned by threads. If any X and Y are found to be similar, any pair that has Y in it is discarded from calculation queue because if X and Y are similar and Y and Z are similar, then X and Z should be similar too.

    Args:
        tpl (tuple): tuple of image pair
        im1_idx (int): index of first image in images list
        im2_idx (int): index of second image in images list
        chunk_idx (int): chunks id between all chunks
        lock (Lock from threading library): for locking the threads
        similarity_threshold (float): decides if 2 image is similar or not
        image_similarities (dictionary): dictionary in form ((img1, img2): similarity_between_img1_and_img2)
        images (dictionary): dictionary of image path:image feature
        bools (numpy.ndarray): index [i, j] indicates whether to calculate the similarity of image i and j or not
        last_checkpoint_time (list): last time of checkpoint, when did image_similarities last saved into a file
        last_verbose_time (list): last time of verbose
        chunk_last_work_time_dict (dictionary): stores when did a chunk last found a similar pair
        batch_idx (int): indicates the index of batch
        chunk_time_threshold (datetime.datetime): inactivation time limit for early stopping
        destination_container_folder_base (string): path to write checkpoints inside
        method (string): method type for similarity calculation
        verbose (int, optional): whether to print progress or not. Defaults to 1.
    """
    now = datetime.datetime.now()

    img1_file, img2_file = tpl

    if bools[im1_idx][im2_idx]:
        # trying to calculate similarity, if shape mismatch: not similar
        try:
            if method == "SSIM":
                sim = structural_similarity(images[img1_file], images[img2_file], full=True)[0]
            elif method == "minhash":
                sim = images[img1_file].jaccard(images[img2_file])
            elif method == "imagehash":
                sim = 1 - (images[img1_file] - images[img2_file]) / len(images[img1_file].hash)
            elif method == "ORB":
                matches = bf.match(images[img1_file], images[img2_file])
                matches = sorted(matches, key=lambda x: x.distance)
                good_matches = [match for match in matches if match.distance < 0.75 * max(matches, key=lambda x: x.distance).distance]
                sim = len(good_matches) / len(matches)
            elif method == "TM":
                sim = np.float64(np.min(cv2.matchTemplate(images[img1_file], images[img2_file], cv2.TM_SQDIFF_NORMED)))
            else:
                print_verbose("e", "Error while similarity calculation(setting pair similarity to -np.inf): " + str(e))
                sim = -np.inf
        except Exception as e:
            print_verbose("e", "Error while similarity calculation(setting pair similarity to -np.inf): " + str(e))
            sim = -np.inf

        if sim > similarity_threshold:
            lock.acquire()
            chunk_last_work_time_dict[chunk_idx] = now

            # if similar, store the similarity and disable indices for computation efficiency
            image_similarities[(img1_file, img2_file)] = sim
            bools[im2_idx, :] = 0
            bools[:, im2_idx] = 0

            if verbose:
                if (now - last_verbose_time[0]).total_seconds() > 60:
                    last_verbose_time[0] = now
                    print_verbose(batch_idx, "remaining combinations to check: " + str(int(bools.sum() / 2)))
            lock.release()

    if (now - last_checkpoint_time[0]).total_seconds() > chunk_time_threshold:
        lock.acquire()
        last_checkpoint_time[0] = now
        with open(os.path.join(destination_container_folder_base, "image_similarities_batch_" + str(batch_idx) + "_checkpoint.json"), "w") as json_file:
            json.dump({str(k): v for (k, v) in list(image_similarities.items())}, json_file, indent="\n")
        print_verbose(batch_idx, "image_similarities saved(checkpoint)")
        lock.release()

# saves similarities into a json
def save_checkpoint(batch_idx, path_to_write, image_similarities):
    """saves checkpoint files of image similarities

    Args:
        batch_idx (int): index of batch
        path_to_write (str): path to write checkpoint
        image_similarities (dictionary): dictionary of image pairs similarity in that batch
    """
    # last checkpoint at the end of process
    dict_to_save = {str(k): v for (k, v) in list(image_similarities.items())}
    with open(os.path.join(path_to_write, "image_similarities_batch_" + str(batch_idx) + ".json"), "w") as json_file:
        json.dump(dict_to_save, json_file, indent="\n")
    print_verbose(batch_idx, "image_similarities saved")

# How to read checkpoint dictionary
def load_checkpoint(path):
    """Loads a saved checkpoint file

    Args:
        path (string): path of checkpoint file

    Returns:
        dictionary: loaded save
    """
    with open(path, 'r') as file:
        loaded_dict = json.load(file)
    loaded = {(k.split(", ")[0][2:-1], k.split(", ")[1][1:-2]):v for (k,v) in loaded_dict.items()}
    return loaded

# writes clusters into destination
def write_clusters(clusters, batch_idx, images_folder_path, destination_container_folder, outliers, transfer):
    """writes image clusters to a folder

    Args:
        clusters (list): list of clusters
        batch_idx (int): index of batch
        images_folder_path(str): folder path of images
        destination_container_folder (str): path to folder to write into
        outliers (list): list of outlier images
        transfer (str): transfer type of images
    """
    # Loop through each cluster, create a folder and copy images in that cluster to folder
    for i, image_list in enumerate([c for c in clusters if len(c) > 1]):
        destination_folder_path = os.path.join(destination_container_folder, "batch_" + str(batch_idx), "cluster_" + str(i))

        os.makedirs(destination_folder_path)
        if transfer == "copy":
            [shutil.copy(os.path.join(images_folder_path, image_filename), destination_folder_path) for image_filename in image_list]
        if transfer == "move":
            [shutil.move(os.path.join(images_folder_path, image_filename), destination_folder_path) for image_filename in image_list]


    # write all non-clustered image into outliers folder
    destination_folder_path = os.path.join(destination_container_folder, "batch_" + str(batch_idx), "outliers")
    os.makedirs(destination_folder_path)
    if transfer == "copy":
        [shutil.copy(os.path.join(images_folder_path, image_filename), destination_folder_path) for image_filename in outliers]
    if transfer == "move":
        [shutil.move(os.path.join(images_folder_path, image_filename), destination_folder_path) for image_filename in outliers]


# prints verboses in a format
def print_verbose(verbose_type, message):
    """Prints verbose messages

    Args:
        verbose_type (int or str): int for indicating batch_idx or string for result/error
        message (str): message to print
    """
    output = "[" + time.strftime("%H:%M:%S") + "] - " 
    if isinstance(verbose_type, int):
        output = output + "[batch " + str(verbose_type) + "] | " + message
    elif verbose_type == "r":
        output = output + "[result]  | " + message
    elif verbose_type == "v":
        output = output + "[verbose] | " + message
    elif verbose_type == "m":
        output = output + "[merge]   | " + message
    elif verbose_type == "f":
        output = output + "[finish]  | " + message
        print(output)
        exit(0)
    elif verbose_type == "e":
        output = output + "[error]   | " + message
        print(output)
        exit(0)
    else:
        print("wrong output verbose type")
        exit(0)

    print(output)

# threads a given function with given parameters to given number of threads
def thread_this(func, params, num_of_threads=1024):
    """Treads given function to speed it up

    Args:
        func (python function): function to speed up
        params (list): list of parameters to run function on
        num_of_threads (int, optional): number of threads to share work between. Defaults to 1024.

    Returns:
        list: list of parallel execution results
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_of_threads) as executor:
        results = list(executor.map(func, params))
    return results

# returns most distinct n corners coordinates of image
def get_corner_features(gray_image, blockSize=2, ksize=3, k=0.04, top_n_corners=100):
    """Extracts most distinct top_n_corners location as an array

    Args:
        gray_image (numpy.ndarray): grayscale image to extarct feature from
        blockSize (int, optional): cv2.cornerHarris parameter. Defaults to 2.
        ksize (int, optional): cv2.cornerHarris parameter. Defaults to 3.
        k (float, optional): cv2.cornerHarris parameter. Defaults to 0.04.
        top_n_corners (int, optional): number of corners to extract from image. Defaults to 100.

    Returns:
        list: list of corner coordinates
    """
    # returns most obvious n corners flatted indices
    corners = cv2.cornerHarris(gray_image, blockSize, ksize, k)
    corners = cv2.dilate(corners, None)
    flattened_corners = corners.flatten()

    indices_of_largest_values = np.argpartition(flattened_corners, -top_n_corners)[-top_n_corners:]
    return sorted(indices_of_largest_values)

# calculates similarity between 2 image perceptual hashs
def image_hash_similarity(imgph1, imgph2):
    """calculates similarity between 2 image perceptual hashs

    Args:
        imgph1 (<class 'imagehash.ImageHash'>): hash of first image
        imgph2 (<class 'imagehash.ImageHash'>): hash of second image

    Returns:
        float: similarity metric between 2 hash
    """
    try:
        return 1 - (imgph1 - imgph2) / len(imgph1.hash)  # Normalize the score to be between 0 and 1
    except Exception as e:
        print_verbose("e", "Error while calculating image hash similarity: " + str(e))
    
# returns images and related features dict
def get_images_dict(method, image_files, images_folder_path, scale):
    """returns images and related features dict

    Args:
        method (str): method to calculate similarity, decides features
        image_files (list): list of image file names
        images_folder_path (str): path of image folder
        scale (float): image scale

    Returns:
        dictionary: dictionary of images and related features
    """
    images = {}
    
    if method == "SSIM":
        images = {image_file:cv2.resize(cv2.imread(os.path.join(images_folder_path, image_file), cv2.IMREAD_GRAYSCALE), (0, 0), fx=scale, fy=scale) for image_file in tqdm(image_files, desc="Reading images, may take a while", leave=False)}
    
    elif method == "minhash":
        def get_image_corners(image_file):
            """gets given images corner features, this method is writed to suit to thread_this() call

            Args:
                image_file (str): name of image file

            Returns:
                list: list of corner features of given image
            """
            return get_corner_features(cv2.resize(cv2.imread(os.path.join(images_folder_path, image_file), cv2.IMREAD_GRAYSCALE), (0, 0), fx=scale, fy=scale))

        results = thread_this(get_image_corners, image_files)
        img_corners_dict = {image_file:results[e] for e, image_file in enumerate(image_files)}

        images = {}
        img_mh = MinHash()
        for (file, corners) in tqdm(list(img_corners_dict.items()), desc="Minhashing features", leave=False):
            img_mh.update_batch(corners)
            images[file] = img_mh.copy()
            img_mh.clear()

    elif method == "imagehash":
        def get_image_hash(image_file):
            """gets given images perceptual hash, this method is writed to suit to thread_this() call

            Args:
                image_file (str): name of image file

            Returns:
                <class 'imagehash.ImageHash'>: hash of given image
            """
            img = Image.open(os.path.join(images_folder_path, image_file))
            resized_image = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))
            return imagehash.phash(resized_image, hash_size=64, highfreq_factor=16)
        
        results = thread_this(get_image_hash, image_files)
        images = {image_file:results[e] for e, image_file in enumerate(image_files)}

    elif method == "ORB":
        orb = cv2.ORB_create()
        def get_image_fetaures(image_file):
            """gets given images ORB features, this method is writed to suit to thread_this() call

            Args:
                image_file (str): name of image file

            Returns:
                numpy.ndarray: features of image
            """
            image_keypoints, image_descriptors = orb.detectAndCompute(cv2.resize(cv2.imread(os.path.join(images_folder_path, image_file), cv2.IMREAD_GRAYSCALE), (0, 0), fx=scale, fy=scale), None)
            return image_descriptors

        results = thread_this(get_image_fetaures, image_files)
        images = {image_file:results[e] for e, image_file in enumerate(image_files)}

    elif method == "TM":
        images = {image_file:cv2.resize(cv2.imread(os.path.join(images_folder_path, image_file), cv2.IMREAD_GRAYSCALE), (0, 0), fx=scale, fy=scale) for image_file in tqdm(image_files, desc="Reading images, may take a while", leave=False)}

    return images

# returns templates and related features dict
def get_templates_dict(method, template_files, template_cluster_dict, scale):
    """returns templates and related features dict

    Args:
        method (str): method to calculate similarity, decides features
        template_files (list): list of template file names
        template_folder_path (str): path of template folder
        scale (float): template scale

    Returns:
        dictionary: dictionary of templates and related features
    """

    templates = {}

    if method == "SSIM":
        templates = {template_file:cv2.resize(cv2.imread(os.path.join(template_cluster_dict[template_file], template_file), cv2.IMREAD_GRAYSCALE), (0, 0), fx=scale, fy=scale) for template_file in tqdm(template_files, desc="Reading templates, may take a while", leave=False)}
            
    elif method == "minhash":
        template_corners_dict = {template_file:get_corner_features(cv2.resize(cv2.imread(os.path.join(template_cluster_dict[template_file], template_file), cv2.IMREAD_GRAYSCALE), (0, 0), fx=scale, fy=scale))
                   for template_file in tqdm(template_files, desc="Reading templates, may take a while", leave=False)}
        
        template_mh = MinHash()
        for (template_file, corners) in tqdm(list(template_corners_dict.items()), leave=False):
            template_mh.update_batch(corners)
            templates[template_file] = template_mh.copy()
            template_mh.clear()

    elif method == "imagehash":
        for template_file in tqdm(template_files, desc="Reading templates, may take a while", leave=False):
            template = Image.open(os.path.join(template_cluster_dict[template_file], template_file))
            resized_template = template.resize((int(template.size[0] * scale), int(template.size[1] * scale)))
            templates[template_file] = imagehash.phash(resized_template, hash_size=64, highfreq_factor=16)

    elif method == "ORB":
        orb = cv2.ORB_create()
        def get_image_fetaures(template_file):
            """gets given templates ORB features, this method is writed to suit to thread_this() call

            Args:
                template_file (str): name of template file

            Returns:
                numpy.ndarray: features of image
            """
            image_keypoints, image_descriptors = orb.detectAndCompute(cv2.resize(cv2.imread(os.path.join(template_cluster_dict[template_file], template_file), cv2.IMREAD_GRAYSCALE), (0, 0), fx=scale, fy=scale), None)
            return image_descriptors

        results = thread_this(get_image_fetaures, template_files)
        templates = {template_file:results[e] for e, template_file in enumerate(template_files)}

    elif method == "TM":
        templates = {template_file:cv2.resize(cv2.imread(os.path.join(template_cluster_dict[template_file], template_file), cv2.IMREAD_GRAYSCALE), (0, 0), fx=scale, fy=scale) for template_file in tqdm(template_files, desc="Reading templates, may take a while", leave=False)}

    return templates
