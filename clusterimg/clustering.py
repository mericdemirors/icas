import os
import shutil
import datetime
import argparse

import cv2
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
import mplcursors

from itertools import combinations
from threading import Lock

from helper_functions import cluster, calculate_similarity, print_verbose, thread_this, image_hash_similarity, save_checkpoint, write_clusters, get_images_dict, get_templates_dict



def parse_arguments():
    """parses and checks argumans provided

    Returns:
        argparse argumans: parsed argumans
    """
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('-path', '-p', type=str, default="", help='Path to image folder. default is: \"\"')
    parser.add_argument('-method', '-m', type=str, default="", help='Method to use. SSIM: slow, good at big images imagehash: mid speed, good at small images minhash: fastest, mid solution for both cases ORB: slow and bad TM: slow and bad. default is: \"\"')
    parser.add_argument('-option', '-o', type=str, default="", help='Option of process. default is: \"\"')
    parser.add_argument('-batch_size', '-b', type=int, default=100, help='Batch size to process images with. default is: 100')
    parser.add_argument('-scale', '-s', type=float, default=1.0, help='Image scale. default is: 1.0')
    parser.add_argument('-threshold', '-t', type=float, default=0.8, help='Threshold to count 2 images as similar. default is: 0.8')
    parser.add_argument('-overwrite', '-ow', type=str, default="N", help='Whether to overwrite old clustered folder or not. default is: N')
    parser.add_argument('-num_of_threads', '-n', type=int, default=4, help='Number of threads to share job pool. default is: 4')
    parser.add_argument('-chunk_time_threshold', '-ct', type=int, default=60, help='Number of seconds to save checkpoint files after. default is: 60')
    parser.add_argument('-transfer', '-tr', type=str, default="copy", help='How to transfer image files. default is: copy')
    parser.add_argument('-verbose', '-v', type=int, default=0, help='Verbose level. default is: 0')
    
    def arguman_check(args, verbose=0):
        """checks given argumans validity

        Args:
            args (argparse argumans): arguman parser object
            verbose (int, optional): verbose level. Defaults to 0.
        """
        valid_methods = ["SSIM", "minhash", "imagehash", "ORB", "TM"]
        valid_options = ["merge", "dontmerge", ""]
        valid_transfer = ["copy", "move"]

        if args.method not in valid_methods:
            print_verbose("e", "invalid method type", verbose)
        if args.option not in valid_options:
            print_verbose("e", "invalid option type", verbose)
        if args.transfer not in valid_transfer:
            print_verbose("e", "invalid transfer type", verbose)
    arguman_check(parser.parse_args(), parser.parse_args().verbose-1)
    
    return parser.parse_args()


args = parse_arguments()
images_folder_path = args.path
method = args.method
option = args.option
batch_size = args.batch_size
scale = args.scale
threshold = args.threshold
overwrite = args.overwrite
num_of_threads = args.num_of_threads
chunk_time_threshold = args.chunk_time_threshold
transfer = args.transfer
verbose = args.verbose

lock = Lock()

# lets user interactively select threshold for some methods
def select_threshold(method, folder_path, num_of_files=1000, verbose=0):
    """Lets user interactively select threshold

    Args:
        method (str): type of similarity calculation method
        folder_path (str): path to image folder
        num_of_files (int, optional): how much image should be processed for interactive selection. Defaults to 1000.
        verbose (int, optional): verbose level. Defaults to 0.

    Returns:
        None: if method is SSIM(structural_similarity) no interactive selection can be made(takes to much time) so return is used for terminating the function
    """
    global threshold
    # sort file names by size tqdm
    all_image_files = filter(lambda x: os.path.isfile(os.path.join(folder_path, x)), os.listdir(folder_path))
    selected_image_files = sorted(all_image_files, key=lambda x: os.stat(os.path.join(folder_path, x)).st_size)[:num_of_files]

    image_feature_dict = get_images_dict(method, selected_image_files, folder_path, scale, verbose=verbose-1)
    print_verbose("v", "process for interactive threshold selection is started.", verbose)
    if method == "SSIM":
        def get_structural_similarity(image_pair):
            """gets given image pairs structural similarity score, this method is writed to suit to thread_this() call

            Args:
                image_pair (tuple): name of pair files

            Returns:
                float: structural similarity
            """
            return structural_similarity(image_pair[0], image_pair[1], full=True)[0]
        
        image_combs = list(combinations(list(image_feature_dict.values()), 2))
        sim_list = thread_this(get_structural_similarity, image_combs)

    elif method == "minhash":
        minhashs_combs = list(combinations(list(image_feature_dict.values()), 2))
        sim_list = [mh1.jaccard(mh2) for (mh1, mh2) in tqdm(minhashs_combs, desc="Calculating similarity", leave=False)]

    elif method == "imagehash":
        file_combs = list(combinations(list(image_feature_dict.keys()), 2))
        sim_list = [image_hash_similarity(image_feature_dict[f1], image_feature_dict[f2], verbose=verbose-1) for (f1, f2) in tqdm(file_combs, desc="Calculating similarity", leave=False)]
    
    elif method == "ORB":
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        def orb_calculate_similarity(image_descriptors_and_bf):
            """calculates similarity over 2 images orb features descriptors

            Args:
                image_descriptors_and_bf (list): [img1_descriptors, img2_descriptors]

            Returns:
                float: similarity of 2 descriptors
            """
            matches = sorted(bf.match(image_descriptors_and_bf[0], image_descriptors_and_bf[1]), key=lambda x: x.distance)
            good_matches = [match for match in matches if match.distance < 0.75 * matches[-1].distance]
            similarity = len(good_matches) / len(matches)

            return similarity

        file_combs = list(combinations(list(image_feature_dict.keys()), 2))
        param_list = [[image_feature_dict[f1], image_feature_dict[f2], bf] for (f1, f2) in file_combs]
        sim_list = thread_this(orb_calculate_similarity, param_list)

    elif method == "TM":
        def get_tempate_matching_similarity(image_pair):
            """gets given image pairs template matching score, this method is writed to suit to thread_this() call

            Args:
                image_pair (tuple): name of pair files

            Returns:
                float: template matching similarity
            """

            return np.min(cv2.matchTemplate(image_pair[0], image_pair[1], cv2.TM_SQDIFF_NORMED))        
        
        image_combs = list(combinations(list(image_feature_dict.values()), 2))
        sim_list = thread_this(get_tempate_matching_similarity, image_combs)

    def on_hover(sel, lenght = len(selected_image_files)):
        """shows an info text if user hovers over plot

        Args:
            sel (<class 'mplcursors._pick_info.Selection'>): interactive selector 
            lenght (int, optional): number of selected files for interactive selection. Defaults to len(selected_image_files).
        """
        sel.annotation.set_text("approximate number of similar pairs in each " + str(lenght) + " image: " + str(int((2*sel.target[0])**0.5 + 1)) + ", threshold: " + str(sel.target[1])[:7])
    def on_click(event):
        """sets threshold on clicked position

        Args:
            event (<class 'matplotlib.backend_bases.MouseEvent'>): event catcher
        """
        global threshold
        if event.button == 1:
            threshold = event.ydata
            plt.close()

    y = sorted(sim_list, reverse=True)
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(y)), y)    
    cursor = mplcursors.cursor(hover=True)
    cursor.connect("add", on_hover)
    plt.gcf().canvas.mpl_connect('button_press_event', on_click)
    plt.show() 

# # setting folders
if option == "merge":
    destination_container_folder = images_folder_path
else:
    base_folder, images_folder_name = os.path.split(images_folder_path)
    destination_container_folder = os.path.join(base_folder, images_folder_name + "_clustered")

if option != "merge":
    if os.path.exists(destination_container_folder) and overwrite not in ["Y", "y"]:
        print_verbose("f", "no permission to overwrite", verbose)
    else:
        if os.path.exists(destination_container_folder):
            shutil.rmtree(destination_container_folder)
        os.makedirs(destination_container_folder)        

# interactive threshold selection is disabled because of "FigureCanvasAgg is non-interactive, and thus cannot be shown" error"
# select_threshold(method, images_folder_path, min(batch_size, 1000))
similarity_threshold = clustering_threshold = threshold

# calculates similarities between given images
def calculate_batch_similarity(batch_idx, image_files, method, verbose=0):
    """calculates similarity inside a batch

    Args:
        batch_idx (int): index of batch
        image_files (list): list of files
        verbose (int, optional): verbose level. Defaults to 0.
        
    Returns:
        dictionary: dictionary of image pairs similarity in that batch
    """
    if len(image_files) < 2:
        return {}

    image_similarities = {}
    images = get_images_dict(method, image_files, images_folder_path, scale, verbose=verbose-1)

    comb = list(combinations(image_files, 2))  # all pair combinations of images
    bools = np.ones((len(image_files), len(image_files)), dtype=np.int8)  # row i means all (i, *) pairs, column j means all (*, j) pairs
    print_verbose(batch_idx, "processing total of " + str(len(comb)) + " pair combinations", verbose)

    # divide the computations to chunks
    if num_of_threads > len(comb):
        chunk_size = len(comb)
    else:
        chunk_size = int((len(comb) // num_of_threads**0.5) + 1)

    chunks = [comb[i : i + chunk_size] for i in range(0, len(comb), chunk_size)]
    chunk_last_work_time_dict = {chunks.index(chunk): datetime.datetime.now() for chunk in chunks}

    last_verbose_time = [datetime.datetime.now()]  # Start time
    last_checkpoint_time = [datetime.datetime.now()]  # Start time
    
    # function to assign at threads
    def calculate_similarity_for_chunk(chunk):
        """calculates similarity in chunk by calling calculate_similarity for each pair

        Args:
            chunk (list): list of item pairs to calculate similarity

        Returns:
            list: currently not in use, could be later
        """
        return [
            calculate_similarity(
                pair,
                image_files.index(pair[0]),
                image_files.index(pair[1]),
                chunks.index(chunk) % num_of_threads,
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
                destination_container_folder,
                method,
                verbose=verbose-1
            )
            for pair in chunk
        ]

    # Threading (WILL CHANGE, LOTS OF SMALL COMPUTATIONS ARE NEEDED TO BE THREADED)
    results = thread_this(calculate_similarity_for_chunk, chunks)
    flat_results = [result for sublist in results for result in sublist]

    return image_similarities

# calculates similarities between given templates
def calculate_template_similarities(batch_idx, template_cluster_dict, template_files, verbose=0):
    """calculates similarity of templates

    Args:
        batch_idx (int): index of batch
        template_cluster_dict (dictionary): dictionary of template files and path of clusters they represent
        template_files (list): list of files
        verbose (int, optional): verbose level. Defaults to 0.

    Returns:
        dictionary: dictionary of image pairs similarity in that batch
    """
    if len(template_files) < 2:
        return {}

    template_similarities = {}
    templates = get_templates_dict(method, template_files, template_cluster_dict, scale, verbose=verbose-1)

    list_of_batch_templates = []
    for unique_path in set([os.path.split(x)[0] for x in template_cluster_dict.values()]):
        list_of_batch_templates.append([template for template, cluster_path in list(template_cluster_dict.items()) if os.path.split(cluster_path)[0] == unique_path])

    discarded_combs = []
    for l in list_of_batch_templates:
        if len(l) > 1:
            discarded_combs = discarded_combs + list(combinations(l, 2))

    comb = list(combinations(template_files, 2))  # all pair combinations of templates
    comb = list(set(comb).difference(set(discarded_combs)))
    bools = np.ones((len(template_files), len(template_files)), dtype=np.int8)  # row i means all (i, *) pairs, column j means all (*, j) pairs

    if len(comb) < 1:
        print_verbose("f", "no template pair combination pair found", verbose)
    if option != "merge":
        print_verbose("r", "processing total of " + str(len(comb)) + " pair combinations", verbose)
    if option == "merge":
        print_verbose("m", "processing total of " + str(len(comb)) + " pair combinations", verbose)

    # divide the computations to chunks
    if num_of_threads > len(comb):
        chunk_size = len(comb)
    else:
        chunk_size = int((len(comb) // num_of_threads**0.5) + 1)

    chunks = [comb[i : i + chunk_size] for i in range(0, len(comb), chunk_size)]
    chunk_last_work_time_dict = {chunks.index(chunk): datetime.datetime.now() for chunk in chunks}

    last_verbose_time = [datetime.datetime.now()]  # Start time
    last_checkpoint_time = [datetime.datetime.now()]  # Start time
    
    # function to assign at threads
    def calculate_similarity_for_chunk(chunk):
        """calculates similarity in chunk by calling calculate_similarity for each pair

        Args:
            chunk (list): list of item pairs to calculate similarity

        Returns:
            list: currently not in use, could be later
        """
        return [
            calculate_similarity(
                pair,
                template_files.index(pair[0]),
                template_files.index(pair[1]),
                chunks.index(chunk) % num_of_threads,
                lock,
                similarity_threshold,
                template_similarities,
                templates,
                bools,
                last_checkpoint_time,
                last_verbose_time,
                chunk_last_work_time_dict,
                batch_idx,
                chunk_time_threshold,
                destination_container_folder,
                method,
                verbose=verbose-1
            )
            for pair in chunk
        ]

    # Threading (WILL CHANGE, LOTS OF SMALL COMPUTATIONS ARE NEEDED TO BE THREADED)
    results = thread_this(calculate_similarity_for_chunk, chunks)
    flat_results = [result for sublist in results for result in sublist]

    return template_similarities

# function to merge batch folders clusters 
def merge_clusters_by_templates(batch_folder_paths, batch_idx, clustering_threshold, verbose=0):
    """merges individual clusters in all batch folders into one result folder

    Args:
        batch_folder_paths (list): list of batch folders path
        batch_idx (int): index of batch
        clustering_threshold (float): decides if 2 image is inside the same cluster of not
        verbose (int, optional): verbose level. Defaults to 0.

    Returns:
        list: list of merged clusters
    """
    # templates and their folders
    template_cluster_dict = {}
    for batch_folder in batch_folder_paths:
        for cluster_folder in os.listdir(batch_folder):
            if cluster_folder != "outliers":
                template_im = os.listdir(os.path.join(batch_folder, cluster_folder))[0]
                template_cluster_dict[template_im] = os.path.join(batch_folder, cluster_folder)
    all_template_files = sorted(list(template_cluster_dict.keys()))

    if option != "merge":
        print_verbose("r", str(len(template_cluster_dict)) + " template found", verbose)
    if option == "merge":
        print_verbose("m", str(len(template_cluster_dict)) + " template found", verbose)

    # compute all template similarities in one pass
    template_similarities = calculate_template_similarities(batch_idx, template_cluster_dict, all_template_files, verbose=verbose-1)

    # clustering templates according to similarities
    clusters, clustered_templates = cluster(sorted(template_similarities.items(), key=lambda x: x[1], reverse=True), clustering_threshold, verbose=verbose-1)

    # setting the folders for merging
    all_cluster_folder_paths = []
    for batch_folder in batch_folder_paths:
        for cluster_folder in os.listdir(batch_folder):
            if cluster_folder != "outliers":
                all_cluster_folder_paths.append([os.path.join(batch_folder, cluster_folder)])
    will_be_merged_clusters = []
    for c in clusters:
        folders_to_merge = [template_cluster_dict[t] for t in c]
        will_be_merged_clusters.append(folders_to_merge)
        for folder in folders_to_merge:
            all_cluster_folder_paths.remove([folder])
    outlier_folders = []
    for batch_folder in batch_folder_paths:
        for cluster_folder in os.listdir(batch_folder):
            if cluster_folder == "outliers":
                outlier_folders.append(os.path.join(batch_folder, cluster_folder))

    result_clusters = all_cluster_folder_paths + will_be_merged_clusters + [outlier_folders]

    return result_clusters

# function to pack all things above into one call
def process_images(batch_idx, image_files, destination_container_folder, verbose=0):
    """function to do the all processing

    Args:
        batch_idx (int): index of batch
        image_files (list): list of image files
        destination_container_folder (str): path to write files into
        verbose (int, optional): verbose level. Defaults to 0.
    """
    image_similarities = calculate_batch_similarity(batch_idx, image_files, method, verbose=verbose-1)

    clusters, clustered_images = cluster(sorted(image_similarities.items(), key=lambda x: x[1], reverse=True), clustering_threshold, verbose=verbose-1)
    outliers = list(set(image_files).difference(set(clustered_images)))
    print_verbose(batch_idx, str(len(clusters)) + " cluster found", verbose)
    write_clusters(clusters, batch_idx, images_folder_path, destination_container_folder, outliers, transfer, verbose=verbose-1)
    save_checkpoint(batch_idx, destination_container_folder, image_similarities, verbose=verbose-1)
    print("-"*70)


if __name__ == "__main__":
    if option != "merge":
        # Sort file paths by size
        all_image_files = filter(lambda x: os.path.isfile(os.path.join(images_folder_path, x)), os.listdir(images_folder_path))
        all_image_files = sorted(all_image_files, key=lambda x: os.stat(os.path.join(images_folder_path, x)).st_size)

        # process the images batch by batch
        for batch_idx, start in enumerate(range(0, len(all_image_files), batch_size)):
            image_files = all_image_files[start : start + batch_size]
            process_images(batch_idx, image_files, destination_container_folder, verbose=verbose-1)

        # if images are done in one batch terminate the code
        if batch_size >= len(all_image_files):
            for file in os.listdir(destination_container_folder):
                new_file_name = file.replace("batch_0", "result")
                os.rename(os.path.join(destination_container_folder, file),
                os.path.join(destination_container_folder, new_file_name))
            os.remove(os.path.join(destination_container_folder, "image_similarities_result.json"))
            print_verbose("f", "no merge needed to single batch", verbose)
        
    if option == "dontmerge":
        print_verbose("f", "terminating because of no merge request", verbose)

    # gets template(first) image from all clusters of all batches
    batch_folder_paths = sorted([os.path.join(destination_container_folder, f)
                                for f in os.listdir(destination_container_folder)
                                if os.path.isdir(os.path.join(destination_container_folder, f))])

    # process templates
    result_clusters = merge_clusters_by_templates(batch_folder_paths, "r", clustering_threshold, verbose=verbose-1)

    if option != "merge":
        print_verbose("r", str(len(result_clusters) - 1) + " cluster found at result", verbose)
    if option == "merge":
        print_verbose("m", str(len(result_clusters) - 1) + " cluster found at result", verbose)

    
    # creating result output folder and copying images
    result_folder_path = os.path.join(destination_container_folder, "results")
    os.mkdir(result_folder_path)
    for e, result_folder in enumerate(result_clusters):
        result_cluster_folder_path = os.path.join(result_folder_path, "cluster_" + str(e))
        if e == len(result_clusters) - 1:
            result_cluster_folder_path = os.path.join(result_folder_path, "outliers")

        os.mkdir(result_cluster_folder_path)
        for folder in result_folder:
            for file in os.listdir(folder):
                if transfer == "copy":
                    shutil.copy(os.path.join(folder, file), os.path.join(result_cluster_folder_path, file))
                if transfer == "move":
                    shutil.move(os.path.join(folder, file), os.path.join(result_cluster_folder_path, file))

    # removing unnecessary files and folders after merging results
    for folder in os.listdir(destination_container_folder):
        if folder != "results":
            if os.path.isdir(os.path.join(destination_container_folder, folder)):
                shutil.rmtree(os.path.join(destination_container_folder, folder))
            if os.path.isfile(os.path.join(destination_container_folder, folder)):
                os.remove(os.path.join(destination_container_folder, folder))
