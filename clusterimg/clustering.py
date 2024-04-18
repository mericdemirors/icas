import os
import shutil
import datetime

import cv2
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
import mplcursors

from itertools import combinations
from threading import Lock

from helper_functions import cluster, calculate_similarity, print_verbose, thread_this, save_checkpoint, write_clusters, get_image_features
from helper_exceptions import *
from global_variables import GLOBAL_THREADS, GLOBAL_THRESHOLD

class Clustering():
    def __init__(self, images_folder_path: str, method: str, threshold: float, batch_size: int, num_of_threads: int=2, size: tuple=(0, 0), scale: tuple=(1.0, 1.0), option: str="", transfer: str="copy", overwrite: bool=False, chunk_time_threshold: int=60, verbose: int=0):
        """initializing clustering object

        Args:
            images_folder_path (str): folder path of images
            method (str): method to calculate similarity, decides features
            threshold (float): decides if X and Y are close or not
            batch_size (int): number of images in each process batch
            num_of_threads (int, optional): number of threads to share threaded jobs. Defaults to 2.
            size (tuple, optional): dsize parameters for cv2.resize. Defaults to (0, 0).
            scale (tuple, optional): fx and fy parameters for cv2.resize. Defaults to (1.0, 1.0).
            option (str, optional): decides process type. Defaults to "".
            transfer (str, optional): transfer type of images. Defaults to "copy".
            overwrite (bool, optional): permission to overwrite old _clustered folder. Defaults to False.
            chunk_time_threshold (int, optional): inactivation time limit for checkpoint saving. Defaults to 60.
            verbose (int, optional): verbose level. Defaults to 0.
        """
        global GLOBAL_THRESHOLD, GLOBAL_THREADS
        self.images_folder_path = images_folder_path
        self.method = method
        self.threshold = threshold
        self.batch_size = batch_size
        self.num_of_threads = num_of_threads
        self.scale = scale
        self.size = size
        self.option = option
        self.transfer = transfer
        self.overwrite = overwrite
        self.chunk_time_threshold = chunk_time_threshold
        self.verbose = verbose
        GLOBAL_THRESHOLD = self.threshold
        GLOBAL_THREADS = self.num_of_threads
        self.lock = Lock()

        if self.option == "merge":
            self.result_container_folder = self.images_folder_path
        else:
            base_folder, images_folder_name = os.path.split(self.images_folder_path)
            self.result_container_folder = os.path.join(base_folder, images_folder_name + "_clustered")

    def __str__(self):
        """casting to string method for printing/debugging object attributes

        Returns:
            str: object attribute information
        """
        attributes = vars(self)
        attr_strings = [f"{key}: {value}" for key, value in attributes.items()]
        return "-"*70 + "\n" + "\n".join(attr_strings) + "\n" + "-"*70

    def arguman_check(self, verbose=0):
        """checks validity of object initialization parameters

        Args:
            verbose (int, optional): verbose level. Defaults to 0.
        """
        valid_methods = ["SSIM", "minhash", "imagehash", "ORB", "TM"]
        valid_options = ["merge", "dontmerge", ""]
        valid_transfer = ["copy", "move"]

        if self.method not in valid_methods:
            print_verbose("e", "invalid method type", verbose)
        if self.option not in valid_options:
            print_verbose("e", "invalid option type", verbose)
        if self.transfer not in valid_transfer:
            print_verbose("e", "invalid transfer type", verbose)

    # TODO interactive threshold selection is disabled because of "FigureCanvasAgg is non-interactive, and thus cannot be shown" error"
    def select_threshold(self, method, folder_path, num_of_files=1000, verbose=0):
        """Lets user interactively select threshold

        Args:
            method (str): type of similarity calculation method
            folder_path (str): path to image folder
            num_of_files (int, optional): how much image should be processed for interactive selection. Defaults to 1000.
            verbose (int, optional): verbose level. Defaults to 0.

        Returns:
            None: if method is SSIM(structural_similarity) no interactive selection can be made(takes to much time) so return is used for terminating the function
        """
        # sort file names by size
        all_image_files = filter(lambda x: os.path.isfile(os.path.join(folder_path, x)), os.listdir(folder_path))
        selected_image_files = sorted(all_image_files, key=lambda x: os.stat(os.path.join(folder_path, x)).st_size)[:num_of_files]

        image_feature_dict = get_image_features(method, selected_image_files, folder_path, self.size, self.scale, verbose=verbose-1)
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
            sim_list = [mh1.jaccard(mh2) for (mh1, mh2) in tqdm(minhashs_combs, desc="Calculating minhash similarity", leave=False)]

        elif method == "imagehash":
            file_combs = list(combinations(list(image_feature_dict.keys()), 2))
            sim_list = [1 - (image_feature_dict[f1] - image_feature_dict[f2]) / len(image_feature_dict[f1].hash) for (f1, f2) in tqdm(file_combs, desc="Calculating imagehash similarity", leave=False)]
        
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
            global GLOBAL_THRESHOLD
            if event.button == 1:
                self.threshold = event.ydata
                plt.close()

        y = sorted(sim_list, reverse=True)
        plt.figure(figsize=(8, 4))
        plt.plot(range(len(y)), y)    
        cursor = mplcursors.cursor(hover=True)
        cursor.connect("add", on_hover)
        plt.gcf().canvas.mpl_connect('button_press_event', on_click)
        plt.show() 

    # calculates similarities between given images
    def calculate_batch_similarity(self, batch_idx, image_paths, verbose=0):
        """calculates similarity inside a batch

        Args:
            batch_idx (int): index of batch
            image_paths (list): list of image paths
            verbose (int, optional): verbose level. Defaults to 0.
            
        Returns:
            dictionary: dictionary of image pairs similarity in that batch
        """
        if len(image_paths) < 2:
            return {}
        
        # get image features
        images = get_image_features(self.method, image_paths, self.size, self.scale, verbose=verbose-1)

        comb = list(combinations(image_paths, 2))  # all pair combinations of images
        bools = np.ones((len(image_paths), len(image_paths)), dtype=np.int8)  # row i means all (i, *) pairs, column j means all (*, j) pairs
        print_verbose(batch_idx, "processing total of " + str(len(comb)) + " pair combinations", verbose)

        # divide the computations to chunks
        if self.num_of_threads > len(comb):
            chunk_size = len(comb)
        else:
            chunk_size = int((len(comb) // self.num_of_threads**0.5) + 1)

        chunks = [comb[i : i + chunk_size] for i in range(0, len(comb), chunk_size)]
        chunk_last_work_time_dict = {chunks.index(chunk): datetime.datetime.now() for chunk in chunks}

        last_verbose_time = [datetime.datetime.now()]  # Start time
        last_checkpoint_time = [datetime.datetime.now()]  # Start time
        
        image_similarities = {}
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
                    image_paths.index(pair[0]),
                    image_paths.index(pair[1]),
                    chunks.index(chunk) % self.num_of_threads,
                    self.lock,
                    self.threshold,
                    image_similarities,
                    images,
                    bools,
                    last_checkpoint_time,
                    last_verbose_time,
                    chunk_last_work_time_dict,
                    batch_idx,
                    self.chunk_time_threshold,
                    self.result_container_folder,
                    self.method,
                    verbose=verbose-1
                )
                for pair in chunk
            ]

        # Threading (WILL CHANGE, LOTS OF SMALL COMPUTATIONS ARE NEEDED TO BE THREADED)
        results = thread_this(calculate_similarity_for_chunk, chunks)
        flat_results = [result for sublist in results for result in sublist]

        return image_similarities

    # calculates similarities between given templates
    def calculate_template_similarities(self, template_paths, verbose=0):
        """calculates similarity of templates

        Args:
            template_paths (list): list of template file paths
            verbose (int, optional): verbose level. Defaults to 0.

        Returns:
            dictionary: dictionary of image pairs similarity in that batch
        """
        if len(template_paths) < 2:
            return {}

        # get template features
        templates = get_image_features(self.method, template_paths, self.size, self.scale, verbose=verbose-1)

        # discarding the template combinations that were in the same batch
        batch_cluster_template_list = [file.split(os.sep)[-3:] for file in template_paths]
        
        # get the templates from same batch folder
        same_batch_templates = []
        for b in set([bct[0] for bct in batch_cluster_template_list]):
            same_batch_templates.append([file for e, file in enumerate(template_paths) if batch_cluster_template_list[e][0] == b])

        # and discard them to decrease number of similarity checks
        discarded_combs = []
        for l in same_batch_templates:
            if len(l) > 1:
                discarded_combs = discarded_combs + list(combinations(l, 2))

        comb = list(combinations(template_paths, 2))  # all pair combinations of templates
        comb = list(set(comb).difference(set(discarded_combs)))
        bools = np.ones((len(template_paths), len(template_paths)), dtype=np.int8)  # row i means all (i, *) pairs, column j means all (*, j) pairs

        if len(comb) < 1:
            print_verbose("f", "no template pair combination pair found", verbose)
        if self.option != "merge":
            print_verbose("r", "processing total of " + str(len(comb)) + " pair combinations", verbose)
        if self.option == "merge":
            print_verbose("m", "processing total of " + str(len(comb)) + " pair combinations", verbose)

        # divide the computations to chunks
        if self.num_of_threads > len(comb):
            chunk_size = len(comb)
        else:
            chunk_size = int((len(comb) // self.num_of_threads**0.5) + 1)

        chunks = [comb[i : i + chunk_size] for i in range(0, len(comb), chunk_size)]
        chunk_last_work_time_dict = {chunks.index(chunk): datetime.datetime.now() for chunk in chunks}

        last_verbose_time = [datetime.datetime.now()]  # Start time
        last_checkpoint_time = [datetime.datetime.now()]  # Start time
        
        template_similarities = {}
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
                    template_paths.index(pair[0]),
                    template_paths.index(pair[1]),
                    chunks.index(chunk) % self.num_of_threads,
                    self.lock,
                    self.threshold,
                    template_similarities,
                    templates,
                    bools,
                    last_checkpoint_time,
                    last_verbose_time,
                    chunk_last_work_time_dict,
                    "r",
                    self.chunk_time_threshold,
                    self.result_container_folder,
                    self.method,
                    verbose=verbose-1
                )
                for pair in chunk
            ]

        # Threading (WILL CHANGE, LOTS OF SMALL COMPUTATIONS ARE NEEDED TO BE THREADED)
        results = thread_this(calculate_similarity_for_chunk, chunks)
        flat_results = [result for sublist in results for result in sublist]

        return template_similarities

    # function to merge batch folders clusters
    def merge_clusters_by_templates(self, batch_folder_paths, verbose=0):
        """merges individual clusters in all batch folders into one result folder

        Args:
            batch_folder_paths (list): list of batch folders path
            verbose (int, optional): verbose level. Defaults to 0.

        Returns:
            list: list of merged clusters
        """
        # get one template from all clusters at each batch
        template_cluster_dict = {}
        for batch_folder in batch_folder_paths:
            for cluster_folder in os.listdir(batch_folder):
                if cluster_folder != "outliers":
                    template_im = os.listdir(os.path.join(batch_folder, cluster_folder))[0]
                    template_cluster_dict[template_im] = os.path.join(batch_folder, cluster_folder)
        all_template_files = sorted(list(template_cluster_dict.keys()))

        if self.option != "merge":
            print_verbose("r", str(len(template_cluster_dict)) + " template found", verbose)
        if self.option == "merge":
            print_verbose("m", str(len(template_cluster_dict)) + " template found", verbose)

        # compute all template similarities in one pass
        template_paths = [os.path.join(template_cluster_dict[file], file) for file in all_template_files]
        template_similarities = self.calculate_template_similarities(template_paths, verbose=verbose-1)

        # clustering templates according to similarities
        clusters, clustered_templates = cluster(sorted(template_similarities.items(), key=lambda x: x[1], reverse=True), self.threshold, verbose=verbose-1)

        # setting the folders for merging
        all_cluster_folder_paths = []
        for batch_folder in batch_folder_paths:
            for cluster_folder in os.listdir(batch_folder):
                if cluster_folder != "outliers":
                    all_cluster_folder_paths.append([os.path.join(batch_folder, cluster_folder)])
        will_be_merged_clusters = []
        for c in clusters:
            folders_to_merge = [os.path.split(t)[0] for t in c]
            will_be_merged_clusters.append(folders_to_merge)
            for folder in folders_to_merge:
                all_cluster_folder_paths.remove([folder])
        outlier_folders = []
        for batch_folder in batch_folder_paths:
            for cluster_folder in os.listdir(batch_folder):
                if cluster_folder == "outliers":
                    outlier_folders.append(os.path.join(batch_folder, cluster_folder))

        template_cluster_folders_to_merge_list = all_cluster_folder_paths + will_be_merged_clusters + [outlier_folders]

        return template_cluster_folders_to_merge_list

    # function to pack all things above into one call
    def create_clusters(self, batch_idx, image_files, verbose=0):
        """function to do the all processing

        Args:
            batch_idx (int): index of batch
            image_files (list): list of image files
            verbose (int, optional): verbose level. Defaults to 0.
        """
        image_paths = [os.path.join(self.images_folder_path, file) for file in image_files]
        image_similarities = self.calculate_batch_similarity(batch_idx, image_paths, verbose=verbose-1)

        clusters, clustered_images = cluster(sorted(image_similarities.items(), key=lambda x: x[1], reverse=True), self.threshold, verbose=verbose-1)
        outliers = list(set(image_paths).difference(set(clustered_images)))
        write_clusters(clusters, batch_idx, self.result_container_folder, outliers, self.transfer, verbose=verbose-1)
        save_checkpoint(batch_idx, self.result_container_folder, image_similarities, verbose=verbose-1)
        print("-"*70)

    # full process in one function
    def process(self):
        """function to capsulate all pipeline in one call
        """
        self.arguman_check(self.verbose-1)

        # creating result folder
        if self.option != "merge":
            if os.path.exists(self.result_container_folder) and not self.overwrite:
                print_verbose("e", "no permission to overwrite", self.verbose)
            else:
                if os.path.exists(self.result_container_folder):
                    shutil.rmtree(self.result_container_folder)
                os.makedirs(self.result_container_folder)

        if self.option != "merge":
            # Sort images by size
            all_image_files = filter(lambda x: os.path.isfile(os.path.join(self.images_folder_path, x)), os.listdir(self.images_folder_path))
            all_image_files = sorted(all_image_files, key=lambda x: os.stat(os.path.join(self.images_folder_path, x)).st_size)

            # process the images batch by batch
            for batch_idx, start in enumerate(range(0, len(all_image_files), self.batch_size)):
                image_files = all_image_files[start : start + self.batch_size]
                self.create_clusters(batch_idx, image_files, verbose=self.verbose-1)

            # if images are done in one batch terminate the code after organizing result folders
            if self.batch_size >= len(all_image_files):
                for file in os.listdir(self.result_container_folder):
                    new_file_name = file.replace("batch_0", "result")
                    os.rename(os.path.join(self.result_container_folder, file), os.path.join(self.result_container_folder, new_file_name))
                os.remove(os.path.join(self.result_container_folder, "image_similarities_result.json"))
                print_verbose("f", "no merge needed to single batch", self.verbose)
            
        if self.option == "dontmerge":
            print_verbose("f", "finishing because of no merge request", self.verbose)

        # gets each batchs folder
        batch_folder_paths = sorted([os.path.join(self.result_container_folder, f)
                                    for f in os.listdir(self.result_container_folder)
                                    if os.path.isdir(os.path.join(self.result_container_folder, f))])

        # merge each batch to get which clusters folders should be merged together
        template_cluster_folders_to_merge_list = self.merge_clusters_by_templates(batch_folder_paths, verbose=self.verbose-1)

        if self.option != "merge":
            print_verbose("r", str(len(template_cluster_folders_to_merge_list) - 1) + " cluster found at result", self.verbose)
        if self.option == "merge":
            print_verbose("m", str(len(template_cluster_folders_to_merge_list) - 1) + " cluster found at result", self.verbose)
        
        # creating result folder and merging cluster folders
        result_folder_path = os.path.join(self.result_container_folder, "results")
        os.mkdir(result_folder_path)
        for e, template_cluster_folders_to_merge in enumerate(template_cluster_folders_to_merge_list):
            cluster_folder_path = os.path.join(result_folder_path, "cluster_" + str(e))
            if e == len(template_cluster_folders_to_merge_list) - 1:
                cluster_folder_path = os.path.join(result_folder_path, "outliers")

            os.mkdir(cluster_folder_path)
            for template_cluster_folder in template_cluster_folders_to_merge:
                for file in os.listdir(template_cluster_folder):
                    if self.transfer == "copy":
                        shutil.copy(os.path.join(template_cluster_folder, file), os.path.join(cluster_folder_path, file))
                    if self.transfer == "move":
                        shutil.move(os.path.join(template_cluster_folder, file), os.path.join(cluster_folder_path, file))

        # removing unnecessary files and folders after merging results
        for folder in os.listdir(self.result_container_folder):
            if folder != "results":
                if os.path.isdir(os.path.join(self.result_container_folder, folder)):
                    shutil.rmtree(os.path.join(self.result_container_folder, folder))
                if os.path.isfile(os.path.join(self.result_container_folder, folder)):
                    os.remove(os.path.join(self.result_container_folder, folder))

    # call method to capsulate process function and custom exceptions
    def __call__(self):
        """calling the object will start the main process and catch any possible exception during
        """
        try:
            self.process()
        except FinishException as fe:
            print(fe.message)
        except ErrorException as ee:
            print(ee.message)
            exit(ee.error_code)
        except WrongTypeException as wte:
            print(wte.message)
            exit(wte.error_code)