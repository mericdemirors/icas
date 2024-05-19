import os
import shutil
import itertools

import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

import torch

from helper_exceptions import *
from helper_functions import write_clusters, print_verbose, image_transfer

class DL_Clusteror():
    def __init__(self, model_trainer, method: str, batch_size:int, number_of_clusters: list=[10], max_iter: list=[200],
                 DBSCAN_eps: list=[0.5], DBSCAN_min_samples: list=[5], HDBSCAN_min_cluster_size: list=[5],
                 HDBSCAN_max_cluster_size: list=[None], option: str="", transfer: str="copy",
                 overwrite: bool=False, verbose: int=0):
        """creates DL_Clusteror object

        Args:
            model_trainer (ModelTrainer): object to hold and manage deep learning model and dataset
            method (str): clustering method
            batch_size (int): batch size of clustering
            number_of_clusters (list, optional): list of number of clusters in dataset, used in parameter grid search. Defaults to [10].
            max_iter (list, optional): list of max_iter values, used in parameter grid search. Defaults to [200].
            DBSCAN_eps (list, optional): list of DBSCAN_eps values, used in parameter grid search. Defaults to [0.5].
            DBSCAN_min_samples (list, optional): list of DBSCAN_min_samples values, used in parameter grid search. Defaults to [5].
            HDBSCAN_min_cluster_size (list, optional): list of HDBSCAN_min_cluster_size values, used in parameter grid search. Defaults to [5].
            HDBSCAN_max_cluster_size (list, optional): list of HDBSCAN_max_cluster_size values, used in parameter grid search. Defaults to [None].
            option (str, optional): clustering option. Defaults to "".
            transfer (str, optional): file transferin option. Defaults to "copy".
            overwrite (bool, optional): permission to overwrite old clustered folder. Defaults to False.
            verbose (int, optional): verbose level. Defaults to 0.
        """
        self.model_trainer = model_trainer
        self.method = method
        self.batch_size = batch_size
        self.number_of_clusters = number_of_clusters
        self.max_iter = max_iter
        self.DBSCAN_eps = DBSCAN_eps
        self.DBSCAN_min_samples = DBSCAN_min_samples
        self.HDBSCAN_min_cluster_size = HDBSCAN_min_cluster_size
        self.HDBSCAN_max_cluster_size = HDBSCAN_max_cluster_size    
        self.option = option  
        self.transfer = transfer
        self.overwrite = overwrite
        self.verbose = verbose  
        
        if self.option == "merge":
            self.result_container_folder = os.path.abspath(self.model_trainer.dataset.root_dir)
        else:
            base_folder, images_folder_name = os.path.split(os.path.abspath(self.model_trainer.dataset.root_dir))
            self.result_container_folder = os.path.join(base_folder, images_folder_name + "_clustered")

        self.arguman_check(verbose=verbose-1)
    
    def __str__(self, verbose: int=0):
        """casting to string method for printing/debugging object attributes

        Args:
            verbose (int, optional): verbose level. Defaults to 0.

        Returns:
            str: object attribute information
        """
        attributes = vars(self)
        attr_strings = [f"{key}: {value}" for key, value in attributes.items()]
        return "-"*70 + "\n" + "\n".join(attr_strings) + "\n" + "-"*70

    # checks arguman validity
    def arguman_check(self, verbose: int=0):
        """checks arguman validity

        Args:
            verbose (int, optional): verbose level. Defaults to 0.
        """
        valid_methods = ["kmeans", "hierarchy", "DBSCAN", "gaussian", "HDBSCAN"]
        valid_options = ["merge", "dontmerge", ""]
        valid_transfer = ["copy", "move"]
        
        if self.method not in valid_methods:
            raise(InvalidMethodException("Invalid method: " + self.method))
        if self.option not in valid_options:
            raise(InvalidOptionException("Invalid option: " + self.option))
        if self.transfer not in valid_transfer:
            raise(InvalidTransferException("Invalid transfer: " + self.transfer))

    # get clustering models
    def get_models(self, verbose: int=0):
        """creates different models for parameter grid search

        Args:
            verbose (int, optional): verbose level. Defaults to 0.
        """
        def calculate_grid_search(verbose: int=0):
            """creates parameters for grid search models

            Args:
                verbose (int, optional): verbose level. Defaults to 0.

            Returns:
                list: list of parameters
            """
            param_grid = []
            if self.method == "kmeans":
                param_grid = list(itertools.product(self.number_of_clusters, self.max_iter))
            elif self.method == "hierarchy":
                param_grid = self.number_of_clusters
            elif self.method == "DBSCAN":
                param_grid = list(itertools.product(self.DBSCAN_eps, self.DBSCAN_min_samples))
            elif self.method == "gaussian":
                param_grid = list(itertools.product(self.number_of_clusters, self.max_iter))
            elif self.method == "HDBSCAN":
                param_grid = list(itertools.product(self.HDBSCAN_min_cluster_size, self.HDBSCAN_max_cluster_size))
            
            return param_grid
        param_grid = calculate_grid_search(verbose-1)

        models = []
        for params in param_grid:
            if self.method == "kmeans":
                models.append(KMeans(n_clusters=params[0], max_iter=params[1], verbose=max(0,verbose-1)))
            elif self.method == "hierarchy":
                models.append(AgglomerativeClustering(n_clusters=params))
            elif self.method == "DBSCAN":
                models.append(DBSCAN(eps=params[0], min_samples=params[1]))
            elif self.method == "gaussian":
                models.append(GaussianMixture(n_components=params[0], max_iter=params[1], verbose=max(0,verbose-1)))
            elif self.method == "HDBSCAN":
                models.append(HDBSCAN(min_cluster_size=params[0], max_cluster_size=params[1]))

        return models

    # finds best clustering model with parameter grid search
    def find_best_model(self, models: list, image_embeds, verbose: int=0):
        """finds best model in grid search by clustering evaluations

        Args:
            models (list): list of clustering models with different parameters
            image_embeds (numpy.ndarray): images latent representations
            verbose (int, optional): verbose level. Defaults to 0.

        Returns:
            sklearn clusterin model: model with best evaluation score
        """
        if len(models) == 1:
            return models[0]
        
        # calculate the silhouette, davies_bouldin and calinski_harabasz scores
        # maximize the silhouette and calinski_harabasz scores, minimize the davies_bouldin score
        silhouette_scores, db_scores, ch_scores = [], [], []
        for model in tqdm(models, desc=f"Finding best {self.method} parameters", leave=False):
            labels = model.fit_predict(image_embeds)
            
            silhouette_scores.append(silhouette_score(image_embeds, labels))
            db_scores.append(davies_bouldin_score(image_embeds, labels))
            ch_scores.append(calinski_harabasz_score(image_embeds, labels))

        silhouette_scores, db_scores, ch_scores = np.array(silhouette_scores), np.array(db_scores), np.array(ch_scores)
        silhouette_scores = (silhouette_scores - silhouette_scores.min())/(silhouette_scores.max()-silhouette_scores.min())
        db_scores = (db_scores - db_scores.min())/(db_scores.max()-db_scores.min())
        ch_scores = (ch_scores - ch_scores.min())/(ch_scores.max()-ch_scores.min())
        
        combined_scores = silhouette_scores - db_scores + ch_scores
        best_model = models[np.argmax(combined_scores)]
        
        return best_model
    
    # calculates clusters from passed batch images
    def calculate_batch_clusters(self, start: int, end: int, verbose: int=0):
        """calculates the clusters in a batch

        Args:
            start (int): index of first image in batch
            end (int): index of last image in batch
            verbose (int, optional): verbose level. Defaults to 0.

        Returns:
            list: list of clusters
        """
        features = self.model_trainer.get_features(start, end)
        paths = list(features.keys())
        image_embeds = np.array(list(features.values()))

        models = self.get_models(verbose=verbose-1)
        best_model = self.find_best_model(models, image_embeds, verbose=verbose-1)
        labels = best_model.fit_predict(image_embeds)
        clusters = [[paths[i] for i in range(len(paths)) if labels[i] == id] for id in set(labels)]
        return clusters

    # calculates clusters from selected template images
    def calculate_template_clusters(self, template_paths: list, verbose: int=0):
        """calculates clusters of templates

        Args:
            template_paths (list): list of template images
            verbose (int, optional): verbose level. Defaults to 0.

        Returns:
            list: list of clusters
        """
        features = {}
        for tp in tqdm(template_paths, desc="Getting template features", leave=False):
            image = self.model_trainer.dataset.read_image(tp)
            image = image[np.newaxis, ...]
            tensor_image = torch.from_numpy(image).to(self.model_trainer.device)
            features[tp] = self.model_trainer.model.embed(tensor_image)

        paths = list(features.keys())
        # transforming output tensors to numpy for clustering
        tensor_values = list(features.values()) 
        numpy_values = [t.cpu().detach().numpy() for t in tensor_values]
        numpy_values = np.array(numpy_values)
        image_embeds = np.squeeze(numpy_values, axis=1)

        models = self.get_models(verbose=verbose-1)
        best_model = self.find_best_model(models, image_embeds, verbose=verbose-1)
        labels = best_model.fit_predict(image_embeds)
        
        clusters = [[paths[i] for i in range(len(paths)) if labels[i] == id] for id in set(labels)]
        return clusters

    # merges clustered templates
    def merge_clusters_by_templates(self, batch_folder_paths: list, verbose: int=0):
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
            print_verbose("r", str(len(template_cluster_dict)) + " template found", verbose=verbose-1)
        if self.option == "merge":
            print_verbose("m", str(len(template_cluster_dict)) + " template found", verbose=verbose-1)

        # compute all template similarities in one pass
        template_paths = [os.path.abspath(os.path.join(template_cluster_dict[file], file)) for file in all_template_files]
        clusters = self.calculate_template_clusters(template_paths, verbose=verbose-1)

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
    def create_clusters(self, batch_idx: int, start: int, end: int, verbose: int=0):
        """creates clusters of a batch of images

        Args:
            batch_idx (int): batch id
            start (int): index of first image in batch
            end (int): index of last image in batch
            verbose (int, optional): verbose level. Defaults to 0.
        """
        clusters = self.calculate_batch_clusters(start, end, verbose=verbose-1)
        write_clusters(clusters, batch_idx, self.result_container_folder, [], self.transfer, verbose=verbose-1)
        
        if verbose > 0:
            print("-"*70)

    # full process in one function
    def process(self, verbose: int=0):
        """function to encapsulate all pipeline in one function

        Args:
            verbose (int, optional): verbose level. Defaults to 0.
        """
        # if model is raw train it
        if self.model_trainer.ckpt_path is None:
            self.model_trainer.train()

        # creating result folder
        if self.option != "merge":
            if os.path.exists(self.result_container_folder) and not self.overwrite:
                raise(OverwritePermissionException("Overwriting permission not granted to overwrite " + self.result_container_folder))
            else:
                if os.path.exists(self.result_container_folder):
                    shutil.rmtree(self.result_container_folder)
                os.makedirs(self.result_container_folder)

        if self.option != "merge":
            for batch_idx, start in tqdm(enumerate(range(0, len(self.model_trainer.dataset), self.batch_size)), desc="Creating clusters", leave=False):
                self.create_clusters(batch_idx, start, start + self.batch_size, verbose=self.verbose-1)

            # if images are done in one batch terminate the code after organizing result folders
            if self.batch_size >= len(self.model_trainer.dataset):
                for file in os.listdir(self.result_container_folder):
                    new_file_name = file.replace("batch_0", "result")
                    os.rename(os.path.join(self.result_container_folder, file), os.path.join(self.result_container_folder, new_file_name))
                print_verbose("f", "no merge needed to single batch", verbose=verbose-1)
            
        if self.option == "dontmerge":
            print_verbose("f", "finishing because of no merge request", verbose=verbose-1)


        # gets each batchs folder
        batch_folder_paths = sorted([os.path.join(self.result_container_folder, f)
                                    for f in os.listdir(self.result_container_folder)
                                    if os.path.isdir(os.path.join(self.result_container_folder, f))])

        # merge each batch to get which clusters folders should be merged together
        template_cluster_folders_to_merge_list = self.merge_clusters_by_templates(batch_folder_paths, verbose=verbose-1)

        if self.option != "merge":
            print_verbose("r", str(len(template_cluster_folders_to_merge_list) - 1) + " cluster found at result", verbose=verbose-1)
        if self.option == "merge":
            print_verbose("m", str(len(template_cluster_folders_to_merge_list) - 1) + " cluster found at result", verbose=verbose-1)
        
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
                    image_transfer(self.transfer, os.path.join(template_cluster_folder, file), os.path.join(cluster_folder_path, file))
                    
        # removing unnecessary files and folders after merging results
        for folder in os.listdir(self.result_container_folder):
            if folder != "results":
                if os.path.isdir(os.path.join(self.result_container_folder, folder)):
                    shutil.rmtree(os.path.join(self.result_container_folder, folder))
                if os.path.isfile(os.path.join(self.result_container_folder, folder)):
                    os.remove(os.path.join(self.result_container_folder, folder))

    # call method to capsulate process function and custom exceptions
    def __call__(self):
        """calling the object will start the main process

        Args:
            verbose (int, optional): _description_. Defaults to 0.
        """
        self.process(verbose=self.verbose-1)