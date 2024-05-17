import os
import shutil
import itertools
import numpy as np
from tqdm import tqdm

from helper_exceptions import *
from helper_functions import write_clusters, print_verbose

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

class DL_Clustering():
    def __init__(self, model_trainer, method, batch_size, number_of_clusters=[10], max_iter=[200], DBSCAN_eps=[0.5],
                DBSCAN_min_samples=[5], HDBSCAN_min_cluster_size=[5], HDBSCAN_max_cluster_size=[None], option="",
                transfer="copy", overwrite=False, verbose=0):
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
            self.result_container_folder = self.model_trainer.dataset.root_dir
        else:
            base_folder, images_folder_name = os.path.split(self.model_trainer.dataset.root_dir)
            self.result_container_folder = os.path.join(base_folder, images_folder_name + "_clustered")

        self.arguman_check(verbose=verbose-1)
    
    def __str__(self, verbose=0):
        """casting to string method for printing/debugging object attributes

        Args:
            verbose (int, optional): verbose level. Defaults to 0.

        Returns:
            str: object attribute information
        """
        attributes = vars(self)
        attr_strings = [f"{key}: {value}" for key, value in attributes.items()]
        return "-"*70 + "\n" + "\n".join(attr_strings) + "\n" + "-"*70

    def arguman_check(self, verbose=0):
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

    def get_models(self, verbose=0):
        """creates different models for parameter grid search

        Args:
            verbose (int, optional): verbose level. Defaults to 0.
        """
        def calculate_grid_search(verbose=0):
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

    def find_best_model(self, models, image_embeds, verbose=0):
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
    
    def calculate_batch_clusters(self, start, end, verbose=0):
        features = self.model_trainer.get_features(start, end)
        paths = list(features.keys())
        image_embeds = np.array(list(features.values()))

        models = self.get_models(verbose=verbose-1)
        best_model = self.find_best_model(models, image_embeds, verbose=verbose-1)
        labels = best_model.fit_predict(image_embeds)
        return paths, labels

    def calculate_template_similarity(self, verbose=0):
        """will be added for clustering template images

        Args:
            verbose (int, optional): verbose level. Defaults to 0.
        """
        pass
    def merge_clusters_my_templates(self, verbose=0):
        """will be added for clustering template images

        Args:
            verbose (int, optional): verbose level. Defaults to 0.
        """
        pass


    def create_clusters(self, batch_idx, start, end, verbose=0):
        """creates clusters of a batch of images

        Args:
            batch_idx (int): batch id
            start (int): index of first image in batch
            end (int): index of last image in batch
            verbose (int, optional): verbose level. Defaults to 0.
        """
        paths, labels = self.calculate_batch_clusters(start, end, verbose=verbose-1)
        clusters = [[paths[i] for i in range(len(paths)) if labels[i] == id] for id in set(labels)]
        write_clusters(clusters, batch_idx, self.result_container_folder, [], self.transfer, verbose=verbose-1)
        
        if verbose > 0:
            print("-"*70)

    def process(self, verbose=0):
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
            for batch_idx, start in enumerate(range(0, len(self.model_trainer.dataset), self.batch_size)):
                self.create_clusters(batch_idx, start, start + self.batch_size, verbose=self.verbose-1)

            # if images are done in one batch terminate the code after organizing result folders
            if self.batch_size >= len(self.model_trainer.dataset):
                for file in os.listdir(self.result_container_folder):
                    new_file_name = file.replace("batch_0", "result")
                    os.rename(os.path.join(self.result_container_folder, file), os.path.join(self.result_container_folder, new_file_name))
                print_verbose("f", "no merge needed to single batch", verbose=verbose-1)
            
        if self.option == "dontmerge":
            print_verbose("f", "finishing because of no merge request", verbose=verbose-1)

        # TODO merging templates will be added

    def __call__(self, verbose=0):
        """calling the object will start the main process and catch any possible exception during

        Args:
            verbose (int, optional): _description_. Defaults to 0.
        """
        try:
            self.process(verbose=verbose-1)
        except (ErrorException, WrongTypeException, InvalidMethodException, InvalidOptionException, 
                InvalidTransferException, OverwritePermissionException, InvalidLossException) as custom_e:
            print(custom_e.message)
            exit(custom_e.error_code)
        except FinishException as fe:
            print(fe.message)