import os
import shutil

from helper_functions import generate_test_dataset
from clustering import Clustering

test_path = os.path.join(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0], "test_images")

for method in ["SSIM", "TM", "imagehash", "minhash", "ORB"]:
    for transfer in ["copy", "move"]:
        # generate test dataset
        generate_test_dataset(test_path, 25)

        # create first Clustering object to cluster but dont merge test images
        clstr1 = Clustering(images_folder_path=test_path, method=method, num_of_threads=8, threshold=0.125,
                        batch_size=13, overwrite="Y", transfer=transfer, verbose=1, size=(100,100), option="dontmerge")
        clstr1.interactive_threshold_selection()
        clstr1()

        # create second Clustering object to merge clustered test images
        clstr2 = Clustering(images_folder_path=test_path + "_clustered", method=method, num_of_threads=8, threshold=0.125,
                        batch_size=13, overwrite="Y", transfer=transfer, verbose=1, scale=(1,1), option="merge")
        clstr2()

        shutil.rmtree(test_path)
        shutil.rmtree(test_path + "_clustered")
        print("Test passed, evidence destroyed.")