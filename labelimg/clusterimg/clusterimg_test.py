import os
import shutil

from .helper_functions import generate_test_dataset
from .helper_exceptions import FinishException
from .Clusteror import Clusteror

def clusterimg_test():
    test_path = os.path.join(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0], "test_images")

    for method in ["SSIM", "TM", "imagehash", "minhash", "ORB"]:
        for transfer in ["copy", "move"]:
            # generate test dataset
            generate_test_dataset(test_path, 25)

            # create first Clusteror object to cluster but dont merge test images
            clstr1 = Clusteror(images_folder_path=test_path, method=method, num_of_threads=8, threshold=0.9,
                            batch_size=13, overwrite="Y", transfer=transfer, verbose=1, size=(100,100), option="dontmerge")
            #clstr1.interactive_threshold_selection()
            try:
                clstr1()
            except FinishException as e:
                pass

            # create second Clusteror object to merge clustered test images
            clstr2 = Clusteror(images_folder_path=test_path + "_clustered", method=method, num_of_threads=8, threshold=0.9,
                            batch_size=13, overwrite="Y", transfer=transfer, verbose=1, scale=(1,1), option="merge")
            try:
                clstr2()
            except FinishException as e:
                pass

            shutil.rmtree(test_path)
            shutil.rmtree(test_path + "_clustered")
            print(f"Test passed for {method} with {transfer}.")

if __name__ == "__main__":
    clusterimg_test()