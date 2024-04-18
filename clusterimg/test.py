import os
import shutil

from helper_functions import generate_test_dataset
from clustering import Clustering

test_path = os.path.join(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0], "test_images")
generate_test_dataset(test_path, 100)

clstr = Clustering(images_folder_path=test_path, method="minhash", num_of_threads=8, threshold=0.20,
                   batch_size=50, overwrite="Y", transfer="copy", verbose=6, size=(100,100), option="dontmerge")
print(clstr)
clstr()

clstr = Clustering(images_folder_path=test_path + "_clustered", method="minhash", num_of_threads=8, threshold=0.20,
                   batch_size=50, overwrite="Y", transfer="copy", verbose=6, scale=(1,1), option="merge")
print(clstr)
clstr()

shutil.rmtree(test_path)
shutil.rmtree(test_path + "_clustered")
print("Test passed, evidence destroyed.")