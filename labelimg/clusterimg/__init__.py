from .Clusteror import Clusteror
try:
    from .DL_Datasets import * # if torch
    from .DL_Models import * # if torch
    from .DL_ModelTrainer import ModelTrainer # if torch and torchvision
    from .DL_Clusteror import DL_Clusteror # if torch
except Exception as e:
    print("Exception during 'DL' clusterimg file imports. Skipping deep learning clusterimg")
    print(e)

try:
    from .clusterimg_test import clusterimg_test
    from .DL_clusterimg_test import DL_clusterimg_test
except Exception as e: 
    print("Exception during 'test' file imports. Skipping test files")
    print(e)
