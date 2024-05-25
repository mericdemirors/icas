# Contribute to the whole system pipeline with:

## * adding new method to Clusterimg:

* 1- add needed parameters to Clusteror.\_\_init\_\_()
* 2- add method name to valid_methods in Clusteror.arguman_check()
* 3- add feature extraction function to helper_functions.get_image_features()
* 4- add similarity calculation to helper_functions.similarity_methods()
* 5- if threading is needed, use helper_functions.thread_this() function with format: helper_functions.thread_this(function_to_pass_to_threads, list_of_parameters_to_pass_that_function), it will return the returned values in a list with same order of list_of_parameters_to_pass_that_function
* 6- do the needed imports

## * adding new deep learning model/dataset or clustering method to Deep Learning Clusterimg

#### deep learning model/dataset:
* 1- you can add your own deep learning model and dataset with attributes and functions explained in Clusterimg README. There wont be and code additions needed for that, just pass your objects to DL_Clusteror.\_\_init\_\_()
* 2- do the needed imports

#### clustering method:
* 1- add needed parameters to DL_Clusteror.\_\_init\_\_()
* 2- add clustering method to valid_methods in DL_Clusteror.arguman_check()
* 3- add your models parameter grid search to DL_Clusteror.calculate_grid_search() function.
* 4- add your model initialization to get_models() function inside DL_Clusteror.calculate_grid_search()
* 5- write a fit_predict() function to your model for fitting the passed data and returning the predicted labels for them. Pay attention to matching your clustering models fit_predict() output with current models output format: numpy.ndarray of shape (n_samples,)
* 6- do the needed imports

## * adding new method to Segmentimg

#### automatic method:
* 1- add needed parameters to Segmentator.\_\_init\_\_()
* 2- add method name to valid_methods in Segmentator.arguman_check()
* 3- add your segmentation function above the helper_functions.segment_image() function. Pay attention to matching your segmentation functions output with current methods output format: two dimensional numpy.ndarray with shape equal to original images shape[:2]. edges are indicated with value 0 and segments are labeled starting from 1
* 4- add your function to helper_functions.segment_image() function with needed parameters
* 5- do the needed imports

#### interactive method:
* 1- these methods require user input beforehand to segment an image
* 2- add method name to valid_methods in Segmentator.arguman_check()
* 3- we recommend you to create a new class .py file to seperate annotation part with segment painting part. Then pass your annotation object to Segmentator.\_\_init\_\_() in a new suitable parameter field
* 4- add a new Segmentator.process() function since methods requiring user input doesn't run on threads
* 5- add your segmentation function above the helper_functions.segment_image() function. Pay attention to matching your segmentation functions output with current methods output format: two dimensional numpy.ndarray with shape equal to original images shape[:2]. edges are indicated with value 0 and segments are labeled starting from 1
* 6- add your function to helper_functions.segment_image() function with needed parameters
* 7- do the needed imports

<br/><br/>
<br/><br/>