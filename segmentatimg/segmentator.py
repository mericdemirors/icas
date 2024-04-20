import cv2
import time
import os
import numpy as np
import threading
from threading import Lock
lock = Lock()

from helper_functions import *

def processed_image_callbacks(event, x, y, flags, callback_info):
    """detects mouse inputs and manages information dictionary

    Args:
        event (opencv event): mouse event to detect
        x (int): column coordinate of mouse
        y (int): row coordinate of mouse
        flags (opencv flags): flags
        callback_info (dictionary): information dictionary that manages segmentation
    """
    callback_info['x'] = x
    callback_info['y'] = y

    # if left button is clicked: action is filling individual cluster
    if event == cv2.EVENT_LBUTTONDOWN:
        callback_info['clicked'] = True
        callback_info["action"] = "fill"
        callback_info["first_cut"] = None
        callback_info["second_cut"] = None

    # if right button is clicked: action is unfilling individual cluster
    elif event == cv2.EVENT_RBUTTONDOWN:
        callback_info['clicked'] = True
        callback_info["action"] = "unfill"
        callback_info["first_cut"] = None
        callback_info["second_cut"] = None
    
    # if left button is double clicked: action is filling clusters rapidly where mouse is hovered
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        callback_info["continuous_filling"] = not callback_info["continuous_filling"]
        if callback_info["continuous_filling"]:
            callback_info["continuous_unfilling"] = False
            callback_info["first_cut"] = None
            callback_info["second_cut"] = None
    
    # if right button is double clicked: action is unfilling clusters rapidly where mouse is hovered
    elif event == cv2.EVENT_RBUTTONDBLCLK:
        callback_info["continuous_unfilling"] = not callback_info["continuous_unfilling"]
        if callback_info["continuous_unfilling"]:
            callback_info["continuous_filling"] = False
            callback_info["first_cut"] = None
            callback_info["second_cut"] = None

    # if middle button is clicked: action is cut clusters with one line
    elif event ==  cv2.EVENT_MBUTTONDOWN:
        callback_info['clicked'] = True
        callback_info["action"] = "cut"
        if callback_info["first_cut"] == None: # initialize first point
            callback_info["first_cut"] = (x, y)
        elif callback_info["second_cut"] == None: # initialize second point
            callback_info["second_cut"] = (x, y)
        else: # shift points
            callback_info["first_cut"] = callback_info["second_cut"]
            callback_info["second_cut"] = (x, y)

def color_callback(event, x, y, flags, color_info):
    """detects mouse inputs and manages information dictionary

    Args:
        event (opencv event): mouse event to detect
        x (int): column coordinate of mouse
        y (int): row coordinate of mouse
        flags (opencv flags): flags
        color_info (dictionary): information dictionary that manages segmentation
    """
    # if left button is clicked: action is selecting color that is clicked on
    if event == cv2.EVENT_LBUTTONDOWN:
        color_info['x'] = x
        color_info['y'] = y
        color_info['clicked'] = True

def user_feedback(callback_info, color_picker_img, color_info):
    """imshows color picker image and extra informations for user

    Args:
        callback_info (dictionary): dictionary that holds information about user actions
        color_picker_img (numpy.ndarray): image to pick color from
        color_info (dictionary): dictionary that holds information about color selection
    """
    click_column = color_info['x']
    click_row = color_info['y']
    color_picker_img_display = color_picker_img.copy()
    
    # if one of continuous modes is on print the mode name on left corner
    if callback_info["continuous_filling"] or callback_info["continuous_unfilling"]:
        if callback_info["continuous_filling"]:
            mode_feedback = "filling"
        elif callback_info["continuous_unfilling"]:
            mode_feedback = "unfilling"
        
        color_picker_img_display = cv2.putText(color_picker_img_display, mode_feedback, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        
    # put circle indicator to selected color
    color_picker_img_display = cv2.circle(color_picker_img_display, (click_column, click_row), 3, [0, 0, 0], 3)
    color_picker_img_display = cv2.circle(color_picker_img_display, (click_column, click_row), 1, [255, 255, 255], 2)

    cv2.imshow("Color Picker", color_picker_img_display)

segmented_img_dict = {} # distionary to save thread processing results
thread_range = 10 # number of images to prepare at both left and right side of current index
thread_stop = False # indicates when to stop threads
def process_image(files, file_no, method, region_size, ruler, k, color_importance):
    """Function to process cluster images with thread

    Args:
        files (list): list of files
        file_no (int): index of current image
        method (str): method to use at image clustering
        region_size (int): region_size parameter for superpixel
        ruler (int): ruler parameter for superpixel
        k (int): k parameter for opencv kmeans
        color_importance (int): importance of pixel colors proportional to pixels coordinates
    """
    global thread_range, thread_stop, segmented_img_dict
    if thread_stop:
        return

    # iterate at surrounding images of current image
    for file_no in range(max(file_no-thread_range, 0), min(file_no+thread_range, len(files))):
        img_path = os.path.join(img_folder, files[file_no])
        
        # add key to dict to prevent upcoming threads to process same image while it is already being processed
        lock.acquire()
        if img_path not in segmented_img_dict.keys(): 
            segmented_img_dict[img_path] = None
        lock.release()

        lock.acquire()
        if segmented_img_dict[img_path] is None: 
            # if iterated image is not processed add it to dictionary
            raw_img = cv2.imread(img_path)
            clustered_img = segment_image(method=method, img_path=img_path, region_size=region_size, ruler=ruler, k=k, color_importance=color_importance)
            segmented_img_dict[img_path] = (raw_img, clustered_img)
        lock.release()

def start_thread_func(files, file_no, method, region_size, ruler, k, color_importance):
    """function to start thread processing and return thread

    Args:
        files (list): list of files
        file_no (int): index of current image
        method (str): method to use at image clustering
        region_size (int): region_size parameter for superpixel
        ruler (int): ruler parameter for superpixel
        k (int): k parameter for opencv kmeans
        color_importance (int): importance of pixel colors proportional to pixels coordinates

    Returns:
        threading.Thread: thread that is created for processing
    """
    thread = threading.Thread(target=process_image, args=(files, file_no, method, region_size, ruler, k, color_importance), daemon=True)
    thread.start()
    print("Thread", thread, "started", file_no)
    return thread

def segment(raw_img, clustered_img, result_img, save_folder, image_name, image_no, color_picker_img):
    """segments given image and saves it to output folder

    Args:
        raw_img (numpy.ndarray): non-processed image
        clustered_img (numpy.ndarray): segmented image
        result_img (numpy.ndarray): image that is being processed
        save_folder (str): folder path to save segmented image
        image_name (str): file name of image
        image_no (int), index of current file
        color_picker_img (numpy.ndarray): image to pick colors from

    Returns:
        str: information for upper function about actions that it needs to take
    """

    # set windows and mouse event listeners for windows
    cv2.namedWindow("Processed Image " + str(image_no), flags = cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
    cv2.imshow("Processed Image " + str(image_no), result_img)
    
    cv2.namedWindow("Color Picker", flags = cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
    cv2.imshow("Color Picker", color_picker_img)

    callback_info = {'clicked': False, 'x': -1, 'y': -1, 'action':"", "first_cut":None, "second_cut":None, 'continuous_filling': False, 'continuous_unfilling': False}
    cv2.setMouseCallback("Processed Image " + str(image_no), processed_image_callbacks, callback_info)

    color_info = {'clicked': False, 'x': -1, 'y': -1}
    cv2.setMouseCallback("Color Picker", color_callback, color_info)

    painted_pixels=np.zeros_like(clustered_img) # pixels that are segmented
    line_img = np.zeros_like(result_img) # image to cut cluster
    ctrl_z_stack = [] # stack for reversing actions
    color = [0,0,0] # BGR values

    while True:
        key = cv2.waitKey(1)

        ### --- --- --- --- --- process_key(key) function will capsulate here --- --- --- --- --- ###
        if key == ord('q'): # quit
            print_verbose("q", "ending_session, waiting for threads...")
            return "quit"
        elif key == ord('n'): # next image without saving current one
            print_verbose("n", "going forward from image" + image_name + " without saving")
            return "next"
        elif key == ord('p'): # previous image without saving current one
            print_verbose("p", "going back from image " + image_name + " without saving")
            return "previous"
        elif key == ord('s'): # save
            print_verbose("s", "going forward from image " + image_name + " after saving")
            cv2.imwrite(os.path.join(save_folder, image_name + "_mask_" + str(time.strftime("%H:%M:%S") + ".png")), result_img)
            return "save"
        elif key == ord('z'): # ctrl + z last action
            if len(ctrl_z_stack) > 0:
                result_img, painted_pixels, line_img = ctrl_z_stack.pop()
                cv2.imshow("Processed Image " + str(image_no), result_img)
        elif key == ord('r'): # reset all actions
            print_verbose("r", "reseting image " + image_name)
            ctrl_z_stack.append((previous_result_img.copy(), previous_painted_pixels.copy(), line_img.copy()))
            result_img = raw_img.copy()
            painted_pixels=np.zeros_like(clustered_img)
            line_img = np.zeros_like(result_img)
            cv2.imshow("Processed Image " + str(image_no), result_img)
        ### --- --- --- --- --- process_key(key) function will capsulate here --- --- --- --- --- ###


        ### --- --- --- --- --- process_color(color_info) function will capsulate here --- --- --- --- --- ###
        if color_info['clicked']: # color selecting
            click_column = color_info['x']
            click_row = color_info['y']
            color = color_picker_img[click_row, click_column]
            color_info["clicked"] = False
        ### --- --- --- --- --- process_color(color_info) function will capsulate here --- --- --- --- --- ###

        # display feedback on color_picker_img(can be more efficient than always calling this function[maybe call on action updates])
        user_feedback(callback_info, color_picker_img, color_info)

        ### --- --- --- --- --- process_action(callback_info) function will capsulate here --- --- --- --- --- ###
        if callback_info["continuous_filling"] or callback_info["continuous_unfilling"]: # if one of continuous modes is on
            click_column = callback_info['x']
            click_row = callback_info['y']
            
            previous_result_img, previous_painted_pixels = result_img.copy(), painted_pixels.copy()

            if callback_info["continuous_filling"]:                
                fill(result_img, clustered_img, painted_pixels, click_row, click_column, color)
            
            elif callback_info["continuous_unfilling"]:
                unfill(result_img, painted_pixels, raw_img, click_row, click_column)

            if np.any(np.equal(previous_result_img, result_img) == False): # means there is a change while continuously filling
                ctrl_z_stack.append((previous_result_img.copy(), previous_painted_pixels.copy(), line_img.copy()))
                cv2.imshow("Processed Image " + str(image_no), result_img)

            continue

        if callback_info['clicked']: # if a clicking action detected
            callback_info["clicked"] = False
            click_column = callback_info['x']
            click_row = callback_info['y']

            ctrl_z_stack.append((result_img.copy(), painted_pixels.copy(), line_img.copy()))

            if callback_info["action"] == "fill": # fill the thing at pos: [click_row, click_column]
                fill(result_img, clustered_img, painted_pixels, click_row, click_column, color)
                
            elif callback_info["action"] == "unfill": # unfill the thing at pos: [click_row, click_column]
                unfill(result_img, painted_pixels, raw_img, click_row, click_column)
                
            elif callback_info["action"] == "cut": # cut the clusters with a line
                if callback_info["first_cut"] != None and callback_info["second_cut"] != None:
                    cv2.line(line_img, callback_info["first_cut"] , callback_info["second_cut"], (255,255,255), 1) 
                    result_img[line_img==255] = raw_img[line_img==255]
                    clustered_img[line_img[:,:,0]==255] = clustered_img.max()+1
                    painted_pixels[line_img[:,:,0]==255] = 0

            cv2.imshow("Processed Image " + str(image_no), result_img)
        ### --- --- --- --- --- process_action(callback_info) function will capsulate here --- --- --- --- --- ###

def start_segmenting(img_folder, save_folder, method, region_size=40, ruler=30, k=15, color_importance=5, color_picker_path=""):
    """function to segment images in a folder in order and save them to output folder

    Args:
        img_folder (str): path of images folder
        save_folder (str): path of output folder
        method (str): method to use preprocess image segmentation
        region_size (int, optional): regions_size parameter for cv2 superpixel. Defaults to 40.
        ruler (int, optional): ruler parameter for cv2 superpixel. Defaults to 30.
        k (int, optional): k parameter for cv2 kmeans. Defaults to 15.
        color_importance (int, optional): color importance parameter for cv2 kmeans. Defaults to 5.
        color_picker_path (str, optional): path to read color picking image.
    """
    global thread_stop

    color_picker_img = cv2.imread(color_picker_path)
    if color_picker_img is None:
        print_verbose("e", "No color picking image passed")

    save_folder = os.path.join(os.path.split(img_folder)[0], save_folder)
    os.makedirs(save_folder, exist_ok=True)
    files = sorted(os.listdir(img_folder))

    file_no = 0
    threads = {}
    while 0 <= file_no < len(files): # iterate over files(segmented images wont overwrite ecah other, they all have a time stamp in their file name)
        for file_no_keys in threads.keys():
            if abs(file_no_keys - file_no_keys) > thread_range:
                threads[file_no_keys].join()
        
        file_name = files[file_no]
        img_path = os.path.join(img_folder, file_name)

        if file_no not in threads.keys():
            thread = start_thread_func(files, file_no, method, region_size, ruler, k, color_importance)
            threads[file_no] = thread
        else:
            pass

        if img_path in segmented_img_dict.keys() and segmented_img_dict[img_path] is not None:
            raw_img, clustered_img = segmented_img_dict[img_path]
            return_code = segment(raw_img, clustered_img, raw_img.copy(), save_folder, file_name, file_no, color_picker_img)

            cv2.destroyWindow("Processed Image " + str(file_no))
            
            if return_code == "quit": # q
                thread_stop = True
                time.sleep(1)
                return # end the session
            elif return_code == "next" or return_code == "save": # n or s
                file_no = file_no + 1
            elif return_code == "previous": # p
                file_no = file_no - 1
    
    thread_stop = True
    cv2.destroyAllWindows()
    time.sleep(1)


try:
    img_folder="/home/mericdemirors/Pictures/titles"
    save_folder="/home/mericdemirors/Pictures/titles_seg"
    method="s" # input("method to segmentate images(s:superpixel, km:kmeans, e:edge): ")

    method_dict = {"s":"superpixel", "km":"kmeans", "e":"edge"}
    method = method_dict[method]

    cp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ColorPicker.png")
    print(cp_path)
    start_segmenting(img_folder, save_folder, method, color_picker_path=cp_path)

    cv2.destroyAllWindows()
except ErrorException as ee:
    print(ee.message)
    exit(ee.error_code)
except WrongTypeException as wte:
    print(wte.message)
    exit(wte.error_code)