import cv2
import time
import os
import numpy as np
import threading
from threading import Lock
lock = Lock()

from helper_functions import *


class Segmentating:
    def __init__(self, image_folder, method, color_picker_image_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ColorPicker.png"), verbose=0):
        """initializing segmenting object

        Args:
            image_folder (str): path to images
            method (str): segmentation method
            color_picker_image_path (str): path to color picking image
            verbose (int, optional): verbose level. Defaults to 0.
        """
        self.image_folder = image_folder
        self.method = method
        self.verbose = verbose

        print(color_picker_image_path)
        self.color_picker_image = cv2.imread(color_picker_image_path)
        if self.color_picker_image is None:
            print_verbose("e", "No color picking image passed", verbose=verbose-1)

        base_folder, images_folder_name = os.path.split(self.image_folder)
        self.save_folder = os.path.join(base_folder, images_folder_name + "_clustered")

        self.segmented_image_dict = {} # distionary to save thread processing results
        self.thread_range = 10 # number of images to prepare at both left and right side of current index
        self.thread_stop = False # indicates when to stop threads

    def processed_image_callbacks(self, event, x, y, flags, callback_info):
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

        # if left button is clicked: action is filling individual segment
        if event == cv2.EVENT_LBUTTONDOWN:
            callback_info['clicked'] = True
            callback_info["action"] = "fill"
            callback_info["first_cut"] = None
            callback_info["second_cut"] = None

        # if right button is clicked: action is unfilling individual segment
        elif event == cv2.EVENT_RBUTTONDOWN:
            callback_info['clicked'] = True
            callback_info["action"] = "unfill"
            callback_info["first_cut"] = None
            callback_info["second_cut"] = None
        
        # if left button is double clicked: action is filling segments rapidly where mouse is hovered
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            callback_info["continuous_filling"] = not callback_info["continuous_filling"]
            if callback_info["continuous_filling"]:
                callback_info["continuous_unfilling"] = False
                callback_info["first_cut"] = None
                callback_info["second_cut"] = None
        
        # if right button is double clicked: action is unfilling segments rapidly where mouse is hovered
        elif event == cv2.EVENT_RBUTTONDBLCLK:
            callback_info["continuous_unfilling"] = not callback_info["continuous_unfilling"]
            if callback_info["continuous_unfilling"]:
                callback_info["continuous_filling"] = False
                callback_info["first_cut"] = None
                callback_info["second_cut"] = None

        # if middle button is clicked: action is cut segments with one line
        elif event == cv2.EVENT_MBUTTONDOWN:
            callback_info['clicked'] = True
            callback_info["action"] = "cut"
            if callback_info["first_cut"] == None: # initialize first point
                callback_info["first_cut"] = (x, y)
            elif callback_info["second_cut"] == None: # initialize second point
                callback_info["second_cut"] = (x, y)
            else: # shift points
                callback_info["first_cut"] = callback_info["second_cut"]
                callback_info["second_cut"] = (x, y)

    def color_callback(self, event, x, y, flags, color_info):
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

    def color_picker_feedback(self, callback_info, color_info):
        """imshows color picker image and extra informations for user

        Args:
            callback_info (dictionary): dictionary that holds information about user actions
            color_info (dictionary): dictionary that holds information about color selection
            verbose (int, optional): verbose level. Defaults to 0.
        """
        click_column = color_info['x']
        click_row = color_info['y']
        color_picker_image_display = self.color_picker_image.copy()
        
        # if one of continuous modes is on print the mode name on left corner
        if callback_info["continuous_filling"] or callback_info["continuous_unfilling"]:
            if callback_info["continuous_filling"]:
                mode_feedback = "filling"
            elif callback_info["continuous_unfilling"]:
                mode_feedback = "unfilling"
            
            color_picker_image_display = cv2.putText(color_picker_image_display, mode_feedback, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
            
        # put circle indicator to selected color
        color_picker_image_display = cv2.circle(color_picker_image_display, (click_column, click_row), 3, [0, 0, 0], 3)
        color_picker_image_display = cv2.circle(color_picker_image_display, (click_column, click_row), 1, [255, 255, 255], 2)

        cv2.imshow("Color Picker", color_picker_image_display)

    def process_image(self, files, file_no, region_size, ruler, k, color_importance, verbose=0):
        """Function to process segment images with thread

        Args:
            files (list): list of files
            file_no (int): index of current image
            region_size (int): region_size parameter for superpixel
            ruler (int): ruler parameter for superpixel
            k (int): k parameter for opencv kmeans
            color_importance (int): importance of pixel colors proportional to pixels coordinates
            verbose (int, optional): verbose level. Defaults to 0.
        """
        if self.thread_stop:
            return

        # iterate at surrounding images of current image
        for file_no in range(max(file_no-self.thread_range, 0), min(file_no+self.thread_range, len(files))):
            image_path = os.path.join(self.image_folder, files[file_no])
            
            # add key to dict to prevent upcoming threads to process same image while it is already being processed
            lock.acquire()
            if image_path not in self.segmented_image_dict.keys(): 
                self.segmented_image_dict[image_path] = None
            lock.release()

            # if image is not processed yet, process it and add to dictionary
            lock.acquire()
            if self.segmented_image_dict[image_path] is None: 
                raw_image = cv2.imread(image_path)
                segmented_image = segment_image(method=self.method, image_path=image_path, region_size=region_size, ruler=ruler, k=k, color_importance=color_importance)
                self.segmented_image_dict[image_path] = (raw_image, segmented_image)
            lock.release()

    def start_thread_func(self, files, file_no, region_size, ruler, k, color_importance, verbose=0):
        """function to start thread processing and return thread

        Args:
            files (list): list of files
            file_no (int): index of current image
            region_size (int): region_size parameter for superpixel
            ruler (int): ruler parameter for superpixel
            k (int): k parameter for opencv kmeans
            color_importance (int): importance of pixel colors proportional to pixels coordinates
            verbose (int, optional): verbose level. Defaults to 0.

        Returns:
            threading.Thread: thread that is created for processing
        """
        thread = threading.Thread(target=self.process_image, args=(files, file_no, region_size, ruler, k, color_importance, verbose-1), daemon=True)
        thread.start()
        return thread

    def save_masks(self, mask_path, painted_pixels, result_image, verbose=0):
        """saves each segment mask individualy

        Args:
            mask_path (str): incomplate path of every mask
            painted_pixels (numpy.ndarray): binary image of segmented and not segmented pixels
            result_image (numpy.ndarray): segmented image
            verbose (int, optional): verbose level. Defaults to 0.
        """
        segment_colors = np.unique(result_image[np.where(painted_pixels == 1)], axis=0)
        # for each unique color save a mask coded with its BGR value
        for color in segment_colors:
            indices = np.argwhere(np.all(result_image == color, axis=-1))
            mask = np.zeros_like(result_image)
            mask[indices[:, 0], indices[:, 1]] = [255,255,255]
            cv2.imwrite(mask_path + str(color) + ".png", mask)

    def segment(self, raw_image, segmented_image, result_image, image_name, image_no, verbose=0):
        """segments given image and saves it to output folder

        Args:
            raw_image (numpy.ndarray): non-processed image
            segmented_image (numpy.ndarray): segmented image
            result_image (numpy.ndarray): image that is being processed
            image_name (str): file name of image
            image_no (int), index of current file
            verbose (int, optional): verbose level. Defaults to 0.

        Returns:
            str: information for upper function about actions that it needs to take
        """

        # set windows and mouse event listeners for windows
        cv2.namedWindow("Processed Image " + str(image_no), flags = cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        cv2.imshow("Processed Image " + str(image_no), result_image)
        
        cv2.namedWindow("Color Picker", flags = cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        cv2.imshow("Color Picker", self.color_picker_image)

        callback_info = {'clicked': False, 'x': -1, 'y': -1, 'action':"", "first_cut":None, "second_cut":None, 'continuous_filling': False, 'continuous_unfilling': False}
        cv2.setMouseCallback("Processed Image " + str(image_no), self.processed_image_callbacks, callback_info)

        color_info = {'clicked': False, 'x': -1, 'y': -1}
        cv2.setMouseCallback("Color Picker", self.color_callback, color_info)

        painted_pixels=np.zeros_like(segmented_image) # pixels that are segmented
        line_image = np.zeros_like(result_image) # image to cut segment
        ctrl_z_stack = [] # stack for reversing actions
        color = [0,0,0] # BGR values

        while True:
            key = cv2.waitKey(1)
            self.color_picker_feedback(callback_info, color_info)

            ### --- --- --- --- --- process_key(key) function will capsulate here --- --- --- --- --- ###
            if key == ord('q'): # quit
                print_verbose("q", "ending_session, waiting for threads...", verbose=verbose-1)
                return "quit"
            elif key == ord('n'): # next image without saving current one
                print_verbose("n", "going forward from image" + image_name + " without saving", verbose=verbose-1)
                return "next"
            elif key == ord('p'): # previous image without saving current one
                print_verbose("p", "going back from image " + image_name + " without saving", verbose=verbose-1)
                return "previous"
            elif key == ord('s'): # save
                print_verbose("s", "going forward from image " + image_name + " after saving", verbose=verbose-1)
                self.save_masks(os.path.join(self.save_folder, image_name + "_mask_"), painted_pixels, result_image, verbose=verbose-1)
                return "save"
            elif key == ord('z'): # ctrl + z last action
                if len(ctrl_z_stack) > 0:
                    result_image, painted_pixels, line_image = ctrl_z_stack.pop()
                    cv2.imshow("Processed Image " + str(image_no), result_image)
            elif key == ord('r'): # reset all actions
                print_verbose("r", "reseting image " + image_name, verbose=verbose-1)
                ctrl_z_stack.append((previous_result_image.copy(), previous_painted_pixels.copy(), line_image.copy()))
                result_image = raw_image.copy()
                painted_pixels=np.zeros_like(segmented_image)
                line_image = np.zeros_like(result_image)
                cv2.imshow("Processed Image " + str(image_no), result_image)
            ### --- --- --- --- --- process_key(key) function will capsulate here --- --- --- --- --- ###

            ### --- --- --- --- --- process_color(color_info) function will capsulate here --- --- --- --- --- ###
            if color_info['clicked']: # color selecting
                click_column = color_info['x']
                click_row = color_info['y']
                color = self.color_picker_image[click_row, click_column]
                color_info["clicked"] = False
            ### --- --- --- --- --- process_color(color_info) function will capsulate here --- --- --- --- --- ###


            ### --- --- --- --- --- process_action(callback_info) function will capsulate here --- --- --- --- --- ###
            if callback_info["continuous_filling"] or callback_info["continuous_unfilling"]: # if one of continuous modes is on
                click_column = callback_info['x']
                click_row = callback_info['y']
                
                previous_result_image, previous_painted_pixels = result_image.copy(), painted_pixels.copy()

                if callback_info["continuous_filling"]:                
                    fill(result_image, segmented_image, painted_pixels, click_row, click_column, color)
                
                elif callback_info["continuous_unfilling"]:
                    unfill(result_image, painted_pixels, raw_image, click_row, click_column)

                if np.any(np.equal(previous_result_image, result_image) == False): # means there is a change while continuously filling
                    ctrl_z_stack.append((previous_result_image.copy(), previous_painted_pixels.copy(), line_image.copy()))
                    cv2.imshow("Processed Image " + str(image_no), result_image)

                continue

            if callback_info['clicked']: # if a clicking action detected
                callback_info["clicked"] = False
                click_column = callback_info['x']
                click_row = callback_info['y']

                ctrl_z_stack.append((result_image.copy(), painted_pixels.copy(), line_image.copy()))

                if callback_info["action"] == "fill": # fill the thing at pos: [click_row, click_column]
                    fill(result_image, segmented_image, painted_pixels, click_row, click_column, color)
                    
                elif callback_info["action"] == "unfill": # unfill the thing at pos: [click_row, click_column]
                    unfill(result_image, painted_pixels, raw_image, click_row, click_column)
                    
                elif callback_info["action"] == "cut": # cut the segments with a line
                    if callback_info["first_cut"] != None and callback_info["second_cut"] != None:
                        cv2.line(line_image, callback_info["first_cut"] , callback_info["second_cut"], (255,255,255), 1) 
                        result_image[line_image==255] = raw_image[line_image==255]
                        segmented_image[line_image[:,:,0]==255] = 0
                        painted_pixels[line_image[:,:,0]==255] = 0

                cv2.imshow("Processed Image " + str(image_no), result_image)
            ### --- --- --- --- --- process_action(callback_info) function will capsulate here --- --- --- --- --- ###

    def start_segmenting(self, region_size=40, ruler=30, k=15, color_importance=5, verbose=0):
        """function to segment images in a folder in order and save them to output folder

        Args:
            region_size (int, optional): regions_size parameter for cv2 superpixel. Defaults to 40.
            ruler (int, optional): ruler parameter for cv2 superpixel. Defaults to 30.
            k (int, optional): k parameter for cv2 kmeans. Defaults to 15.
            color_importance (int, optional): color importance parameter for cv2 kmeans. Defaults to 5.
            verbose (int, optional): verbose level. Defaults to 0.
        """
        os.makedirs(self.save_folder, exist_ok=True)
        files = sorted(os.listdir(self.image_folder))

        file_no = 0
        threads = {}
        while 0 <= file_no < len(files):
            
            # if file is already processed and far passed join its thread
            for file_no_keys in threads.keys():
                if abs(file_no_keys - file_no_keys) > self.thread_range:
                    threads[file_no_keys].join()
            
            file_name = files[file_no]
            image_path = os.path.join(self.image_folder, file_name)

            # if file is not processed yet, start a process for it
            if file_no not in threads.keys():
                thread = self.start_thread_func(files, file_no, region_size, ruler, k, color_importance, verbose=verbose-1)
                threads[file_no] = thread

            # if files process is done start segmenting it
            if image_path in self.segmented_image_dict.keys() and self.segmented_image_dict[image_path] is not None:
                raw_image, segmented_image = self.segmented_image_dict[image_path]
                return_code = self.segment(raw_image, segmented_image, raw_image.copy(), file_name, file_no, verbose=verbose-1)

                cv2.destroyWindow("Processed Image " + str(file_no))
                
                if return_code == "quit": # q
                    self.thread_stop = True
                    cv2.destroyAllWindows()
                    time.sleep(1)
                    return # end the session
                elif return_code == "next" or return_code == "save": # n or s
                    file_no = file_no + 1
                elif return_code == "previous": # p
                    file_no = file_no - 1
        
        self.thread_stop = True
        cv2.destroyAllWindows()
        time.sleep(1)

    def process(self, verbose=0):
        """main process to capsulate every process

        Args:
            verbose (int, optional): level of verbose. Defaults to 0.
        """
        self.start_segmenting(verbose=verbose-1)
        cv2.destroyAllWindows()

    def __call__(self):
        """calling the object will start the main process and catch any possible exception during
        """
        try:
            self.process(self.verbose-1)
        except ErrorException as ee:
            print(ee.message)
            exit(ee.error_code)
        except WrongTypeException as wte:
            print(wte.message)
            exit(wte.error_code)