import cv2
import time
import os
import numpy as np
import threading
from threading import Lock
lock = Lock()

from helper_functions import *


class Segmentating:
    def __init__(self, image_folder, save_folder, method, verbose=0):
        self.folder = image_folder
        self.save_folder = save_folder
        self.method = method
        self.verbose=0

        self.segmented_img_dict = {} # distionary to save thread processing results
        self.thread_range = 10 # number of images to prepare at both left and right side of current index
        self.thread_stop = False # indicates when to stop threads

    def processed_image_callbacks(self, event, x, y, flags, callback_info, verbose=0):
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

    def color_picker_feedback(self, callback_info, color_picker_img, color_info, verbose=0):
        """imshows color picker image and extra informations for user

        Args:
            callback_info (dictionary): dictionary that holds information about user actions
            color_picker_img (numpy.ndarray): image to pick color from
            color_info (dictionary): dictionary that holds information about color selection
            verbose (int, optional): verbose level. Defaults to 0.
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

    def process_image(self, files, file_no, method, region_size, ruler, k, color_importance, verbose=0):
        """Function to process segment images with thread

        Args:
            files (list): list of files
            file_no (int): index of current image
            method (str): method to use at image segmenting
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
            img_path = os.path.join(self.folder, files[file_no])
            
            # add key to dict to prevent upcoming threads to process same image while it is already being processed
            lock.acquire()
            if img_path not in self.segmented_img_dict.keys(): 
                self.segmented_img_dict[img_path] = None
            lock.release()

            # if image is not processed yet, process it and add to dictionary
            lock.acquire()
            if self.segmented_img_dict[img_path] is None: 
                raw_img = cv2.imread(img_path)
                segmented_img = segment_image(method=method, img_path=img_path, region_size=region_size, ruler=ruler, k=k, color_importance=color_importance)
                self.segmented_img_dict[img_path] = (raw_img, segmented_img)
            lock.release()

    def start_thread_func(self, files, file_no, method, region_size, ruler, k, color_importance, verbose=0):
        """function to start thread processing and return thread

        Args:
            files (list): list of files
            file_no (int): index of current image
            method (str): method to use at image segmenting
            region_size (int): region_size parameter for superpixel
            ruler (int): ruler parameter for superpixel
            k (int): k parameter for opencv kmeans
            color_importance (int): importance of pixel colors proportional to pixels coordinates
            verbose (int, optional): verbose level. Defaults to 0.

        Returns:
            threading.Thread: thread that is created for processing
        """
        thread = threading.Thread(target=self.process_image, args=(files, file_no, method, region_size, ruler, k, color_importance, verbose-1), daemon=True)
        thread.start()
        return thread

    def save_masks(self, mask_path, painted_pixels, result_img, verbose=0):
        """saves each segment mask individualy

        Args:
            mask_path (str): incomplate path of every mask
            painted_pixels (numpy.ndarray): binary image of segmented and not segmented pixels
            result_img (numpy.ndarray): segmented image
            verbose (int, optional): verbose level. Defaults to 0.
        """
        segment_colors = np.unique(result_img[np.where(painted_pixels == 1)], axis=0)
        # for each unique color save a mask coded with its BGR value
        for color in segment_colors:
            indices = np.argwhere(np.all(result_img == color, axis=-1))
            mask = np.zeros_like(result_img)
            mask[indices[:, 0], indices[:, 1]] = [255,255,255]
            cv2.imwrite(mask_path + str(color) + ".png", mask)

    def segment(self, raw_img, segmented_img, result_img, save_folder, image_name, image_no, color_picker_img, verbose=0):
        """segments given image and saves it to output folder

        Args:
            raw_img (numpy.ndarray): non-processed image
            segmented_img (numpy.ndarray): segmented image
            result_img (numpy.ndarray): image that is being processed
            save_folder (str): folder path to save segmented image
            image_name (str): file name of image
            image_no (int), index of current file
            color_picker_img (numpy.ndarray): image to pick colors from
            verbose (int, optional): verbose level. Defaults to 0.

        Returns:
            str: information for upper function about actions that it needs to take
        """

        # set windows and mouse event listeners for windows
        cv2.namedWindow("Processed Image " + str(image_no), flags = cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        cv2.imshow("Processed Image " + str(image_no), result_img)
        
        cv2.namedWindow("Color Picker", flags = cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        cv2.imshow("Color Picker", color_picker_img)

        callback_info = {'clicked': False, 'x': -1, 'y': -1, 'action':"", "first_cut":None, "second_cut":None, 'continuous_filling': False, 'continuous_unfilling': False}
        cv2.setMouseCallback("Processed Image " + str(image_no), self.processed_image_callbacks, callback_info)

        color_info = {'clicked': False, 'x': -1, 'y': -1}
        cv2.setMouseCallback("Color Picker", self.color_callback, color_info)

        painted_pixels=np.zeros_like(segmented_img) # pixels that are segmented
        line_img = np.zeros_like(result_img) # image to cut segment
        ctrl_z_stack = [] # stack for reversing actions
        color = [0,0,0] # BGR values

        while True:
            key = cv2.waitKey(1)

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
                self.save_masks(os.path.join(save_folder, image_name + "_mask_"), painted_pixels, result_img, verbose=verbose-1)
                return "save"
            elif key == ord('z'): # ctrl + z last action
                if len(ctrl_z_stack) > 0:
                    result_img, painted_pixels, line_img = ctrl_z_stack.pop()
                    cv2.imshow("Processed Image " + str(image_no), result_img)
            elif key == ord('r'): # reset all actions
                print_verbose("r", "reseting image " + image_name, verbose=verbose-1)
                ctrl_z_stack.append((previous_result_img.copy(), previous_painted_pixels.copy(), line_img.copy()))
                result_img = raw_img.copy()
                painted_pixels=np.zeros_like(segmented_img)
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
            self.color_picker_feedback(callback_info, color_picker_img, color_info, verbose=verbose-1)

            ### --- --- --- --- --- process_action(callback_info) function will capsulate here --- --- --- --- --- ###
            if callback_info["continuous_filling"] or callback_info["continuous_unfilling"]: # if one of continuous modes is on
                click_column = callback_info['x']
                click_row = callback_info['y']
                
                previous_result_img, previous_painted_pixels = result_img.copy(), painted_pixels.copy()

                if callback_info["continuous_filling"]:                
                    fill(result_img, segmented_img, painted_pixels, click_row, click_column, color)
                
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
                    fill(result_img, segmented_img, painted_pixels, click_row, click_column, color)
                    
                elif callback_info["action"] == "unfill": # unfill the thing at pos: [click_row, click_column]
                    unfill(result_img, painted_pixels, raw_img, click_row, click_column)
                    
                elif callback_info["action"] == "cut": # cut the segments with a line
                    if callback_info["first_cut"] != None and callback_info["second_cut"] != None:
                        cv2.line(line_img, callback_info["first_cut"] , callback_info["second_cut"], (255,255,255), 1) 
                        result_img[line_img==255] = raw_img[line_img==255]
                        segmented_img[line_img[:,:,0]==255] = 0
                        painted_pixels[line_img[:,:,0]==255] = 0

                cv2.imshow("Processed Image " + str(image_no), result_img)
            ### --- --- --- --- --- process_action(callback_info) function will capsulate here --- --- --- --- --- ###

    def start_segmenting(self, image_folder, save_folder, method, region_size=40, ruler=30, k=15, color_importance=5, color_picker_path="", verbose=0):
        """function to segment images in a folder in order and save them to output folder

        Args:
            image_folder (str): path of images folder
            save_folder (str): path of output folder
            method (str): method to use preprocess image segmentation
            region_size (int, optional): regions_size parameter for cv2 superpixel. Defaults to 40.
            ruler (int, optional): ruler parameter for cv2 superpixel. Defaults to 30.
            k (int, optional): k parameter for cv2 kmeans. Defaults to 15.
            color_importance (int, optional): color importance parameter for cv2 kmeans. Defaults to 5.
            color_picker_path (str, optional): path to read color picking image.
            verbose (int, optional): verbose level. Defaults to 0.
        """
        color_picker_img = cv2.imread(color_picker_path)
        if color_picker_img is None:
            print_verbose("e", "No color picking image passed", verbose=verbose-1)

        save_folder = os.path.join(os.path.split(image_folder)[0], save_folder)
        os.makedirs(save_folder, exist_ok=True)
        files = sorted(os.listdir(image_folder))

        file_no = 0
        threads = {}
        while 0 <= file_no < len(files):
            
            # if file is already processed and far passed join its thread
            for file_no_keys in threads.keys():
                if abs(file_no_keys - file_no_keys) > self.thread_range:
                    threads[file_no_keys].join()
            
            file_name = files[file_no]
            img_path = os.path.join(image_folder, file_name)

            # if file is not processed yet, start a process for it
            if file_no not in threads.keys():
                thread = self.start_thread_func(files, file_no, method, region_size, ruler, k, color_importance, verbose=verbose-1)
                threads[file_no] = thread

            # if files process is done start segmenting it
            if img_path in self.segmented_img_dict.keys() and self.segmented_img_dict[img_path] is not None:
                raw_img, segmented_img = self.segmented_img_dict[img_path]
                return_code = self.segment(raw_img, segmented_img, raw_img.copy(), save_folder, file_name, file_no, color_picker_img, verbose=verbose-1)

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

        cp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ColorPicker.png")
        print(cp_path)
        self.start_segmenting(self.folder, self.save_folder, self.method, color_picker_path=cp_path, verbose=verbose-1)

        cv2.destroyAllWindows()

    def __call__(self):
        """calling the object will start the main process and catch any possible exception during
        """
        try:
            self.process(self.verbose)
        except ErrorException as ee:
            print(ee.message)
            exit(ee.error_code)
        except WrongTypeException as wte:
            print(wte.message)
            exit(wte.error_code)