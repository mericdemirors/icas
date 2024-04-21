import cv2
import time
import os
import numpy as np
import threading
from threading import Lock
lock = Lock()

from helper_functions import *

class Segmentating:
    def __init__(self, image_folder, method, color_picker_image_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ColorPicker.png"), region_size = 40, ruler = 30, k = 15, color_importance = 5, templates=[], attentions=[], segments=[], masks=[], thread_range = 10, verbose=0):
        """initializing segmenting object

        Args:
            image_folder (str): path to images
            method (str): segmentation method
            color_picker_image_path (str): path to color picking image            
            region_size (int, optional): regions_size parameter for cv2 superpixel. Defaults to 40.
            ruler (int, optional): ruler parameter for cv2 superpixel. Defaults to 30.
            k (int, optional): k parameter for cv2 kmeans. Defaults to 15.
            color_importance (int, optional): color importance parameter for cv2 kmeans. Defaults to 5.
            temp_att_seg_mask (dictionary): templates, template matching masks, segments and masks
            thread_range (int, optional): depth of image processings at previous and upcoming images on list. Defaults to 10.
            verbose (int, optional): verbose level. Defaults to 0.
        """
        self.image_folder = image_folder
        self.files = sorted([os.path.join(self.image_folder, file) for file in os.listdir(self.image_folder)])
        self.method = method
        self.verbose = verbose
        self.region_size = region_size
        self.ruler = ruler
        self.k = k
        self.color_importance = color_importance

        # if attentions == [] and masks == []:
        #     for (temp, seg) in list(zip(templates, segments)):
        #         border_temp = cv2.copyMakeBorder(temp, 1,1,1,1, cv2.BORDER_CONSTANT, value=[0,0,0])
        #         selected_part_B = flood(border_temp[:,:,0], (0, 0), connectivity=1).astype(np.uint8)
        #         selected_part_G = flood(border_temp[:,:,1], (0, 0), connectivity=1).astype(np.uint8)
        #         selected_part_R = flood(border_temp[:,:,2], (0, 0), connectivity=1).astype(np.uint8)
        #         selected_part = np.logical_and(selected_part_B, selected_part_G, selected_part_R)
        #         att = np.ones_like(border_temp)
        #         att[selected_part == 1]
        #         att[:,:,0][selected_part==1] = border_temp[:,:,0][selected_part==1]
        #         att[:,:,1][selected_part==1] = border_temp[:,:,1][selected_part==1]
        #         att[:,:,2][selected_part==1] = border_temp[:,:,2][selected_part==1]
        #         attentions.append(att[1:-1, 1:-1])

        #         border_seg = cv2.copyMakeBorder(seg, 1,1,1,1, cv2.BORDER_CONSTANT, value=[0,0,0])
        #         selected_part_B = flood(border_seg[:,:,0], (0, 0), connectivity=1).astype(np.uint8)
        #         selected_part_G = flood(border_seg[:,:,1], (0, 0), connectivity=1).astype(np.uint8)
        #         selected_part_R = flood(border_seg[:,:,2], (0, 0), connectivity=1).astype(np.uint8)
        #         selected_part = np.logical_and(selected_part_B, selected_part_G, selected_part_R)
        #         mask = np.ones_like(border_seg)
        #         mask[selected_part == 1]
        #         mask[:,:,0][selected_part==1] = border_seg[:,:,0][selected_part==1]
        #         mask[:,:,1][selected_part==1] = border_seg[:,:,1][selected_part==1]
        #         mask[:,:,2][selected_part==1] = border_seg[:,:,2][selected_part==1]
        #         masks.append(mask[1:-1, 1:-1])

        self.temp_att_seg_mask = list(zip(templates, attentions, segments, masks))

        self.refresh_images = False
        self.empty_images()

        self.color_picker_image = cv2.imread(color_picker_image_path)
        if self.color_picker_image is None:
            print_verbose("e", "No color picking image passed", verbose=verbose-1)

        base_folder, images_folder_name = os.path.split(self.image_folder)
        self.save_folder = os.path.join(base_folder, images_folder_name + "_segmented")

        self.segmented_image_dict = {} # distionary to save thread processing results
        self.thread_range = thread_range # number of images to prepare at both left and right side of current index
        self.thread_stop = False # indicates when to stop threads

    def empty_images(self):
        """empties object image attributes
        """
        self.raw_image = None
        self.result_image = None
        self.segmented_image = None
        self.painted_pixels = None
        self.orig_raw_image = None
        self.orig_result_image = None
        self.orig_segmented_image = None
        self.orig_painted_pixels = None
        cv2.destroyAllWindows()
    
    def reset_images(self):
        """resets object image attributes to originals
        """
        self.raw_image = self.orig_raw_image.copy()
        self.result_image = self.orig_result_image.copy()
        self.segmented_image = self.orig_segmented_image.copy()
        self.painted_pixels = self.orig_painted_pixels.copy()
    
    def set_images(self, raw_image, orig_segmented_image):
        """sets object image attributes to given images

        Args:
            raw_image (numpy.ndarray): nonprocessed image
            orig_segmented_image (numpy.ndarray): pre-segmented image
        """
        if self.raw_image is None:
            self.raw_image = raw_image.copy()
            self.orig_raw_image = raw_image.copy()
        if self.result_image is None:
            self.result_image = raw_image.copy()
            self.orig_result_image = raw_image.copy()
        if self.segmented_image is None:
            self.segmented_image = orig_segmented_image.copy()
            self.orig_segmented_image = orig_segmented_image.copy()
        if self.painted_pixels is None:
            self.painted_pixels = np.zeros(self.segmented_image.shape)
            self.orig_painted_pixels = np.zeros(self.segmented_image.shape)
        cv2.destroyAllWindows()

    def display_images(self, file_no):
        """refresh the image displays only if a change happend

        Args:
            file_no (int): image file number
        """
        if self.refresh_images:
            cv2.imshow("Processed Image " + str(file_no), self.result_image)
            if cv2.getWindowProperty("Segmented Image(Debug)", cv2.WND_PROP_VISIBLE) > 0:
                cv2.imshow("Segmented Image(Debug)", (self.segmented_image).astype(np.uint8))
                cv2.imshow("Painter Pixels(Debug)", (self.painted_pixels*255).astype(np.uint8))
            self.refresh_images = False

    def click_event_listener(self, event, x, y, flags, callback_info):
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

    def color_event_listener(self, event, x, y, flags, color_info):
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

    def display_color_picker(self, callback_info, color_info):
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

    def create_thread(self, file_no, region_size, ruler, k, color_importance, verbose=0):
        """function to start thread processing and return thread

        Args:
            file_no (int): index of current image
            region_size (int): region_size parameter for superpixel
            ruler (int): ruler parameter for superpixel
            k (int): k parameter for opencv kmeans
            color_importance (int): importance of pixel colors proportional to pixels coordinates
            verbose (int, optional): verbose level. Defaults to 0.

        Returns:
            threading.Thread: thread that is created for processing
        """

        def pass_image_to_thread(file_no, region_size, ruler, k, color_importance, verbose=0):
            """Function to process segment images with thread

            Args:
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
            for file_no in range(max(file_no-self.thread_range, 0), min(file_no+self.thread_range, len(self.files))):
                image_path = os.path.join(self.image_folder, self.files[file_no])
                
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

        thread = threading.Thread(target=pass_image_to_thread, args=(file_no, region_size, ruler, k, color_importance, verbose-1), daemon=True)
        thread.start()
        return thread

    def save_masks(self, mask_path, result_image, painted_pixels, verbose=0):
        """saves each segment mask individualy

        Args:
            mask_path (str): incomplate path of every mask
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

    def process_keyboard_input(self, file_no, ctrl_z_stack, key, verbose=0):
        """processes keyboard inputs

        Args:
            file_no (int): image file number
            ctrl_z_stack (list): list of last actions for reverse
            key (str): keyboard input
            verbose (int, optional): verbose level. Defaults to 0.

        Returns:
            str: what action is taken
        """
        image_name = os.path.split(self.files[file_no])[1]
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
            self.save_masks(os.path.join(self.save_folder, image_name + "_mask_"), self.result_image, self.painted_pixels, verbose=verbose-1)
            return "save"
        elif key == ord('z') and ctrl_z_stack: # reverse
            self.refresh_images = True
            self.result_image, self.segmented_image, self.painted_pixels = ctrl_z_stack.pop()
        elif key == ord('r'): # reset
            self.refresh_images = True
            ctrl_z_stack.append((self.result_image.copy(), self.segmented_image.copy(), self.painted_pixels.copy()))
            self.reset_images()
        elif key == ord('d'): # debug
            if cv2.getWindowProperty("Segmented Image(Debug)", cv2.WND_PROP_VISIBLE) > 0:
                cv2.destroyWindow("Segmented Image(Debug)")
                cv2.destroyWindow("Painter Pixels(Debug)")
            else:
                cv2.imshow("Segmented Image(Debug)", (self.segmented_image).astype(np.uint8))
                cv2.imshow("Painter Pixels(Debug)", (self.painted_pixels*255).astype(np.uint8))
        elif key == ord('t'): # template match
            return "template"

    def process_color_picker_input(self, color_info, previous_color):
        """selects color

        Args:
            color_info (dictionary): contains color selection informartion
            previous_color (list): value of previous color

        Returns:
            tuple: color for next function call
        """
        if color_info['clicked']: # color selecting
            click_column = color_info['x']
            click_row = color_info['y']
            color = self.color_picker_image[click_row, click_column]
            color_info["clicked"] = False
            return color
        else:
            return previous_color

    def take_action(self, ctrl_z_stack, color, callback_info, action_type=""):
        """processes taken action

        Args:
            ctrl_z_stack (list): list of changes in case of reversing
            color (list): values of selected color
            callback_info (dictionary): contains selected action information

        Returns:
            tuple: previous result and painted pixel images
        """
        ctrl_z_stack.append((self.result_image.copy(), self.segmented_image.copy(), self.painted_pixels.copy()))
        
        if callback_info["continuous_filling"] or callback_info["continuous_unfilling"]: # if one of continuous modes is on
            click_column = callback_info['x']
            click_row = callback_info['y']
            
            if callback_info["continuous_filling"]:                
                fill(self.result_image, self.segmented_image, self.painted_pixels, click_row, click_column, color)
            
            elif callback_info["continuous_unfilling"]:
                unfill(self.result_image, self.painted_pixels, self.raw_image, click_row, click_column)

        if callback_info['clicked']: # if a clicking action detected
            callback_info["clicked"] = False
            click_column = callback_info['x']
            click_row = callback_info['y']

            if callback_info["action"] == "fill": # fill the thing at pos: [click_row, click_column]
                fill(self.result_image, self.segmented_image, self.painted_pixels, click_row, click_column, color)
                
            elif callback_info["action"] == "unfill": # unfill the thing at pos: [click_row, click_column]
                unfill(self.result_image, self.painted_pixels, self.raw_image, click_row, click_column)
                
            elif callback_info["action"] == "cut": # cut the segments with a line
                if callback_info["first_cut"] != None and callback_info["second_cut"] != None:
                    line_image = np.zeros_like(self.result_image) # image to cut segment
                    cv2.line(line_image, callback_info["first_cut"] , callback_info["second_cut"], (255,255,255), 1) 
                    self.result_image[line_image==255] = self.raw_image[line_image==255]
                    self.segmented_image[line_image[:,:,0]==255] = 0
                    self.painted_pixels[line_image[:,:,0]==255] = 0

        if action_type == "template":
            put_template_segments(self.raw_image, self.result_image, self.painted_pixels, self.temp_att_seg_mask)

        previous_result_image, previous_segmented_image, previous_painted_pixels = ctrl_z_stack.pop()
        if np.any(np.equal(previous_result_image, self.result_image) == False): # there is a change
            self.refresh_images = True
            ctrl_z_stack.append((previous_result_image.copy(), previous_segmented_image.copy(), previous_painted_pixels.copy()))

    def segment_image(self, file_no, verbose=0):
        """segments given image and saves it to output folder

        Args:
            file_no (int), index of current file
            verbose (int, optional): verbose level. Defaults to 0.

        Returns:
            str: information for upper function about actions that it needs to take
        """
        self.result_image = self.raw_image.copy()
        # set windows and mouse event listeners for windows
        cv2.namedWindow("Processed Image " + str(file_no), flags = cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        cv2.imshow("Processed Image " + str(file_no), self.result_image)
        
        cv2.namedWindow("Color Picker", flags = cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        cv2.imshow("Color Picker", self.color_picker_image)

        callback_info = {'clicked': False, 'x': -1, 'y': -1, 'action':"", "first_cut":None, "second_cut":None, 'continuous_filling': False, 'continuous_unfilling': False}
        cv2.setMouseCallback("Processed Image " + str(file_no), self.click_event_listener, callback_info)

        color_info = {'clicked': False, 'x': -1, 'y': -1}
        cv2.setMouseCallback("Color Picker", self.color_event_listener, color_info)

        ctrl_z_stack = [] # stack for reversing actions
        color = [0,0,0] # BGR values

        while True:
            key = cv2.waitKey(1)
            self.display_color_picker(callback_info, color_info)
            action = self.process_keyboard_input(file_no, ctrl_z_stack, key, verbose=0)
            if action and action != "template":
                return action
            
            color = self.process_color_picker_input(color_info, color)
            self.take_action(ctrl_z_stack, color, callback_info, action_type=action)
            self.display_images(file_no)

    def process(self, region_size=40, ruler=30, k=15, color_importance=5, verbose=0):
        """function to segment images in a folder in order and save them to output folder

        Args:
            region_size (int, optional): regions_size parameter for cv2 superpixel. Defaults to 40.
            ruler (int, optional): ruler parameter for cv2 superpixel. Defaults to 30.
            k (int, optional): k parameter for cv2 kmeans. Defaults to 15.
            color_importance (int, optional): color importance parameter for cv2 kmeans. Defaults to 5.
            verbose (int, optional): verbose level. Defaults to 0.
        """
        file_no = 0
        threads = {}
        while 0 <= file_no < len(self.files):
            image_path = self.files[file_no]

            # if file is not processed yet, start a process for it
            if file_no not in threads.keys():
                thread = self.create_thread(file_no, region_size, ruler, k, color_importance, verbose=verbose-1)
                threads[file_no] = thread

            # if files process is done start segmenting it
            if image_path in self.segmented_image_dict.keys() and self.segmented_image_dict[image_path] is not None:
                raw_image, orig_segmented_image = self.segmented_image_dict[image_path]                
                self.empty_images()
                self.set_images(raw_image, orig_segmented_image)
                return_code = self.segment_image(file_no, verbose=verbose-1)
                
                if return_code == "next" or return_code == "save": # n or s
                    file_no = (file_no + 1)%len(self.files)
                elif return_code == "previous": # p
                    file_no = (file_no - 1)%len(self.files)
                elif return_code == "quit": # q
                    break
                
        self.thread_stop = True
        cv2.destroyAllWindows()

    def __call__(self):
        """calling the object will start the main process and catch any possible exception during
        """
        try:
            os.makedirs(self.save_folder, exist_ok=True)
            self.process(self.region_size, self.ruler, self.k, self.color_importance, verbose=self.verbose-1)
        except ErrorException as ee:
            print(ee.message)
            exit(ee.error_code)
        except WrongTypeException as wte:
            print(wte.message)
            exit(wte.error_code)