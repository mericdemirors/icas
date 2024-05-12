# origin of the code: https://github.com/opencv/opencv/blob/master/samples/python/grabcut.py
import numpy as np
import cv2
from skimage.morphology import flood_fill

from helper_exceptions import *

class GrabcutSegmentor():
    def __init__(self):
        """initializing GrabcutSegmentor object
        """
        self.DRAW_BG = {"color" : [0,0,0], "val" : 0} # right click
        self.DRAW_FG = {"color" : [255,255,255], "val" : 1} # left click
        self.DRAW_PR_BG = {"color" : [40,40,40], "val" : 2} # ctrl + right click
        self.DRAW_PR_FG = {"color" : [200,200,200], "val" : 3} # ctrl + left click
        self.reset()

    def reset(self):
        """resetting GrabcutSegmentor object variables
        """
        self.paint_dict = None
        self.thickness = 3
        self.rect = (0,0,0,0)                # rect x,y,w,h
        self.display_rects = []              # selected rectangles for displaying
        self.segment_rects = []              # selected rectangles for segmenting
        self.currently_drawing = False       # flag for drawing action
        self.currently_drawing_rect = False  # flag for rectangle action
        self.rect_or_mask = -1               # flag for selecting rect or mask mode

    def annotation_event_listener(self, event, x, y, flags, param):
        """mouse callbacks for annotation types

        Args:
            event (opencv event): mouse event to detect
            x (int): column coordinate of mouse
            y (int): row coordinate of mouse
            flags (opencv flags): flags
            param (dictionary): parameters
        """
        # rectangle selection with middle button for grabcut
        if event == cv2.EVENT_MBUTTONDOWN:
            self.currently_drawing_rect = True
            self.ix, self.iy = x,y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.currently_drawing_rect:
                self.image = self.altered.copy()
                for r in self.display_rects:
                    cv2.rectangle(self.image, (r[0], r[1]), (r[2], r[3]), [0,255,0], 1)
                cv2.rectangle(self.image, (self.ix, self.iy), (x, y), [0,255,0], 1)
        elif event == cv2.EVENT_MBUTTONUP:
            self.currently_drawing_rect = False
            for r in self.display_rects:
                cv2.rectangle(self.altered, (r[0], r[1]), (r[2], r[3]), [0,255,0], 1)
            cv2.rectangle(self.altered, (self.ix, self.iy), (x, y), [0,255,0], 1)
            self.display_rects.append((self.ix, self.iy, x, y))
            self.rect = (min(self.ix, x), min(self.iy, y), abs(self.ix - x), abs(self.iy - y))
            self.segment_rects.append(self.rect)
            self.rect_or_mask = 0
            self.altered = self.image.copy()
        if self.currently_drawing_rect:
            return
 
        # annotation type selection with left/right click and CRTL key
        if event == cv2.EVENT_LBUTTONDOWN:
            self.currently_drawing = True
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                self.paint_dict = self.DRAW_PR_FG
            else:
                self.paint_dict = self.DRAW_FG
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.currently_drawing = True
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                self.paint_dict = self.DRAW_PR_BG
            else:
                self.paint_dict = self.DRAW_BG
        elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
            self.currently_drawing = False

        # drawing annotation
        if self.currently_drawing:
            cv2.circle(self.altered, (x,y), self.thickness, self.paint_dict["color"], -1)
            self.image = self.altered.copy()
            cv2.circle(self.mask, (x,y), self.thickness, self.paint_dict["val"], -1)

    def on_trackbar_change(self, value):
        """function to update brush thickness with window trackbar

        Args:
            value (int): new thickness value
        """
        self.thickness = value
   
    def get_segments(self):
        """generate labeled segments from grabcut segmented image

        Returns:
            numpy.ndarray: labeled segments
        """
        segments = self.mask.copy().astype(np.int32)
        segments[segments==2] = 0
        segments[segments!=0] = 1
        segments = segments - 1 # now -1 means background, 0 means foreground

        def label_the_segments(image, segment_value, start_id=1):
            segment_pixels = np.where(image == segment_value)
            segment_id = start_id
            while len(segment_pixels[0]) != 0: # while image has pixels with value 0 which means non-labeled segment
                ri, ci = segment_pixels[0][0], segment_pixels[1][0] # get a segment pixel
                
                image = flood_fill(image, (ri, ci), segment_id, connectivity=1, in_place=True) # floodfill segment
                extracted_segment = np.array(image == image[ri][ci]).astype(np.int16) # extract only segment as binary
                extracted_segment = cv2.dilate(extracted_segment, np.ones((3,3)), iterations=1) # expand segment borders by one pixel to remove edges
                np.putmask(image, extracted_segment != 0, segment_id) # overwrite expanded segment to image

                segment_id = segment_id + 1
                segment_pixels = np.where(image == segment_value)
            return segment_id
        
        last_id = label_the_segments(segments, segment_value=0, start_id=1)
        last_id = label_the_segments(segments, segment_value=-1, start_id=last_id)
        return segments

    def segment(self, file_path):
        """function to interactively segment with grabcut

        Args:
            file_path (str): file path to segment

        Returns:
            numpy.ndarray: labeled segments
        """
        self.image = cv2.imread(file_path)
        self.original = self.image.copy()                             # original copy
        self.altered = self.image.copy()                              # copy to store annotations
        self.mask = np.zeros(self.image.shape[:2], dtype = np.uint8)  # background initialized mask
        self.display = np.zeros(self.image.shape, np.uint8)           # display image

        # annotations and display windows
        cv2.namedWindow("Segments(press 'space' to refine segmentation)")
        cv2.namedWindow("Annotations", flags= cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback("Annotations", self.annotation_event_listener)
        cv2.createTrackbar("brush size","Annotations",self.thickness,100, self.on_trackbar_change)
        cv2.moveWindow("Annotations", self.image.shape[1]+10,90)
        
        while True:
            cv2.imshow("Segments(press 'space' to refine segmentation)", self.display)
            cv2.imshow("Annotations", self.image)
            key = cv2.waitKey(1)

            # key bindings
            if key == ord("q"):
                cv2.destroyWindow("Segments(press 'space' to refine segmentation)")
                cv2.destroyWindow("annotation")
                raise(GrabcutSegmentorQuitException("GrabcutSegmentor received key q for quitting"))
            if key == ord("f"):
                cv2.destroyWindow("Segments(press 'space' to refine segmentation)")
                cv2.destroyWindow("Annotations")
                return self.get_segments()
            elif key == ord("r"): # reset everything
                self.reset()
                self.image = self.altered = self.original.copy()
                self.mask = np.zeros(self.image.shape[:2], dtype = np.uint8)
                self.display = np.zeros(self.image.shape, np.uint8)
            elif key == ord(" "): # segment the image
                bgdmodel = np.zeros((1, 65), np.float64)
                fgdmodel = np.zeros((1, 65), np.float64)
                if (self.rect_or_mask == 0):
                    # calculate each drawed rectangles grabcut and merge them in one
                    merged_mask = self.mask.copy()
                    for r in self.segment_rects:
                        temp_mask = np.zeros_like(merged_mask)
                        cv2.grabCut(self.original, temp_mask, r, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)

                        merged_mask[(temp_mask == 2) & ((merged_mask == 2) | (merged_mask == 0))] = 2
                        merged_mask[(temp_mask == 3) & ((merged_mask == 2) | (merged_mask == 3) | (merged_mask == 0))] = 3
                    self.mask = merged_mask.copy()

                    self.rect_or_mask = 1
                elif (self.rect_or_mask == 1):
                    # grabcut with mask
                    cv2.grabCut(self.original, self.mask, self.rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)
            foreground = np.where((self.mask==1) + (self.mask==3), 255, 0).astype(np.uint8)
            self.display = self.original.copy()
            self.display[foreground == 0] = 0