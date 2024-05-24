import cv2
import numpy as np
from skimage.morphology import flood_fill

from segment_anything import SamPredictor, SamAutomaticMaskGenerator

from .helper_exceptions import SAMPromptGenerationQuitException

class SAMSegmentator():
    def __init__(self, SAM, device:str="cpu", verbose: int=0):
        """initializing SAMSegmentator object

        Args:
            SAM (SamPredictor model): SAM model
            device (str, optional): device to use. Defaults to "cpu".
        """
        self.SAM = SAM
        self.device = device
        self.SAM_setted_image = None
        self.verbose = verbose
        self.DRAW_BG = {"color" : [0,0,255], "val" : 0} # right click
        self.DRAW_FG = {"color" : [0,255,0], "val" : 1} # left click
        self.reset()

    # resets SAMSegmentator attributes
    def reset(self, verbose: int=0):
        """resetting SAMSegmentator object variables

        Args:
            verbose (int, optional): verbose level. Defaults to 0.
        """
        self.paint_dict = None
        self.clicked = False                # flag for drawing action
        self.currently_drawing_box = False # flag for box action
        self.prompt_boxes = []              # selected boxes for prompting
        self.prompt_coords = []             # selected coords for prompting
        self.prompt_labels = []             # selected labels for prompting
        self.ctrl_z_stack = []              # to store and reverse annotations

    # listens for user input
    def annotation_event_listener(self, event, x:int, y:int, flags, param, verbose: int=0):
        """mouse callbacks for annotation types

        Args:
            event (opencv event): mouse event to detect
            x (int): column coordinate of mouse
            y (int): row coordinate of mouse
            flags (opencv flags): flags
            param (dictionary): parameters
            verbose (int, optional): verbose level. Defaults to 0.
        """
        # box selection with middle button
        if event == cv2.EVENT_MBUTTONDOWN:
            self.currently_drawing_box = True
            self.ix, self.iy = x,y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.currently_drawing_box:
                self.image = self.altered.copy()
                for r in self.prompt_boxes:
                    cv2.rectangle(self.image, (r[0], r[1]), (r[2], r[3]), [255,0,0], 1)
                cv2.rectangle(self.image, (self.ix, self.iy), (x, y), [255,0,0], 1)
        elif event == cv2.EVENT_MBUTTONUP:
            self.ctrl_z_stack.append((self.prompt_boxes.copy(), self.prompt_coords.copy(), self.prompt_labels.copy()))
            self.currently_drawing_box = False
            self.draw_annotations(box_x=x, box_y=y)
            x = max(x, 0) # clips negatives to zero
            x = min(max(x, 0), self.image.shape[1]) # clips out of bounds values to max value
            y = max(y, 0) # clips negatives to zero
            y = min(max(y, 0), self.image.shape[0]) # clips out of bounds values to max value
            self.prompt_boxes.append(np.array([self.ix, self.iy, x, y]))
            self.altered = self.image.copy()
        if self.currently_drawing_box:
            return

        # annotation type selection with left/right click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked = True
            self.paint_dict = self.DRAW_FG
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clicked = True
            self.paint_dict = self.DRAW_BG
        elif self.clicked and (event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP):
            self.ctrl_z_stack.append((self.prompt_boxes.copy(), self.prompt_coords.copy(), self.prompt_labels.copy()))
            self.clicked = False
            self.draw_annotations(click_x=x, click_y=y)
            self.prompt_coords.append([x,y])
            self.prompt_labels.append(self.paint_dict["val"])
            self.image = self.altered.copy()
    
    # draws annotations on image
    def draw_annotations(self, box_x:int=None, box_y:int=None, click_x:int=None, click_y:int=None, verbose: int=0):
        """draws currently active annotations

        Args:
            box_x (int, optional): x coords input for new box annotation,
            None means no box annotation is provided when it is called Defaults to None.
            box_y (int, optional): y coords input for new box annotation,
            None means no box annotation is provided when it is called Defaults to None.
            click_x (int, optional): x coords input for new point annotation,
            None means no point annotation is provided when it is called Defaults to None.
            click_y (int, optional): y coords input for new point annotation,
            None means no point annotation is provided when it is called Defaults to None.
            verbose (int, optional): verbose level. Defaults to 0.
        """
        # if there is a new annotation
        for r in self.prompt_boxes:
            cv2.rectangle(self.altered, (r[0], r[1]), (r[2], r[3]), [255,0,0], 1)
        if box_x:
            cv2.rectangle(self.altered, (self.ix, self.iy), (box_x, box_y), [255,0,0], 1)

        for (past_x,past_y), past_v in zip(self.prompt_coords, self.prompt_labels):
            # select color according to point label
            if past_v == 0:
                past_c = [0,0,255]
            elif past_v == 1:
                past_c = [0,255,0]
            cv2.circle(self.altered, (past_x,past_y), 3, past_c, -1)
        if click_x:
            cv2.circle(self.altered, (click_x,click_y), 3, self.paint_dict["color"], -1)

    # calculates binary mask from prompt predicted masks
    def get_mask_from_prompt(self, image, prompt_boxes: list, prompt_coords: list, prompt_labels: list, verbose:int=0):
        """creates binary mask with given prompt

        Args:
            image (numpy.ndarray): image to generate mask for
            prompt_boxes (list): box annotations
            prompt_coords (list): point annotations
            prompt_labels (list): point annotation labels
            verbose (int, optional): verbose level. Defaults to 0.

        Returns:
            numpy.ndarray: binary mask
        """
        # assigning coords and labels to their boxes
        # since multiple boxes and multiple coords/labels are not supported each box will be passed with
        # its own related coords and labels individualy
        coords_list = [np.array([pc for pc in prompt_coords 
                    if ((r[0]<=pc[0]<=r[2]) and (r[1]<=pc[1]<=r[3]))])for r in prompt_boxes]
        label_list = [np.array([prompt_labels[e] for e,pc in enumerate(prompt_coords)
                    if ((r[0]<=pc[0]<=r[2]) and (r[1]<=pc[1]<=r[3]))]) for r in prompt_boxes]

        # set the image
        if not np.array_equal(self.SAM_setted_image, image):
            self.SAM.set_image(image)
            self.SAM_setted_image = image.copy()

        # segment each pass individualy with its prompt
        masks = []
        for (c, l, b) in zip(coords_list, label_list, prompt_boxes):
            if len(c) == 0:
                c = None
            if len(l) == 0:
                l = None
            so = self.SAM.predict(point_coords=c, point_labels=l, box=b)
            masks.append(so[0][0])
        # generate one mask and get labels
        final_mask = np.logical_or.reduce(masks)
        return final_mask

    # generated prompt with user input for prompted mask prediction
    def generate_mask(self, image, verbose: int=0):
        """function to interactively generate SAM mask

        Args:
            image (numpy.ndarray): original image

        Returns:
            numpy.ndarray: mask
        """
        self.reset()
        self.original = image.copy()
        self.image = self.original.copy()
        self.altered = self.original.copy()

        cv2.namedWindow("Annotations", flags= cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback("Annotations", self.annotation_event_listener)
        mask = None
        
        while True:
            cv2.imshow("Annotations", self.image)
            self.draw_annotations()
            key = cv2.waitKey(1)

            # key bindings
            if key == ord("q"):
                cv2.destroyWindow("Annotations")
                cv2.destroyWindow("Mask")
                raise(SAMPromptGenerationQuitException("SAMSegmentator received key q for quitting"))
            if key == ord(" "):
                mask = self.get_mask_from_prompt(self.original, self.prompt_boxes, self.prompt_coords, self.prompt_labels)
                display_image = self.original.copy()
                display_image[mask == 0] = 0
                cv2.imshow("Mask", display_image)
            if key == ord("f"):
                cv2.destroyWindow("Annotations")
                cv2.destroyWindow("Mask")
                if mask is None:
                    mask= self.get_mask_from_prompt(self.original, self.prompt_boxes, self.prompt_coords, self.prompt_labels)
                return mask
            elif key == ord("z"): # reverse last annotation
                if len(self.ctrl_z_stack) > 0:
                    last_state = self.ctrl_z_stack.pop()
                    (self.prompt_boxes, self.prompt_coords, self.prompt_labels) = last_state
                    self.image = self.altered = self.original.copy()
            elif key == ord("r"): # reset everything
                self.reset()
                self.image = self.altered = self.original.copy()

    # labels the image segments
    def label_the_segments(self, image, segment_value:int, start_id:int=1):
        """labels the seperate segments with ids

        Args:
            image (numpy.npdarray): image to label
            segment_value (int): which segments to label
            start_id (int, optional): starting id of labels. Defaults to 1.

        Returns:
            numpy.ndarray: labeled image
        """
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
        return image

    # labels the auto predictor sam output mask
    def get_label_from_SAM_auto_output(self, SAM_auto_output, verbose: int=0):
        """creates labeled image from SAM output

        Args:
            SAM_auto_output (list): list of informations about found masks

        Returns:
            numpy.ndarray: labeled image
        """
        # get al masks and mark each of them with an unique id
        masks = [x["segmentation"] for x in SAM_auto_output]
        labeled_image = np.zeros(self.original.shape[:2], dtype=np.int16)
        for e,mask in enumerate(masks):
            labeled_image[mask] = e

        # labeling the backgroun segments
        labeled_image = self.label_the_segments(labeled_image, 0, labeled_image.max()+1)

        return labeled_image
    
    # labels the prompted predicted masks
    def get_label_from_SAM_with_prompt_output_mask(self, SAM_with_prompt_output_mask, verbose: int=0):
        """creates labeled image from SAM output

        Args:
            SAM_with_prompt_output_mask (numpy.ndarray): binary merged mask from SAM output

        Returns:
            numpy.ndarray: labeled image
        """
        labeled_image = np.zeros(self.original.shape[:2], dtype=np.int16)
        # mark masked pixels with -1
        labeled_image[SAM_with_prompt_output_mask] = -1

        # labeling the masked pixels
        labeled_image = self.label_the_segments(labeled_image, -1, 1)
        # labeling the non-masked pixels
        labeled_image = self.label_the_segments(labeled_image, 0, labeled_image.max()+1)

        return labeled_image

    # segments the images
    def segment(self, image_path:str, verbose: int=0):
        """segmentation using SAM model

        Args:
            image_path (str): path to image

        Returns:
            numpy.ndarray: segmented image
        """
        image = cv2.imread(image_path)
        self.original = image.copy()

        if type(self.SAM) == SamPredictor:
            mask = self.generate_mask(image)
            SAM_segment = self.get_label_from_SAM_with_prompt_output_mask(mask)
        
        elif type(self.SAM) == SamAutomaticMaskGenerator:
            # get mask and label the segments
            SAM_auto_output = self.SAM.generate(image)
            SAM_segment = self.get_label_from_SAM_auto_output(SAM_auto_output)
        
        return SAM_segment
    
    def __call__(self, image_path:str):
        return self.segment(image_path, verbose=self.verbose-1)