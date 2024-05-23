# GrabcutSegmentator
Interactive segmentation with [opencv's grabcut](https://docs.opencv.org/4.x/d3/d47/group__imgproc__segmentation.html#ga909c1dda50efcbeaa3ce126be862b37f) method.

## Functions

### * *__init__(self)*
creates GrabcutSegmentator object and initializes the object attributes.

### * *reset(self)*
resets the updatable attributes.

### * *def annotation_event_listener(self, event, x:int, y:int, flags, param)*
listens for user mouse inputs for 'Annotations' window.
* event: opencv event that is happening
* x: column coordinate of mouse
* y: row coordinate of mouse
* flags: opencv flags
* param: placeholder for any additional parameters

Rectangle is drawn with mouse middle button. Rectangles will be attention area for the grabcut algorithm. Foreground and background annotations are done with left and right mouse click. Holding down the CTRL key turns them to probable foreground and background.

### * *def on_trackbar_change(self, value:int)*
updates the brush size for mouse annotations with a slider at 'Annotations' window.
* value: new value for brush size  

### * *def get_segments(self)*
generates labeled segments from binary mask.  

returns image with labeled segments

### * *def segment(self, file_path:str)*
segments the passed image. Two window is displayed for user: 'Segments' and 'Annotations'. 'Segments' is for displaying the current segments of image to user. 'Annotations' is for displaying the current annotations to user.
* file_path: path to image  

Four type of keyboard input is accepted during segmentation:
* q: quits the segmentation by raising a 'GrabcutSegmentatorQuitException'
* f: finishes the segmentation and returns the labeled segments of image
* r: resets the all attributes
* space: runs grabcut algorithm once with current annotations. Multiple runs can be needed for convergence  

returns image with labeled segments


### * *def __call__(self, file_path:str)*
calls self.segment() function with given parameter.
* file_path: path to image  

## Attributes
* DRAW_BG: dictionary for background mouse annotation
* DRAW_FG: dictionary for foreground mouse annotation
* DRAW_PR_BG: dictionary for probable background mouse annotation
* DRAW_PR_FG: dictionary for probable foreground mouse annotation
* paint_dict: current dictionary for annotation
* brush_size: brush size
* rect: current rectangle annotation
* display_rects: rectangles for displaying
* segment_rects: rectangles for segmenting
* currently_drawing: flag for foreground/background annotation
* currently_drawing_rect: flag for rectangle annotation
* rect_or_mask: flag for selecting rect or mask mode at grabcut
* ix: initial column coordinate of mouse at annotation 
* iy: initial row coordinate of mouse at annotation 
* image: displayed image at 'Annotations' window
* altered: image to store annotations
* mask: binary segmented mask
* original: original image
* display: displayed image at 'Segments' window  

<br/><br/>
<br/><br/>

# SAMSegmentator
Interactive segmentation with [Meta's Segment Anything Model](https://github.com/facebookresearch/segment-anything).

## Functions

### * *__init__(self, SAM, device:str="cpu", verbose: int=0)*
creates SAMSegmentator object and initializes the object attributes.

### * *reset(self, verbose: int=0)*
resets the updatable attributes.

### * *annotation_event_listener(self, event, x:int, y:int, flags, param, verbose: int=0)*
listens for user mouse inputs for 'Annotations' window.
* event: opencv event that is happening
* x: column coordinate of mouse
* y: row coordinate of mouse
* flags: opencv flags
* param: placeholder for any additional parameters

Rectangle is drawn with mouse middle button. Rectangles will be boxes for the SAM model with prompt. Foreground and background point annotations are done with left and right mouse click.

### * *draw_annotations(self, box_x:int=None, box_y:int=None, click_x:int=None, click_y:int=None, verbose: int=0)*
draws currently active annotations
* box_x: x coords for newly annotated box annotation if there is any
* box_y: y coords for newly annotated box annotation if there is any
* click_x: x coords for newly annotated click annotation if there is any
* click_y: y coords for newly annotated click annotation if there is any

### * *get_mask_from_prompt(self, image, prompt_boxes: list, prompt_coords: list, prompt_labels: list, verbose:int=0)*
predicts the masks with SamPredictor, then creates the segmented mask
* image: image to predict over
* prompt_boxes: box prompts for SamPredictor
* prompt_coords: coord prompts for SamPredictor
* prompt_labels: label prompts for SamPredictor
Since multiple boxes with multiple points are not supported, each box is processed with its related coords and then all generated masks are merged into one  

returns generated mask

### * *generate_mask(self, image, verbose: int=0)*
interactively creates SAM mask. Two window is displayed for user: 'Mask' and 'Annotations'. 'Mask' is for displaying the current segments of image to user and will be visible after first mask generation. 'Annotations' is for displaying the current annotations to user.
* image: image to generate mask for
Four type of keyboard input is accepted during segmentation:
* q: quits the segmentation and raises SAMPromptGenerationQuitException exception
* space: generated segmentation
* f: finishes the segmentation and returns the labeled segments of image
* z: reverses the last annotation
* r: resets the annotations  

returns generated mask


### * *label_the_segments(self, image, segment_value:int, start_id:int=1)*
generates labeled segments from mask.
* image: mask to label
* segment_value: value to search and label in the image
* start_id: starting value for segment labeling  

returns image with labeled segments

### * *get_label_from_SAM_auto_output(self, SAM_auto_output, verbose: int=0)*
creates labeled segments from SamAutomaticMaskGenerator
* SAM_auto_output: output ofr the SamAutomaticMaskGenerator model  

returns image with labeled segments


### * *get_label_from_SAM_with_prompt_output_mask(self, SAM_with_prompt_output_mask, verbose: int=0)*
creates labeled segments from SamPredictor
* SAM_auto_output: output ofr the SamPredictor model  

returns image with labeled segments


### * *segment(self, image_path:str, verbose: int=0)*
calls the mask generation functions according to SAM model type
* image_path: path to image  

returns segmented image

### * *__call__(self, image_path:str)*
calls self.segment() function with given parameter.
* image_path: path to image  

returns segmented image

## Attributes
* SAM: SAM model
* device: device to run model on
* SAM_setted_image: SAM models current image
* DRAW_BG: dictionary for background mouse annotation
* DRAW_FG: dictionary for foreground mouse annotation
* paint_dict: current dictionary for annotation
* clicked: flag for foreground/background annotation
* currently_drawing_box: flag for rectangle annotation
* prompt_boxes: annotated box prompts
* prompt_coords: annotated coord prompts
* prompt_labels: annotated label prompts
* ctrl_z_stack: stack to reverse the last annotation
* ix: initial column coordinate of mouse at annotation 
* iy: initial row coordinate of mouse at annotation 
* image: displayed image at 'Annotations' window
* altered: image to store annotations
* original: original image

<br/><br/>
<br/><br/>

# Segmentator

## Functions

### * *__init__(self, image_folder:str, color_picker_image_path:str=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ColorPicker.png"), method:str="", templates_path="", attentions_path="", segments_path="", masks_path="", thread_range:int=10, template_threshold:float=None, edge_th:int=60, bilateral_d:int=7, sigmaColor:int=100, sigmaSpace:int=100, templateWindowSize:int=7, searchWindowSize:int=21, h:int=10, hColor:int=10, region_size:int=40, ruler:int=30, k:int=15, color_importance:int=5, number_of_bins:int=20, segment_scale:int=100, sigma:float=0.5, min_segment_size:int=100, segment_size:int=100, color_weight:float=0.5, SAMSegmentator=None, verbose:int=0)*
creates Segmentator object and initializes the object attributes.
* image_folder: path to images
* color_picker_image_path: path to color picking image
* method: segmentation method
* templates: path to folder of templates to search in raw images
* attentions: path to folder of template masks to where to pay attention, will be derived from templates if not provided
* segments: path to folder of segments to paint detected templates
* masks: path to folder of segment masks to where to paint, will be derived from segments if not provided
* thread_range: depth of image processings at previous and upcoming images on list
* template_threshold: max error rate to consider a template as matched, if None, best match is considered
* edge_th: threshold to consider a pixel as edge
* bilateral_d: window size for cv2.bilateral()
* sigmaColor: color strength for cv2.bilateral()
* sigmaSpace: distance strength for cv2.bilateral()
* templateWindowSize: window size for cv2.fastNlMeansDenoisingColored()
* searchWindowSize: window size for cv2.fastNlMeansDenoisingColored()
* h: noise remove strenght for cv2.fastNlMeansDenoisingColored()
* hColor: color noise remove strenght for cv2.fastNlMeansDenoisingColored()
* region_size: region_size parameter for superpixel
* ruler: ruler parameter for superpixel
* k: k parameter for opencv kmeans or graph segmentation
* color_importance: importance of pixel colors proportional to pixels coordinates for kmeans
* number_of_bins: number of segments to extract from chan vase method output
* segment_scale: segment scale parameter for felzenszwalb
* sigma: standard deviation of Gaussian kernel in felzenszwalb or sigma parameter for graph segmentation
* min_segment_size: min size of a segment for felzenszwalb or graph
* segment_size: size of segments for felzenszwalb, quickshift or graph
* color_weight: weight of color to space in quickshift
* SAMSegmentator: SAMSegmentator object for SAM method


### * *__str__(self)*
to string method  

returns explanatory string

### * *arguman_check(self, templates:list, attentions:list, segments:list, masks:list, verbose:int=0)*
checks arguman validity. Ensures number of images are same, passed method is valid and color_picker_image exists
* templates: template files for template matching
* attentions: attention files for template matching
* segments: segment files for template matching
* masks: mask files for template matching


### * *empty_images(self)*
empties the image attributes

### * *reset_images(self)*
resets the image attributes

### * *set_images(self, raw_image, orig_segmented_image)*
sets the image attributes
* raw_image: nonprocessed image
* orig_segmented_image: nonprocessed segmented image

### * *display_images(self, file_no:int)*
refreshs image displays
* file_no: index of image

### * *click_event_listener(self, event, x:int, y:int, flags, callback_info:dict)*
listens for user mouse inputs for 'Processed Image' window.
* event: opencv event that is happening
* x: column coordinate of mouse
* y: row coordinate of mouse
* flags: opencv flags
* callback_info: dictionary to store user input info

### * *color_event_listener(self, event, x:int, y:int, flags, color_info:dict)*
listens for user mouse inputs for 'Color Picker' window.
* event: opencv event that is happening
* x: column coordinate of mouse
* y: row coordinate of mouse
* flags: opencv flags
* callback_info: dictionary to store user input info

### * *display_color_picker(self, callback_info:dict, color_info:dict)*
displays color picker image and shows mode information
* callback_info: dictionary to store user input info
* color_info: dictionary to store user input info

### * *create_thread(self, file_no:int, verbose:int=0)*
creates new thread to process the upcoming images. Image segmentation processes are done in the background if possible to prepare upcoming images
* file_no: index of image  

returns created thread

### * *save_masks(self, mask_path:str, result_image, painted_pixels, verbose:int=0)*
saves masks according to paint color
* mask_path: incomplate path to save masks
* result_image: painted image
* painted_pixels: image to store which pixels are painted

### * *process_keyboard_input(self, file_no:int, ctrl_z_stack:list, key, verbose:int=0)*
processes keyboard input.
* file_no: index of image
* ctrl_z_stack: stack to reverse the last annotation
* key: keyboard input
Eight different inputs are accepted:
* q: quits the segmentation, returns 'quit' action
* n: passes the next image without saving current one, returns 'next' action
* p: passes the previouse  image without saving current one, returns 'previous action'
* space: saves the masks and passes the next one, returns 'save' action
* z: reverses the last annotation
* r: resets the all attributes
* d: opens debug mode to show image segmentation in 'Segmented Image' window and painted pixels in 'Painted Pixels' window
* t: applies template matching  

returns action corresponding to keyboard input

### * *process_color_picker_input(self, color_info:dict, previous_color)*
processes input to get color 
* color_info: user input info
* previous_color: previous selected color  

returns selected color

### * *take_action(self, ctrl_z_stack:list, color, callback_info:dict, action_type:str="")*
applies selected action
* ctrl_z_stack: stack to reverse last annotation
* color: selected color
* callback_info: user input info
* action_type: action to process
There are five annotation type:
* continuous_filling: fills the hovered segments
* continuous_unfilling: unfills the hovered segments
* fill: fills the clicked segment
* unfill: unfills the clicked segment
* cut: cuts between middle button clicks
There are five actions(only template action covered in take_action() function, others are covered in process() function):
* quit: quits the segmentation
* next: passes to next image without saving
* previous: passes to previous image without saving
* save: saves image and passes to next image
* template: applies template matching

### * *manual_segmenting(self, file_no:int, verbose:int=0)*
function to capsulate manual segmentation process. Two window is displayed for user: 'Processed Image' and 'Color Picker'. 'Processed Image' is for displaying the current segments of image to user. 'Color Picker' is for picking the painting color. Also two additional window named 'Segmented Image' and 'Painted Pixels' for debug purposes can be toggled on and off.
* file_no: index of image  

returns action

### * *process(self, verbose:int=0)*
function to capsulate all segmentation process.

### * *grabcut_process(self, verbose:int=0)*
function to capsulate all segmentation process, specialized to grabcut since grabcut doesnt run with threads.

### * *SAM_process(self, verbose:int=0)*
function to capsulate all segmentation process, specialized to SAM since SAM doesnt run with threads.

### * *__call__(self)*
calls process() functions, and catches exceptions to close threads before terminating

## Attributes
* image_folder: path to folder that contains image
* files: list of image paths to segment
* color_picker_image_path: path to color_picker_image
* method: method to use for segmentation
* template_threshold: threshold value for template matching, None results in selection of best match
* threads: dictionary to store threads
* temp_att_seg_mask: list that holds images for template macthing
* refresh_images: boolean to decide when to refresh images
* save_folder: path to folder to save masks
* segmented_image_dict: dictionary to store segmented images
* thread_range: controls how many threads will word to prepare upcoming images
* thread_stop: boolean to decide when to stop threads
* segment_image_parameters: parameters for segmentation methods
* SAMSegmentator: SAM model
* color_picker_image: image to pick painting colors
* raw_image: nonprocessed image
* result_image: painted image
* segmented_image: segmented image
* painted_pixels: image to store which pixels are painted
* orig_raw_image: original nonprocessed image
* orig_result_image: original painted image
* orig_segmented_image: original segmented image
* orig_painted_pixels: original image to store which pixels are painted