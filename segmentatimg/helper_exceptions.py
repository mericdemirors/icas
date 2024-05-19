class ErrorException(Exception): # can be handled by ThreadProcessException
    def __init__(self, message:str):
        self.message = message
        self.error_code = 500
 
    def __str__(self):
        return("Custom exception to signal for notifying caller object that process is terminated with 'e' verbose type")
    
class WrongTypeException(Exception): # can be handled by ThreadProcessException
    def __init__(self, message:str):
        self.message = message
        self.error_code = 2300
    def __str__(self):
        return("Custom exception to signal for notifying caller object that print_verbose() is called with a wrong verbose type")
    
class ColorPickerException(Exception):
    def __init__(self, message:str):
        self.message = message
        self.error_code = 300
    def __str__(self):
        return("Custom exception to signal for notifying caller object about ColorPicker image errors")

class NotMatchingTemplatesAndSegmentsException(Exception):
    def __init__(self, message:str):
        self.message = message
        self.error_code = 1401
    def __str__(self):
        return("Custom exception to signal for notifying caller object that given templates and segments are not matching")

class NotMatchingAttentionAndMasksException(Exception):
    def __init__(self, message:str):
        self.message = message
        self.error_code = 1402
    def __str__(self):
        return("Custom exception to signal for notifying caller object that given attentions and masks are not matching")

class InvalidMethodException(Exception):
    def __init__(self, message:str):
        self.message = message
        self.error_code = 900
    def __str__(self):
        return("Custom exception to signal for notifying caller object that an invalid segmentation method is selected")

class GrabcutSegmentorQuitException(Exception):
    def __init__(self, message:str):
        self.message = message
        self.error_code = 700
    def __str__(self):
        return("Custom exception to signal for notifying caller object that grabcut segmentation has been quitted midway")

class SAMPromptGenerationQuitException(Exception):
    def __init__(self, message:str):
        self.message = message
        self.error_code = 1900
    def __str__(self):
        return("Custom exception to signal for notifying caller object that sam prompt generation has been quitted midway")

class ThreadProcessException(Exception):
    def __init__(self, message:str):
        self.message = message
        self.error_code = 2000
    def __str__(self):
        return("Custom exception to signal for notifying caller object that thread process to segment imagaes in the background is failed")