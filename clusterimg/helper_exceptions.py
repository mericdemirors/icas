class FinishException(Exception):
    def __init__(self, message:str):
        self.message = message
        self.error_code = 0 # this exception is considered a 'succesful termination'
    def __str__(self):
        return("Custom exception to signal for notifying caller object that process is terminated with 'f' verbose type")
    
class ErrorException(Exception):
    def __init__(self, message:str):
        self.message = message
        self.error_code = 500
    def __str__(self):
        return("Custom exception to signal for notifying caller object that process is terminated with 'e' verbose type")
    
class WrongTypeException(Exception):
    def __init__(self, message:str):
        self.message = message
        self.error_code = 2300
    def __str__(self):
        return("Custom exception to signal for notifying caller object that print_verbose() is called with a wrong verbose type")
    
class InvalidMethodException(Exception):
    def __init__(self, message:str):
        self.message = message
        self.error_code = 901
    def __str__(self):
        return("Custom exception to signal for notifying caller object that an invalid clustering method is selected")

class InvalidOptionException(Exception):
    def __init__(self, message:str):
        self.message = message
        self.error_code = 902
    def __str__(self):
        return("Custom exception to signal for notifying caller object that an invalid clustering option is selected")

class InvalidTransferException(Exception):
    def __init__(self, message:str):
        self.message = message
        self.error_code = 903
    def __str__(self):
        return("Custom exception to signal for notifying caller object that an invalid file transfer is selected")

class InvalidLossException(Exception):
    def __init__(self, message:str):
        self.message = message
        self.error_code = 904
    def __str__(self):
        return("Custom exception to signal for notifying caller object that an invalid loss type is selected")

class OverwritePermissionException(Exception):
    def __init__(self, message:str):
        self.message = message
        self.error_code = 1500
    def __str__(self):
        return("Custom exception to signal for notifying caller object that folder overwrite permission has not granted")
