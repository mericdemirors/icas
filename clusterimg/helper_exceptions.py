class FinishException(Exception):
    def __init__(self, message):
        self.message = message
        self.error_code = 600

    def __str__(self):
        return("Custom exception to signal for notifying caller object that process is terminated with 'f' verbose type")
    
class ErrorException(Exception):
    def __init__(self, message):
        self.message = message
        self.error_code = 500
 
    def __str__(self):
        return("Custom exception to signal for notifying caller object that process is terminated with 'e' verbose type")
    
class WrongTypeException(Exception):
    def __init__(self, message):
        self.message = message
        self.error_code = 2300
    def __str__(self):
        return("Custom exception to signal for notifying caller object that print_verbose() is called with a wrong verbose type")