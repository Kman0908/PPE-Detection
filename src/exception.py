class CustomException(Exception):
    def __init__(self, error_msg):
        super().__init__(error_msg)
        self.error_msg = error_msg
    
    def __str__(self):
        return str(self.error_msg)