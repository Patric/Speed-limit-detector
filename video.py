import pathlib
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

class video:
    file_loc = 0
    def __init__(self, file_loc):
        #checking if the path is correct
        self.file_loc = file_loc


    def capture(self):
        while not os.path.exists(self.file_loc):
            print(f"Current location: {pathlib.Path().absolute()}")
            self.file_loc = input('Enter .mp4 or .avi file location or type quit: ')
            if self.file_loc == "quit":
                return False

        if self.file_loc != "quit":
            return cv2.VideoCapture(self.file_loc)