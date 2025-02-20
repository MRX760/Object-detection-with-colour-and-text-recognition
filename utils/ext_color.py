
import cv2
import extcolors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import re
import os


def extract_colors(filename, tol):
    filename = cv2.cvtColor(filename, cv2.COLOR_BGR2RGB)
    filename = Image.fromarray(np.uint8(filename))
    colors_x = extcolors.extract_from_image(filename, tolerance=tol, limit=5)
    for i in colors_x[0]:
        if any(j == (0,0,0) or j == (255,255,255) or j == (1,1,1) or j == (2,2,2) for j in i): # delete color with 0,0,0 RGB value (background)
            colors_x[0].remove(i)
    return colors_x

def extract(file, tolerance):
    df_color = extract_colors(file, tolerance)
    return df_color

# home = os.getcwd()
# img = cv2.imread(home+'\\as.jpg')
# extract(img)
