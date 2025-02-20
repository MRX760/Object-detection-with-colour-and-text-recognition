#after refining the second time on HSV color space

import colorsys
import numpy as np
import cv2
from sklearn.cluster import KMeans

def get_rgb(img_file): #input is img array
    img_rgb = cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB)
    image = img_rgb.copy()

    pixels = image.reshape(-1, 3)

    for i in pixels:
        if np.all(i <= [5, 5, 5]):  # Increase the pixel value to separate background and image
            i[0] = 0
            i[1] = i[0]
            i[2] = i[1]

    num_clusters = 4  # Number of clusters
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(pixels)
    
    labels = kmeans.labels_  # Labels for each pixel
    dominant_colors = np.round(kmeans.cluster_centers_).astype(int)

    unique_values, counts = np.unique(labels, return_counts=True)

    # Remove the black cluster if present
    for i in dominant_colors:
        if np.all(i <= [5, 5, 5]):
            ignored_index = np.where(dominant_colors == i)[0]
            dominant_colors = np.delete(dominant_colors, ignored_index[0], axis=0)
            counts = np.delete(counts, ignored_index[0])
            break

    # Find the index of the second most frequent color
    first_dominant_index = np.argsort(counts)[-1]
    second_dominant_index = np.argsort(counts)[-2]
    last_dominant_index = np.argsort(counts)[-3]
    first_major_color = dominant_colors[first_dominant_index]
    second_major_color = dominant_colors[second_dominant_index]
    last_major_color = dominant_colors[last_dominant_index]
    
    # print("Second Dominant Color:", first_major_color)
    return first_major_color, second_major_color, last_major_color

def rgb_to_hsv(r, g, b):
    # Normalize the RGB values to the range [0, 1]
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0
    
    # Convert the normalized RGB to HSV
    h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
    
    # Convert and round the hue to the range [0, 360]
    h = round(h * 360)
    
    # Convert and round the saturation and value to the range [0, 100]
    s = round(s * 100) #should be 100
    v = round(v * 100)
    
    return h, s, v #should be v instead of 100
    
def classify_color(hue, saturation, brightness):
    color = h_color(hue)
    # print(color)
    color = s_color(saturation, color, hue)
    # print(color)
    color = b_color(brightness, hue, color)
    # print(color)
    return color

def h_color(hue):
    # hue
    if hue <= 10: #modified after test11 + refining the the HSV color to know the boundaries between red and orange
        return 'red'
    elif hue <= 40:
        return 'orange'
    elif hue <= 70:
        return 'yellow'
    elif hue <= 180:
        return 'green'
    elif hue <= 253:
        return 'blue'
    elif hue <= 305: #modified after test12
        return 'purple'
    elif hue <= 340:
        return 'pink'
    else:
        return 'red'

def s_color(saturation, color, hue):
    # saturation
    if color == 'orange':
        if saturation <= 12: #modified after refining HSV color space
            return 'white'
        elif saturation <= 60: 
            return 'brown'
        else:
            return color
    elif color == 'blue' and hue >= 235:
        if saturation <= 4: #modified after test9
            return 'white'
        elif saturation <= 70:
            return 'purple'
        else:
            return color
    elif color == 'blue' and hue <=190: #modified after test8
        if saturation <= 4: #modified after test9
            return 'white'
        else:
            return 'cyan'
    elif color == 'blue': 
        if saturation <=4: #modified after test9
            return 'white'
        elif saturation <=40:
            return 'blue-gray'
        else:
            return color
    elif color =='red' and hue <=12:
        if saturation <= 4: #modified after test9
            return 'white'
        elif saturation <= 70: #added after refining the HSV color space
            return 'pink-brown'
        elif saturation <= 75: #modified after test8 
            return 'red-brown'
        else:
            return 'red'
        
    elif color == 'red':
        if saturation <= 4: #modified after test9
            return 'white'
        elif saturation <= 35:
            return 'pink-brown' #modified after refining the HSV color space
        elif saturation <= 85: #modified after refining HSV color space
            return 'pink'
        else:
            return color
    elif color =='pink': #added after test12
        if saturation <= 4:
            return 'white'
        elif saturation <= 80: 
            return 'pink-purple'
        else:
            return 'pink'
    elif color == 'green' and hue<=80: #added after test12
        if saturation <=6:
            return 'white'
        elif saturation <= 15:
            return 'green-gray'
        else:
            return color
    else:
        if saturation <= 4: #modified after test9
            return 'white'
        else:
            return color

# brightness
def b_color(brightness, hue, color):
    if color == 'orange':
        if brightness <= 6:
            return 'black'
        elif brightness <= 80:
            return 'brown'
        else:
            return color
    elif color == 'red':
        if brightness <= 6:
            return 'black'
        elif brightness <= 35:
            return 'brown'
        else:
            return color
    elif color == 'green-gray': #added after test12
        if brightness <=5:
            return 'black'
        elif brightness <=11:
            return 'green'
        elif brightness <= 65:
            return 'gray'
        else:
            return 'green' #added after test13
    elif color == 'white':
        if brightness <= 15:
            return 'black'
        elif brightness <= 90: #modified after refining the HSV color space
            return 'gray'
        else:
            return color
    elif color == 'red-brown':
        if brightness <= 6:
            return 'black'
        elif brightness <= 90:
            return 'brown'
        else:
            return 'red' #modified after refining the HSV color space
    elif color == 'pink-brown': #added after refining the HSV color space
        if brightness <= 6:
            return 'black'
        elif brightness <= 90:
            return 'brown'
        else:
            return 'pink'
    elif color == 'blue':
        if brightness <=20: #modified after test 7
            return 'black'
        else:
            return color
    elif color == 'blue-gray':
        if brightness <=20:
            return 'black'
        elif brightness <= 80:
            return 'gray'
        else:
            return 'blue'
    elif color == 'pink' and hue >=328: #added after test8
        if brightness <=10:
            return 'black'
        elif brightness <= 55: #modified after test11
            return 'red'
        else:
            return 'pink'
    elif color == 'pink':
        if brightness <= 15:
            return 'black'
        elif brightness <= 40: 
            return 'purple'
        else:
            return color
    elif color == 'pink-purple': #added after test12
        if brightness <=6:
            return 'black'
        elif brightness <=60:
            return 'purple'
        else:
            return 'pink'
    elif color == 'cyan': 
        if brightness <= 5:
            return 'black'
        elif brightness <= 58:
            return 'green'
        else:
            return 'blue'
    elif color == 'yellow' and hue <=65: #added after test8
        if brightness <=6:
            return 'black'
        elif brightness <= 70:
            return 'green'
        else:
            return color
    elif color == 'purple' and hue <= 250: #added after test10
        if brightness <= 10:
            return 'black'
        elif brightness <= 60:
            return 'gray'
        else:
            return color
    else:
        if brightness <= 6:
            return 'black'
        else:
            return color