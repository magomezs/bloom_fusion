import sys
import keyboard
from PIL import Image
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False
import numpy as np
import PIL.Image
import time
import functools
import tensorflow_hub as hub
import rawpy
import cv2
from os import listdir
from os.path import isfile, join


#--------------------------------------------------functions------------------------------------------------------------




def get_bbox(image):
    global posList1, posList2
    posList1 = []
    posList2 = []
    def onMouse(event, x, y, flags, param):
        #global posList
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(posList1)>len(posList2):
                posList2.append((x, y))
            else:
                posList1.append((x, y))
    factor=4;
    resized_im= cv2.resize(image, (0, 0), fx=1.0/factor, fy=1.0/factor)
    cv2.namedWindow('source_image')
    cv2.setMouseCallback('source_image', onMouse)
    cv2.imshow('source_image', resized_im)
    cv2.waitKey();
    posNp1 = np.array(posList1)  # convert to NumPy for later use
    posNp2 = np.array(posList2)  # convert to NumPy for later use
    print(posNp1)
    print(posNp1*factor)
    return [posNp1*factor, posNp2*factor]


def crop_images_set (source_directory, output_directory):
    # get files
    source_files = [f for f in listdir(source_directory) if isfile(join(source_directory, f))]
    im_number=1;
    for source_file in source_files:
        print(source_file)
        try:
            source_im_path = os.path.join(source_directory, source_file)
            if (source_im_path.count('.CR2')>0):
                print('1')
                raw = rawpy.imread(source_im_path)  # access to the RAW image
                print('2')
                rgb = raw.postprocess()  # a numpy RGB array
                print('3')
                source_im = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # the OpenCV image
                print('4')
            else:
                source_im= cv2.imread(source_im_path)

            imagePoints = get_bbox(source_im)
            points1 = imagePoints[0]
            points2 = imagePoints[1]
            for point1, point2 in zip(points1, points2):
                try:
                    cropped = source_im[point1[1]:point2[1], point1[0]:point2[0]]
                    output_im_path = os.path.join(output_directory, str(im_number) + '.jpg')
                    print(output_im_path)
                    cv2.imwrite(output_im_path, cropped)
                    im_number = im_number + 1;
                except:
                    break
        except:
            break

        #try:  # used try so that if user pressed other than the given key error will not be shown
        #    if keyboard.is_pressed('n'):  # if key 'q' is pressed
        #        print('You Pressed A Key!')
        #        break  # finishing the loop
        #except:
        #    break  # if user pressed a key other than the given key the loop will break




#-----------------------------------------------------main--------------------------------------------------------------

source_directory = '/home/ia_vision/PycharmProjects/StyleTransfer/style_images/cianobacterias_samples'
output_directory = '/home/ia_vision/PycharmProjects/StyleTransfer/style_images/set1'
crop_images_set(source_directory, output_directory)






#style_image=style_image[:, 150:340, 100:510, :]