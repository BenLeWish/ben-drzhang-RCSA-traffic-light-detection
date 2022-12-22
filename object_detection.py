#########################################################################
## Project: Detect and Classify Traffic Lights                         ##
## Author: Ben Wish Le and Dr. Zhang                                   ##
## Date created: October 1st, 2022                                     ##
## Description: This shows how to detect objects like traffic images.  ##
#########################################################################

import tensorflow as tf                 # Machine Learning Lib
from tensorflow import keras            # Neural Network Lib

import numpy as np                      # Scientific Computing lib
import cv2                              # Computer Visionn lib
import glob                             # Filename handling lib           

# Inception V3 model for Keras
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Objection: 1. Use pretrained neural network from COCO data set to detect objects 
#            2. cited source: 
#               a.
#               b.
LABEL_PERSON = 1
LABEL_CAR    = 3
LABEL_BUS    = 6
LABEL_TRUCK  = 8
LABEL_TRAFFIC_LIGHT = 10
LABEL_STOP_SIGN     = 13

def accept_box(boxes, box_index, tolerance):
    """
    ELIMINATE DUNPLICATE BOUNDING BOXES
    """
    box = boxes[box_index]
    
    for index in range (box_index):
        
        other_box = boxes[index]
        m = abs(center(other_box, "x") - center(box,"x")) 
        n = abs(center(other_box, "y") - center(box,"y"))
        
        if m < tolerance and n < tolerance: 
            return False
        
    return True

def get_files(pattern):
    """Create a list of all the images in a directory
    *parameter: pagttern str, gthe pattern of the filenames
    *return: a list of the files that match the specified pattern
    """
    files = []
    
    for file_name in glob.iglob(pattern, recursive = True):     # for each matched files 
        files.append(file_name)                                 # add the image file to the list of files
    return files                                                # return the complete file list

def load_model (model_name):
    """ 1. Download pretrained object detection model
        2. Save it in hard drive
    *Parameter: str name of the pretrained object detection model
    """
    url = 'http:'
    
    model_dir = tf.keras.utils.get_file(fname = model_name, intar = True, origin = url) # download a file from a URL that is not in cache
    
    print("Model path:", str(model_dir))
    
    model_dir = str(model_dir) + "/saved_model"
    model = tf.saved_model.load(str(model_dir))
    
    return model
