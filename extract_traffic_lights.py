#########################################################################
## Project: Detect and Classify Traffic Lights                         ##
## Author: Ben Wish Le and Dr. Zhang                                   ##
## Date created: October 1st, 2022                                     ##
## Description: This shows how to extract traffic lights from images.  ##
#########################################################################

#Import libs
import cv2                      # computer vision lib
import object_detection         # get methods for object detection in images 


#get a list of jpeg image files containing traffic lights
files = object_detection.get_files('traffic_light_input/*.jpg')

#load the object detection model
this_model =  object_detection.load_ssd_coco()

#keep track of the number of the traffic lights found
traffic_light_count = 0;

#keep track of the number of the image files that were processed
file_count = 0;

#display the count of the nmumber of images we need to processed
print ("Number of images:", len(files))

# DECTECT OBJECT IN THE IMAGES                                          #
# img_rgb is the original image in RGB format                           #
# out is a dictionary containing the results of the object detection    #
# Go thru  each image file, one at a time                               #
# file name is the name of the file                                     #
(img_rgb, out, file_name) = object_detection.perform_object_detection(model=this_model, file_name = file, save_annotated = None, model_traffic_light = None)


for file in files:
    if (file_count %10)==0:                                                 # this is a cluster of 10 files
        print ("images processed:", file_count)                             # show the number of processed files
        print ("Number of traffic lights indentified:", traffic_light_count)# show total numner of indentified traffic lights
        
    file_count = file_count + 1                                             # increase the number of files by 1
 
 # For each traffic light (i.e. bounding box) that was detected   
    for index in range (len(out['boxes'])):                         
        obj_class = out["detection_classes"] [index]                        # extract the type of detected object 
       
        if obj_class == object_detection.LABEL_TRAFFIC_LIGHT:               # if the dectected object is traffic light
            
            box = out["boxes"][index]                                       # extract the coordinates of the bouding box
            traffic_light = img_rgb[box["y"]:box["y2"], box["x"]:box["x2"]] # crop/extract the traffic light from the image
            traffic_light = cv2.cvtColor(traffic_light, cv2.COLOR_RGB2BGR)  # convert the format from RGB to BGR format
            cv2.imwrite("traffic_light_cropped/" + str(traffic_light_count)+".jpg", traffic_light) # store the cropped image in a folder named 'traffic_light_cropped
            traffic_light_count = traffic_light_count + 1                   # increase the number of traffic lights by 1


# show the total  number of the identified traffic lights
print ("Number of traffic lights identified:", traffic_light_count)      
 