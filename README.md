# Computer-Vision-Art-Program-C/C++
A multi-option Art Program using OpenCV. Object Tracking and Identifying. Abstract Art with contours. Posting it public because it is open to be improved and developed. Some possibilities are noted in README.

OpenCV:

The main tool being used in this project is OpenCV.  This is a software that is an open source library for computer vision with over 2500 different algorithms, assisting in building strong infrastructure for various applications. The latest update (OpenCV >= 3.3.0) contains DNN modules that use the Caffe Deep Learning framework, which allows easy accessibility to develop or create programs. Many of the tutorials available online and in the OpenCV library for detecting and tracking objects are based on a given image/video file. We want our program to use a live feed such as the default webcam with proper speed and efficiency and also let the user play around with the live feed.

In our case, OpenCV is primarily used to call colour conversion functions and a few functions that can be manually created from numerous lines of code. For example, we convert our BGR live feed into HSV or Gray by using the cvtColor() function. OpenCV also simplifies the process of creating shapes by calling from classes so we can code rectangles, an ellipse or other shapes for users to prompt in the program. In addition to OpenCV, its caffe model module accesses GoogleNet’s database in order for our identification portion of the program.

Caffe Model:

Caffe Model is an open framework used for deep learning. Written in C++, it supports many different types of deep learning architectures geared towards image classification and image segmentation. 

Caffe has over 600+ citations, 150+ contributors, 4700+ forks and >1 pull request every day 

The reason we choose Caffe instead of Torch or other frameworks is because:
Speed: Caffe has state-of-the-art models and massive data
Modularity:  It is easy to extend new tasks and settings
Openness: It has a common code and reference models for reproducibility
Community: It has a huge community with joint discussion and development through BSD-2 licensing

Improvements

Track and Identify Option:
Integrate YOLO real-time detection with user interaction for faster identification
Toggle the tracker to stop when main colours are not visible in the frame anymore
Prompt program to look for only certain colours in the bounding box histogram
Display coordinates or object information while tracking 

Contour Option:
Create an erase or cutting algorithm to allow users to remove fragments from the final coloured frame
Add a paint/writing feature to the final coloured frame to make more touches
Add an internal option for single object contouring or contour all objects in the frame
Toggle threshold of the live HSV feed with buttons that automatically reduce noise to a satisfying range

This program is very heavily based on imagination and abstract art so programmers can come up with endless improvements to develop the same features or even potentially add more


