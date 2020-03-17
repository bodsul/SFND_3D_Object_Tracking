# SFND 3D Object Tracking

Welcome to the final project of the camera course. By completing all the lessons, you now have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at our program schematic to see what we already have accomplished and what's still missing.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks: 
1. First, you will develop a way to match 3D objects over time by using keypoint correspondences. 
2. Second, you will compute the TTC based on Lidar measurements. 
3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.

If you get the error `libc++abi.dylib: terminating with uncaught exception of type cv::Exception: OpenCV(4.1.0) /tmp/opencv-20190505- 12101-14vk1fh/opencv-4.1.0/modules/dnn/src/darknet/darknet_io.cpp:694: error: (-213:The function/feature is not implemented) Transpose the weights (except for convolutional) is not implemented in function 'ReadDarknetFromWeightsStream'. It is as a result of yolov3.weights file not getting stored correctly on Github. Perhaps a good use case of git-lfs. Please follow the instructions here to fix it https://stackoverflow.com/questions/54785928/opencv-implementation-of-yolo-v3-reproduces-exception-on-a-gcp-instance.`

## Implementation Explanations
In this project we detect lidar point cloud from Lidar, image bounding boxes and keypoints from camera image for a driving scene. This is done frame by frame. We detect image keypoints using several key point detectors (to be added) and match keypoints between consecutive frames using several key point matchers (to be added).

Using camera intrinsic and extrinsic calibration parameters and also the lidar calibration parameters we are able to
project lidar points to the image plane. This gives a correspondence between image bounding boxes and lidar points. These parts of the project were already implemented for us or in previous sections of the course.

Below we describe extra implementations added in `src/camFusion_Student.cpp`:

We get a correspondence between image bounding boxes and keypoints by assigning each keypoint to all bounding boxes
they fall into. Similarly, we have a correspondence between image bounding boxes and keypoint matches. This is implemented in `clusterKptMatchesWithROI`.

Using the association of bounding boxes to keypoints and keypoint matches, we can match bounding boxes in corresponding frames. First for each bounding box in the current frame we predict matched key points to
the previous frame using the matches in the ROI of the bounding box. Next we compute the IOU of the predicted key points with the keypoints in the ROI of the bounding boxes in the previous frame. This gives a score between all pair
of bounding boxes in the current frame and the previous frame. Using the score, we implement a standard association algorithm. The details can be found in `matchBoundingBoxes`.

Using the matched bounding boxes and lidar points that project into the bounding boxes, we get can estimate a trajectory in 3D space which can be used to estimate a lidar based TTC. The details are implemented in `computeTTCLidar`.

Add descriptions of `computeTTCCamera`, descriptions of examples where TTC estimate of the Lidar sensor does not seem plausible and description of comparison of the performance of the different keypoint detection and matching algorithms after it is clarified what `computeTTCCamera` should be doing.