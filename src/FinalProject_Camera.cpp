
/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"
#include "objectDetection2D.hpp"
#include "lidarData.hpp"
#include "camFusion.hpp"

using namespace std;

/* MAIN PROGRAM */
void run(string detectorType, string descriptorType, ofstream &TTCCamerafile, ofstream &TTCLidarfile)
{
    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 18;   // last file index to load
    int imgStepWidth = 1; 
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // object detection
    string yoloBasePath = dataPath + "dat/yolo/";
    string yoloClassesFile = yoloBasePath + "coco.names";
    string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    string yoloModelWeights = yoloBasePath + "yolov3.weights";

    // Lidar
    string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
    string lidarFileType = ".bin";

    // calibration data for camera and lidar
    cv::Mat P_rect_00(3,4,cv::DataType<double>::type); // 3x4 projection matrix after rectification
    cv::Mat R_rect_00(4,4,cv::DataType<double>::type); // 3x3 rectifying rotation to make image planes co-planar
    cv::Mat RT(4,4,cv::DataType<double>::type); // rotation matrix and translation vector
    
    RT.at<double>(0,0) = 7.533745e-03; RT.at<double>(0,1) = -9.999714e-01; RT.at<double>(0,2) = -6.166020e-04; RT.at<double>(0,3) = -4.069766e-03;
    RT.at<double>(1,0) = 1.480249e-02; RT.at<double>(1,1) = 7.280733e-04; RT.at<double>(1,2) = -9.998902e-01; RT.at<double>(1,3) = -7.631618e-02;
    RT.at<double>(2,0) = 9.998621e-01; RT.at<double>(2,1) = 7.523790e-03; RT.at<double>(2,2) = 1.480755e-02; RT.at<double>(2,3) = -2.717806e-01;
    RT.at<double>(3,0) = 0.0; RT.at<double>(3,1) = 0.0; RT.at<double>(3,2) = 0.0; RT.at<double>(3,3) = 1.0;
    
    R_rect_00.at<double>(0,0) = 9.999239e-01; R_rect_00.at<double>(0,1) = 9.837760e-03; R_rect_00.at<double>(0,2) = -7.445048e-03; R_rect_00.at<double>(0,3) = 0.0;
    R_rect_00.at<double>(1,0) = -9.869795e-03; R_rect_00.at<double>(1,1) = 9.999421e-01; R_rect_00.at<double>(1,2) = -4.278459e-03; R_rect_00.at<double>(1,3) = 0.0;
    R_rect_00.at<double>(2,0) = 7.402527e-03; R_rect_00.at<double>(2,1) = 4.351614e-03; R_rect_00.at<double>(2,2) = 9.999631e-01; R_rect_00.at<double>(2,3) = 0.0;
    R_rect_00.at<double>(3,0) = 0; R_rect_00.at<double>(3,1) = 0; R_rect_00.at<double>(3,2) = 0; R_rect_00.at<double>(3,3) = 1;
    
    P_rect_00.at<double>(0,0) = 7.215377e+02; P_rect_00.at<double>(0,1) = 0.000000e+00; P_rect_00.at<double>(0,2) = 6.095593e+02; P_rect_00.at<double>(0,3) = 0.000000e+00;
    P_rect_00.at<double>(1,0) = 0.000000e+00; P_rect_00.at<double>(1,1) = 7.215377e+02; P_rect_00.at<double>(1,2) = 1.728540e+02; P_rect_00.at<double>(1,3) = 0.000000e+00;
    P_rect_00.at<double>(2,0) = 0.000000e+00; P_rect_00.at<double>(2,1) = 0.000000e+00; P_rect_00.at<double>(2,2) = 1.000000e+00; P_rect_00.at<double>(2,3) = 0.000000e+00;    

    // misc
    double sensorFrameRate = 10.0 / imgStepWidth; // frames per second for Lidar and camera
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis;
    std::vector<cv::Scalar> colors = {cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), cv::Scalar(255, 0, 255),
                                        cv::Scalar(25, 255, 130), cv::Scalar(0, 255, 255), cv::Scalar(128, 0, 128), cv::Scalar(255, 215, 0), 
                                        cv::Scalar(255, 100, 100), cv::Scalar(10, 255, 10), cv::Scalar(20, 20, 255), cv::Scalar(55, 100, 155),
                                        cv::Scalar(155, 155, 100), cv::Scalar(100, 255, 255), cv::Scalar(128, 100, 128), cv::Scalar(120, 215, 20),
                                        cv::Scalar(55, 10, 100)};
    //map from boxID to track id 
    //std::unordered_map<int, int> box_id_to_track_id_map;
    //TTC_matrix[i] is a vector of TTCS for object with track id i
    std::unordered_map<int, std::vector<std::pair<int, float>>> track_id_to_TTCs_map;
    std::vector<std::pair<int, float>> TTCLidarEgoVehicle;
    int max_track_id = -1;
    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex+=imgStepWidth)
    {
        bVis = false;
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file 
        cv::Mat img = cv::imread(imgFullFilename);

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = img;
        dataBuffer.push_back(frame);

        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;


        /* DETECT & CLASSIFY OBJECTS */

        float confThreshold = 0.2;
        float nmsThreshold = 0.4;        
        detectObjects((dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->boundingBoxes, confThreshold, nmsThreshold,
                      yoloBasePath, yoloClassesFile, yoloModelConfiguration, yoloModelWeights, bVis);
        
        cout << "#2 : DETECT & CLASSIFY OBJECTS done" << endl;


        /* CROP LIDAR POINTS */

        // load 3D Lidar points from file
        string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
        std::vector<LidarPoint> lidarPoints;
        loadLidarFromFile(lidarPoints, lidarFullFilename);

        // remove Lidar points based on distance properties
        float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 20.0, maxY = 2.0, minR = 0.1; // focus on ego lane
        //cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);
    
        (dataBuffer.end() - 1)->lidarPoints = lidarPoints;

        cout << "#3 : CROP LIDAR POINTS done" << endl;


        /* CLUSTER LIDAR POINT CLOUD */
        // associate Lidar points with camera-based ROI
        float shrinkFactor = 0.10; // shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
        clusterLidarWithROI((dataBuffer.end()-1)->boundingBoxes, (dataBuffer.end() - 1)->lidarPoints, shrinkFactor, P_rect_00, R_rect_00, RT);
        //cluster point cloud to get 3D bounding boxes
        int max_iterations_ransac = 100;
        float distance_tolerance_ransac = 0.5;
        float cluster_tolerance = 1.0f;
        int min_cluster_size = 30;
        bool lidar_points_cropped = false;
        RemoveGroundPlane(*(dataBuffer.end()-1), max_iterations_ransac, distance_tolerance_ransac, lidar_points_cropped);
        // Visualize 3D objects
        bVis = false;
        if(bVis)
        {
            show3DObjects((dataBuffer.end()-1)->boundingBoxes, cv::Size(4.0, 20.0), cv::Size(2000, 2000), true);
        }
        bVis = false;

        cout << "#4 : CLUSTER LIDAR POINT CLOUD done" << endl;
        
        
        // REMOVE THIS LINE BEFORE PROCEEDING WITH THE FINAL PROJECT
        //continue; // skips directly to the next image without processing what comes beneath

        /* DETECT IMAGE KEYPOINTS */

        // convert current image to grayscale
        cv::Mat imgGray;
        cv::cvtColor((dataBuffer.end()-1)->cameraImg, imgGray, cv::COLOR_BGR2GRAY);

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        
        detKeypoints(keypoints, imgGray, false, detectorType);

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;

        cout << "#5 : DETECT KEYPOINTS done" << endl;


        /* EXTRACT KEYPOINT DESCRIPTORS */

        cv::Mat descriptors;
        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        cout << "#6 : EXTRACT DESCRIPTORS done" << endl;

        //for the first frame associate bounding box with the keypoints it contains
        if (dataBuffer.size() == 1)
            clusterKptMatchesWithROI(*(dataBuffer.end() - 1));                    

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
            string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorType, matcherType, selectorType);

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            cout << "#7 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            
            /* TRACK 3D OBJECT BOUNDING BOXES */

            // STUDENT ASSIGNMENT
            // TASK FP.1 -> match list of 3D objects (vector<BoundingBox>) between current and previous frame (implement ->matchBoundingBoxes)
            //Associate bounding box with the keypoints and keypoint matches it contains
            clusterKptMatchesWithROI(*(dataBuffer.end() - 1));
            map<int, int> bbBestMatches;
            float iouTolerance = 0.2f;
            matchBoundingBoxes(bbBestMatches, *(dataBuffer.end()-2), *(dataBuffer.end()-1), iouTolerance, max_track_id); // associate bounding boxes between current and previous frame using keypoint matches
            // EOF STUDENT ASSIGNMENT
            std::cout << "number of matches: " << bbBestMatches.size() << std::endl;
            // store matches in current data frame
            (dataBuffer.end()-1)->bbMatches = bbBestMatches;
            cout << "#8 : TRACK 3D OBJECT BOUNDING BOXES done" << endl;

            /* COMPUTE TTC ON OBJECT IN FRONT */
            cv::Mat visImg;
            // loop over all BB match pairs
            for (auto it1 = (dataBuffer.end() - 1)->bbMatches.begin(); it1 != (dataBuffer.end() - 1)->bbMatches.end(); ++it1)
            {
                if(it1==(dataBuffer.end() - 1)->bbMatches.begin()){
                    visImg = (dataBuffer.end() - 1)->cameraImg.clone();
                } 
                // find bounding boxes associates with current match
                BoundingBox *prevBB, *currBB;
                for (auto it2 = (dataBuffer.end() - 1)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 1)->boundingBoxes.end(); ++it2)
                {
                    if (it1->second == it2->boxID) // check wether current match partner corresponds to this BB
                    {
                        currBB = &(*it2);
                    }
                }

                for (auto it2 = (dataBuffer.end() - 2)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 2)->boundingBoxes.end(); ++it2)
                {
                    if (it1->first == it2->boxID) // check wether current match partner corresponds to this BB
                    {
                        prevBB = &(*it2);
                    }
                }

                // compute TTC for current match
                if( currBB->lidarPoints.size()>0 && prevBB->lidarPoints.size()>0 ) // only compute TTC if we have Lidar points
                {   
                    // STUDENT ASSIGNMENT
                    // TASK FP.2 -> compute time-to-collision based on Lidar data (implement -> computeTTCLidar)
                    double ttcCamera; 
                    computeTTC(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttcCamera);
                    // EOF STUDENT ASSIGNMENT

                    // STUDENT ASSIGNMENT
                    // TASK FP.3 -> assign enclosed keypoint matches to bounding box (implement -> clusterKptMatchesWithROI)
                    // TASK FP.4 -> compute time-to-collision based on camera (implement -> computeTTCCamera)
                    //showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00, R_rect_00, RT, &visImg);
                    track_id_to_TTCs_map[currBB->trackID].push_back({imgIndex+1, ttcCamera});
                    cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y), cv::Point(currBB->roi.x + currBB->roi.width, currBB->roi.y + currBB->roi.height),\
                        colors[currBB->trackID % colors.size()], 2);
                    char str_id[5];
                    sprintf(str_id, "track id %d", currBB->trackID);
                    putText(visImg, str_id, cv::Point(currBB->roi.x, currBB->roi.y-10), \
                    cv::FONT_HERSHEY_SIMPLEX, 0.9, colors[currBB->trackID], 2);
                    if(currBB->vehicle_in_front)
                    {
                        double ttcLidar;
                        computeTTC((dataBuffer.end() - 2)->lidarPointsVehicleInFront, (dataBuffer.end() - 1)->lidarPointsVehicleInFront, sensorFrameRate, ttcLidar);
                        TTCLidarEgoVehicle.push_back({imgIndex, ttcLidar});
                        char str[200];
                        sprintf(str, "TTC Lidar : %f s, TTC Camera : %f s", ttcLidar, ttcCamera);
                        putText(visImg, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0,0,255));
                    }
                } // eof TTC computation
            } // eof loop over all BB matches    
            if(visImg.size[0]>0 && visImg.size[1]>0)
            {
                string windowName = "Final Results : TTC";
                cv::namedWindow(windowName, 4);
                cv::imshow(windowName, visImg);
                cv::waitKey(1);
            }        
        }

    } // eof loop over all images

    //save TTCs
    for(auto TTCs: track_id_to_TTCs_map)
    {
        //std:cout << "track_id: " << TTCs.first << std::endl;
        for(auto frame_ttc_pair: TTCs.second) TTCCamerafile << detectorType << ", " << descriptorType << ", " \
        << TTCs.first << ", " << frame_ttc_pair.first << ", " << frame_ttc_pair.second << "\n" ;
        std::cout << std::endl;
    }
    for(auto frame_ttc_pair: TTCLidarEgoVehicle) TTCLidarfile << detectorType << ", " << descriptorType << ", " \
        << frame_ttc_pair.first << ", " << frame_ttc_pair.second << "\n" ;
    return;
}

int main(int argc, const char *argv[])
{
    ofstream TTCLidarfile, TTCCamerafile;
    TTCLidarfile.open("../TTCLidar.csv");
    TTCCamerafile.open("../TTCCamera.csv");
    if(TTCLidarfile.is_open() && TTCCamerafile.is_open())
    {
        TTCLidarfile << "Detector, Descriptor, frame, TTCLidar for vehicle in front\n";
        TTCCamerafile << "Detector, Descriptor, frame, track_id, TTCCamera\n";
        vector<string> detectorTypes {"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"};
        vector<string> descriptorTypes {"BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"};
        for(string detectorType: detectorTypes){
            for(string descriptorType: descriptorTypes) {
                //remove incompatible combinations
                if (descriptorType == "BRIEF" && detectorType == "SIFT") continue;
                if (descriptorType == "AKAZE" && detectorType != "AKAZE") continue;
                cout << "detector: " << detectorType << " " << "descriptor: " << descriptorType << endl;
                run(detectorType, descriptorType, TTCCamerafile, TTCLidarfile);
            }
        }
        TTCLidarfile.close();
        TTCCamerafile.close();
    }
    else
    {
        if(!TTCLidarfile.is_open()) cout << "Unable to open TTCLidar.csv\n";
        if(!TTCCamerafile.is_open()) cout << "Unable to open TTCCamera.csv\n";
    }
    
    return 0;
}