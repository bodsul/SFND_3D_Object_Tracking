
#ifndef camFusion_hpp
#define camFusion_hpp

#include <stdio.h>
#include <vector>
#include <unordered_set>
#include <opencv2/core.hpp>
#include "dataStructures.h"

struct CvPoint2fHash;
float IOU(const std::unordered_set<cv::Point2f, CvPoint2fHash>& first, const std::unordered_set<cv::Point2f, CvPoint2fHash>& second);
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT,\
float minZ = -1.5, float maxZ = -0.9, float minX = 2.0, float maxX = 20.0, float maxY = 2.0, float minR = 0.1);
void RemoveGroundPlane(DataFrame& currFrameBuffer, float maxIterationsRansac, double distanceThresholdRansac, bool lidarPointsCropped);
void clusterKptMatchesWithROI(DataFrame& currFrameBuffer);
void matchBoundingBoxes(std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame, float iouTolerance, int& max_track_id);
void match3DBoundingBoxes(std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame, float iouTolerance, int& max_track_id);
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait=true);
void computeTTC(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC);                  
#endif /* camFusion_hpp */
