
#ifndef camFusion_hpp
#define camFusion_hpp

#include <stdio.h>
#include <vector>
#include <unordered_set>
#include <opencv2/core.hpp>
#include "dataStructures.h"

struct CvPoint2fHash;
float IOU(const std::unordered_set<cv::Point2f, CvPoint2fHash>& first, const std::unordered_set<cv::Point2f, CvPoint2fHash>& second);
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT);
// void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches);
void clusterKptMatchesWithROI(DataFrame& currFrameBuffer);
void matchBoundingBoxes(std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame, float iouTolerance=0.1);

void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait=true);

void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg=nullptr);
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC);                  
#endif /* camFusion_hpp */
