
#include <iostream>
#include <algorithm>
#include <numeric>
#include <random>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"
#include "lidarData.hpp"
using namespace std;

//function object for hashing cv::Point2f types
struct CvPoint2fHash
{
	std::size_t operator() (const cv::Point2f &pt) const
	{
		return std::hash<float>()(pt.x) ^ std::hash<float>()(pt.y);
	}
};

struct 
{
    bool operator() (const std::pair<std::pair<int, int>, float>& item, const std::pair<std::pair<int, int>, float>& other) const
    {
        return item.second > other.second;
    }
}CustomGreaterThan;

std::unordered_set<int> pickSet(int N, int k, std::mt19937& gen)
{
    std::unordered_set<int> elems;
    for (int r = N - k; r < N; ++r) {
        int v = std::uniform_int_distribution<>(1, r)(gen);

        // there are two cases.
        // v is not in candidates ==> add it
        // v is in candidates ==> well, r is definitely not, because
        // this is the first iteration in the loop that we could've
        // picked something that big.

        if (!elems.insert(v).second) {
            elems.insert(r);
        }   
    }
    return elems;
}

std::vector<LidarPoint> RansacPlane(std::vector<LidarPoint>& cloud, int maxIterations, float distanceTol)
{
	std::vector<LidarPoint> inliersResult;
	int max_inliers = 0;
	srand(time(NULL));
	
	// TODO: Fill in this function

	// For max iterations 
	// Randomly sample subset and fit line

	// Measure distance between every point and fitted line
	// If distance is smaller than threshold count it as inlier

	// Return indicies of inliers from fitted line with most inliers
	std::vector<int> sample(3);
	std::vector<LidarPoint> pts(3);

	for(int j = 0; j < maxIterations; ++ j)
	{
		std::vector<LidarPoint> currResult;
		std::random_device rd;
		std::mt19937 gen(rd());
		std::unordered_set<int> sample = pickSet(cloud.size(), 3, gen);
		int l = 0;
		for(int k: sample){
			pts[l] = cloud[k];
			l++;
		}
		float A = (pts[1].y - pts[0].y)*(pts[2].z-pts[0].z) - (pts[2].y-pts[0].y)*(pts[1].z-pts[0].z);
		float B = (pts[2].x - pts[0].x)*(pts[1].z-pts[0].z) - (pts[1].x-pts[0].x)*(pts[2].z-pts[0].z);
		float C = (pts[1].x - pts[0].x)*(pts[2].y-pts[0].y) - (pts[2].x-pts[0].x)*(pts[1].y-pts[0].y);
		if(A==0 && B==0 && C==0) continue;
		float D = -(A*pts[0].x + B*pts[0].y + C*pts[0].z);
		float distance = std::abs(A*pts[1].x + B*pts[1].y + C*pts[1].z + D)/std::sqrt(A*A + B*B + C*C);
		for(int k = 0; k < cloud.size(); ++k){
			LidarPoint pt = cloud[k];
			float distance = std::abs(A*pt.x + B*pt.y + C*pt.z + D)/std::sqrt(A*A + B*B + C*C);
			if(distance < distanceTol) currResult.push_back(pt);
		}
		if(currResult.size() > max_inliers)
		{
			max_inliers = currResult.size();
			inliersResult = currResult;
		}
	}
    std::cout << "size " << cloud.size() << " inliers size " << inliersResult.size() << std::endl;
	return inliersResult;
}

void RemoveGroundPlane(DataFrame& currFrameBuffer, float maxIterationsRansac, double distanceThresholdRansac, bool lidarPointsCropped)
{
    std::vector<LidarPoint> inliers;
    if(!lidarPointsCropped)
    {
        // remove Lidar points based on distance properties
        float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 20.0, maxY = 2.0, minR = 0.1; // focus on ego lane
        std::vector<LidarPoint> cropped_lidarPoints=currFrameBuffer.lidarPoints;
        cropLidarPoints(cropped_lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);
        inliers = RansacPlane(cropped_lidarPoints, maxIterationsRansac, distanceThresholdRansac);
    }
    else
    {
        inliers = RansacPlane(currFrameBuffer.lidarPoints, maxIterationsRansac, distanceThresholdRansac);
    }
   
    for(auto pt: inliers){
        currFrameBuffer.lidarPointsVehicleInFront.push_back(pt);
    }
}

// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT, 
float minZ, float maxZ, float minX, float maxX, float maxY, float minR)\
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            if( !enclosingBoxes[0]->vehicle_in_front && (*it1).x>=minX && (*it1).x<=maxX && (*it1).z>=minZ && (*it1).z<=maxZ && (*it1).z<=0.0 && abs((*it1).y)<=maxY && (*it1).r>=minR )
            {
                enclosingBoxes[0]->vehicle_in_front = true;
            }
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
// void clusterKptMatchesWithROI(std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
void clusterKptMatchesWithROI(DataFrame& currFrame)
{
    // ...
    for(BoundingBox& box: currFrame.boundingBoxes)
    {
        for(cv::KeyPoint kpt: currFrame.keypoints)
        {
            if(box.roi.contains(kpt.pt)) box.keypoints.push_back(kpt);
        }

        for(cv::DMatch match: currFrame.kptMatches)
        {
            if(box.roi.contains(currFrame.keypoints[match.trainIdx].pt)) box.kptMatches.push_back(match);
        }
        // std::cout << "n_keypoints: " << box.keypoints.size() << std::endl;
        // std::cout << "n_matches: " << box.kptMatches.size() << std::endl;
    }
}

void computeTTC(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // ...
    double deltaT=1/frameRate;
    double currAvgX=0, currAvgY=0, currAvgZ=0;
    double prevAvgX=0, prevAvgY=0, prevAvgZ=0;
    double avgVelX, avgVelY, avgVelZ;
    for(auto pt: lidarPointsCurr)
    {
        currAvgX+=pt.x;
        currAvgY+=pt.y;
        currAvgZ+=pt.z;
    }
    currAvgX/=lidarPointsCurr.size();
    currAvgY/=lidarPointsCurr.size();
    currAvgZ/=lidarPointsCurr.size();

    for(auto pt: lidarPointsPrev)
    {
        prevAvgX+=pt.x;
        prevAvgY+=pt.y;
        prevAvgZ+=pt.z;
    }
    prevAvgX/=lidarPointsPrev.size();
    prevAvgY/=lidarPointsPrev.size();
    prevAvgZ/=lidarPointsPrev.size();
    avgVelX = (currAvgX-prevAvgX)/deltaT;
    avgVelY = (currAvgY-prevAvgY)/deltaT;
    avgVelZ = (currAvgZ-prevAvgZ)/deltaT; // this should be small since Z points up
    double avgSpeed = pow(avgVelX*avgVelX + avgVelY*avgVelY + avgVelZ*avgVelZ, 0.5);
    double distance = pow(currAvgX*currAvgX + currAvgY*currAvgY + currAvgY*currAvgY, 0.5);
    TTC = distance/avgSpeed;
}

float IOU(const std::unordered_set<cv::Point2f, CvPoint2fHash>& first, const std::unordered_set<cv::Point2f, CvPoint2fHash>& second)
{
    if(first.size()==0 || second.size()==0) return -1;
    uint intersection_size = 0;
    for(auto entry: first)
    {
        if(std::find(second.begin(), second.end(), entry)!=second.end()) intersection_size++;
    }
    return (float)intersection_size/(first.size() + second.size()-intersection_size);
}

void matchBoundingBoxes(std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame, float iouTolerance, int& max_track_id)
{
    //std::cout << "max_track_id: " << max_track_id << std::endl;
    std::map<int, BoundingBox*> curr_box_id_to_box_ptr_map, prev_box_id_to_box_ptr_map;
    std::vector<std::pair<std::pair<int, int>, float>> iou_for_all_box_pairs;
    bool first_loop_over_prev_bboxes = true;
    for (BoundingBox& bbox: currFrame.boundingBoxes)
    {
        curr_box_id_to_box_ptr_map.insert({bbox.boxID, &bbox});
        //get indices of predicted keypoints matched in prevframe based on keypoint match assigned to bbox
        std::unordered_set<int> predictedKptIdxs; 
        for(cv::DMatch match: bbox.kptMatches)
        {
            predictedKptIdxs.insert(match.queryIdx);
        }
        //get predicted keypoints matched in prevframe based on keypoint match assigned to bbox
        //use keypoint.pt to simplify hash function for unordered_set
        std::unordered_set<cv::Point2f, CvPoint2fHash> predictedKpts;

        for(int idx: predictedKptIdxs){
            predictedKpts.insert(prevFrame.keypoints[idx].pt);
        }

        //for each bounding box in prevframe compute iou of predicted keypoints to keypoints of bounding box
        for (BoundingBox& mbbox: prevFrame.boundingBoxes)
        {
            if (first_loop_over_prev_bboxes){
                prev_box_id_to_box_ptr_map.insert({mbbox.boxID, &mbbox});
            }
            std:unordered_set<cv::Point2f, CvPoint2fHash> kptsToMatch;
            for(auto kpt: mbbox.keypoints)
            {
                kptsToMatch.insert(kpt.pt);
            }
            float iou = IOU(predictedKpts, kptsToMatch);
            auto potential_matched_pair = std::pair<int, int>(mbbox.boxID, bbox.boxID);
            auto res = std::pair<std::pair<int, int>, float>(potential_matched_pair, iou);
            iou_for_all_box_pairs.push_back(res);
        }
        first_loop_over_prev_bboxes = false;
    }
    //sort oll pair of bounding box potential matches by IOU
    std::sort(iou_for_all_box_pairs.begin(), iou_for_all_box_pairs.end(), CustomGreaterThan);
    //associate bounding boxes
    std::unordered_set<int> matched_current_boxes, matched_previous_boxes;
    for(auto box_pair_iou: iou_for_all_box_pairs)
    {
        //stop assignment if iouTolerance is hit
        if(box_pair_iou.second < iouTolerance) break;
        if(std::find(matched_previous_boxes.begin(), matched_previous_boxes.end(),\
         box_pair_iou.first.first) == matched_previous_boxes.end() &&
         std::find(matched_current_boxes.begin(), matched_current_boxes.end(),\
         box_pair_iou.first.second) == matched_current_boxes.end())
         {
             matched_previous_boxes.insert(box_pair_iou.first.first);
             matched_current_boxes.insert(box_pair_iou.first.second);
             //std::cout << "matched_iou: " << box_pair_iou.second << std::endl;
             //std::cout << "prev_box_id: " << box_pair_iou.first.first <<  " curr_box_id: " <<  box_pair_iou.first.second << std::endl;
             bbBestMatches.insert(box_pair_iou.first);
            //assign track id based on match
            if(prev_box_id_to_box_ptr_map[box_pair_iou.first.first]->trackID != -1)
            {
                //std::cout << "track_id_old" << prev_box_id_to_box_ptr_map[box_pair_iou.first.first]->trackID << std::endl;
                curr_box_id_to_box_ptr_map[box_pair_iou.first.second]->trackID = prev_box_id_to_box_ptr_map[box_pair_iou.first.first]->trackID;
            }
            else
            {
                curr_box_id_to_box_ptr_map[box_pair_iou.first.second]->trackID = ++max_track_id;
                //std::cout << "track_id_new" << curr_box_id_to_box_ptr_map[box_pair_iou.first.second]->trackID << std::endl;
            }
         }
    }
}

void match3DBoundingBoxes(std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame, float iouTolerance)
{
    // std::vector<std::pair<std::pair<int, int>, float>> iou_for_all_box_pairs;
    // for (BoundingBox bbox: currFrame.boundingBoxes)
    // {
    //     //get indices of predicted keypoints matched in prevframe based on keypoint match assigned to bbox
    //     std::unordered_set<int> predictedKptIdxs; 
    //     for(cv::DMatch match: bbox.kptMatches)
    //     {
    //         predictedKptIdxs.insert(match.queryIdx);
    //     }
    //     //get predicted keypoints matched in prevframe based on keypoint match assigned to bbox
    //     //use keypoint.pt to simplify hash function for unordered_set
    //     std::unordered_set<cv::Point2f, CvPoint2fHash> predictedKpts;

    //     for(int idx: predictedKptIdxs){
    //         //predictedKpts.push_back(prevFrame.keypoints[idx]);
    //         predictedKpts.insert(prevFrame.keypoints[idx].pt);
    //     }

    //     //for each bounding box in prevframe compute iou of predicted keypoints to keypoints of bounding box
    //     for (BoundingBox mbbox: prevFrame.boundingBoxes)
    //     {
    //         std:unordered_set<cv::Point2f, CvPoint2fHash> kptsToMatch;
    //         for(auto kpt: mbbox.keypoints)
    //         {
    //             kptsToMatch.insert(kpt.pt);
    //         }
    //         float iou = IOU(predictedKpts, kptsToMatch);
    //         auto matched_pair = std::pair<int, int>(mbbox.boxID, bbox.boxID);
    //         auto res = std::pair<std::pair<int, int>, float>(matched_pair, iou);
    //         iou_for_all_box_pairs.push_back(res);
    //     }
    // }
    // //sort oll pair of bounding box potential matches by IOU
    // std::sort(iou_for_all_box_pairs.begin(), iou_for_all_box_pairs.end(), CustomGreaterThan);
    // //associate bounding boxes
    // std::unordered_set<int> matched_current_boxes, matched_previous_boxes;
    // for(auto box_pair_iou: iou_for_all_box_pairs)
    // {
    //     //stop assignment if iouTolerance is hit
    //     if(box_pair_iou.second < iouTolerance) break;
    //     if(std::find(matched_previous_boxes.begin(), matched_previous_boxes.end(),\
    //      box_pair_iou.first.first) == matched_previous_boxes.end() &&
    //      std::find(matched_current_boxes.begin(), matched_current_boxes.end(),\
    //      box_pair_iou.first.second) == matched_current_boxes.end())
    //      {
    //          matched_previous_boxes.insert(box_pair_iou.first.first);
    //          matched_current_boxes.insert(box_pair_iou.first.second);
    //         //  std::cout << "matched_iou: " << box_pair_iou.second << std::endl;
    //         //  std::cout << "prev_box_id: " << box_pair_iou.first.first <<  " curr_box_id: " <<  box_pair_iou.first.second << std::endl;
    //          bbBestMatches.insert(box_pair_iou.first);
    //      }
    // }
}