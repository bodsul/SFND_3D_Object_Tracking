#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
        descSource.convertTo(descSource, CV_8U);
        descRef.convertTo(descRef, CV_8U);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        matcher = cv::FlannBasedMatcher::create();
        descSource.convertTo(descSource, CV_32F);
        descRef.convertTo(descRef, CV_32F);
    }
    else
    {
        std::cout << "Unimplemented matcher: " << matcherType;
        exit(1);
    }
    

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        std::vector<std::vector<cv::DMatch>> tmp_matches;
        matcher->knnMatch(descSource, descRef, tmp_matches, 2);
        for (auto best_two_matches : tmp_matches)
        {
            if(best_two_matches[0].distance < 0.8*best_two_matches[1].distance) matches.push_back(best_two_matches[0]);
        }
    }
    else
    {
        std::cout << "Unimplemented selector: " << selectorType;
        exit(1);
    }
    std::cout << "Number of matched keypoints: " << matches.size() << std::endl;
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SIFT::create();
    }
    else
    {
        std::cout << "Unmplimented descriptor: " << descriptorType << std::endl;
    }
    
    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

void visualize_keypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string windowName)
{
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::namedWindow(windowName, 6);
    imshow(windowName, visImage);
    cv::waitKey(25);

}

void detKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis, std::string detectorType)
{
    double t = (double)cv::getTickCount();
    if (detectorType.compare("SHITOMASI") == 0)
    {
        detKeypointsShiTomasi(keypoints, img);
    }
    else if (detectorType.compare("HARRIS") == 0)
    {
        detKeypointsHarris(keypoints, img);
    }
    else if (detectorType.compare("ORB") == 0)
    {
        detKeypointsORB(keypoints, img);
    }
    else if (detectorType.compare("AKAZE") == 0)
    {
        detKeypointsAKAZE(keypoints, img);
    }
    else if (detectorType.compare("FAST") == 0)
    {
        detKeypointsAKAZE(keypoints, img);
    }
    else if (detectorType.compare("SURF") == 0)
    {
        detKeypointsSurf(keypoints, img);
    }
    else if (detectorType.compare("BRISK") == 0)
    {
        detKeypointsBRISK(keypoints, img);
    }
    else if (detectorType.compare("SIFT") == 0)
    {
        detKeypointsSIFT(keypoints, img);
    }
    else
    {
        cout << "Unimplemented detector method: " << detectorType << endl;
        exit(1);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << detectorType << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    // visualize results
    if (bVis)
    {
        visualize_keypoints(keypoints, img, detectorType + " Detector Results");
    }
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
}

// Detect keypoints in image using the Harris corner detector
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, true, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }

}

void detKeypointsFast(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img)
{
    cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
    detector->detect(img, keypoints);  
}

void detKeypointsBRISK(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img)
{
    cv::Ptr<cv::BRISK> detector = cv::BRISK::create();
    detector->detect(img, keypoints);
}

void detKeypointsSurf(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img)
{
    int minHessian = 400;
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create( minHessian );
    detector->detect( img, keypoints );    
}

void detKeypointsSIFT(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img)
{
    cv::Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create();
    detector->detect( img, keypoints );     
}

void detKeypointsORB(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img)
{
    cv::Ptr<cv::ORB> detector = cv::ORB::create();
    detector->detect( img, keypoints );     
}

void detKeypointsAKAZE(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img)
{
    cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();
    detector->detect( img, keypoints );      
}