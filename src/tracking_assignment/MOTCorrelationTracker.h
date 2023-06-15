#ifndef MOTCORRELATION_H
#define MOTCORRELATION_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <set>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/tracking/tracking_legacy.hpp>

#include "Hungarian.h"


using namespace std;

class MOTCorrelationTracker
{
public:
    MOTCorrelationTracker();
    void inputPaths(const string& directory_name, const string& source_path, const string& tracketracker_namerName);
    void initTracker();
    int runObjectTracking();

private:
    // path to files
    string path_video_input;
    string path_video_output;
    string path_class_input;
    string path_net_input;
    string tracker_name;

    // constants
    const vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};
    const float INPUT_WIDTH = 640.0;
    const float INPUT_HEIGHT = 640.0;
    const float SCORE_THRESHOLD = 0.2;
    const float NMS_THRESHOLD = 0.4;
    const float CONFIDENCE_THRESHOLD = 0.4;

    int max_age = 1;
    int min_hits = 3;
    double iouThreshold = 0.3;

    // structs
    struct Detection
    {
        int class_id;
        float confidence;
        cv::Rect box;
    };

    struct Track
    {
        cv::Ptr<cv::Tracker> tracker;
        int track_id;
        int class_id;
        float confidence;
        cv::Rect box;
        int num_hit;
        int num_miss;
    };

    // Data Storage
    vector<Track> multi_tracker;
    vector<Detection> detector_output;
    vector<string> class_list;

    vector<vector<double>> iouMatrix;
    vector<int> assignment;

    set<int> unmatchedDetections;
    set<int> unmatchedTracks;
    set<int> allItems;
    set<int> matchedItems;
    vector<cv::Point> matchedPairs;

    // video input details
    double input_fps;
    int fw;
    int fh;

    // variables
    int track_count = 0;
    int total_frames = 0;

    // init capture, writer and network
    cv::VideoCapture cap;
    cv::VideoWriter out;
    cv::dnn::Net net;

    // methods
    void load_class_list(vector<string> &class_list);
    void load_net(cv::dnn::Net &net);
    cv::Mat format_yolov5(const cv::Mat &source);
    void detect(cv::Mat &image, cv::dnn::Net &net, vector<Detection> &output, const vector<string> &className);
    
    void createTracker(cv::Mat &frame, Detection& detection);
    void getTrackersPred(cv::Mat &frame);
    void associate();
    void updateTrackers(cv::Mat &frame, vector<Detection>& detector_output);

    void drawBBox(cv::Mat &frame, vector<Detection> &output, const vector<string> &class_list);
    void drawBBox(cv::Mat &frame, vector<Track> &multi_tracker, const vector<string> &class_list);
};

#endif
