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
#include <opencv2/imgproc.hpp>

#include "Hungarian.h"
#include "Utils.h"

using namespace std;

class MOTCorrelationTracker
{
public:
    MOTCorrelationTracker();
    void inputPaths(const string& directory_name, const string& source_path, const string& tracketracker_namerName);
    void initTracker();
    int runObjectTracking();

private:
    /********User Input********/
    string path_video_input;
    string path_video_output;
    string path_class_input;
    string path_net_input;
    string tracker_name;

    /********Constants********/
    // For Drawing BBoxes
    const vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};
    
    // For Detection
    const float INPUT_WIDTH = 640.0;
    const float INPUT_HEIGHT = 640.0;
    const float CLASS_CONF_THRES = 0.25;
    const float NMS_THRES= 0.4;
    const float DETECT_CONF_THRES= 0.4;
    const double SCALE_FACTOR = 1.0 / 3.0;

    // For Association
    const int MAX_AGE = 1;
    const int MIN_HITS = 3;
    const float IOU_THRES = 0.3;

    // For Debug
    const int DEBUG_FLAG = 1;

    /********Data Structs********/
    struct Detection
    {
        int class_id;
        float confidence;
        cv::Rect bbox;
    };

    struct Track
    {
        cv::Ptr<cv::Tracker> tracker;
        int track_id;
        int class_id;
        float confidence;
        cv::Rect bbox;
        int num_hit;
        int num_miss;
    };

    /********Data Storage********/
    // For Detections
    vector<Detection> detector_output;
    vector<string> class_list;

    // For Tracking
    vector<Track> multi_tracker;

    // For Association
    vector<vector<double>> iou_matrix;
    vector<int> assignment;
    set<int> unmatched_detections;
    set<int> unmatched_tracks;
    set<int> all_detections;
    set<int> matched_detections;
    vector<cv::Point> matched_pairs;

    /********Video Input Details********/
    double input_fps;
    int fw;
    int fh;

    /********Counters********/
    int track_count = 0;
    int total_frames = 0;

    /********Video and DNN Init********/
    cv::VideoCapture cap;
    cv::VideoWriter out;
    cv::dnn::Net net;

    /********Methods********/
    // For Init of MOTCorrelationTracker
    void loadClassList(vector<string> &class_list);
    void loadNet(cv::dnn::Net &net);

    // For Detection
    void detect(cv::Mat &image, cv::dnn::Net &net, vector<Detection> &output, const vector<string> &className);
    
    // For Tracking
    void createTracker(cv::Mat &frame, Detection& detection);
    void getTrackersPred(cv::Mat &frame);

    // For Association
    void associate();
    void updateTrackers(cv::Mat &frame, vector<Detection>& detector_output);

    // For Drawing BBoxes
    void drawBBox(cv::Mat &frame, vector<Detection> &output, const vector<string> &class_list);
    void drawBBox(cv::Mat &frame, vector<Track> &multi_tracker, const vector<string> &class_list);
};

#endif
