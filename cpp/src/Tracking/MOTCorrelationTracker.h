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

#include "../common/Hungarian.h"
#include "../common/Utils.h"
#include "../common/Init_NN.h"
#include "../Detection/Detection.h"

using namespace std;

class MOTCorrelationTracker
{
public:
    MOTCorrelationTracker();
    void inputPaths(const string& directory_name, const string& source_path, const string& tracker_name);
    void initTracker();
    int runObjectTracking();

    /********Data Storage********/
    // For NN
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

    /********Counters********/
    int track_count = 0;

    /********Methods********/
    // For Tracking
    void createTrack(cv::Mat& frame, Detection& detection);
    void createTracker(cv::Mat& shrunk_frame, Detection& detection, cv::Ptr<cv::Tracker>& new_tracker);
    void getTrackersPred(cv::Mat& frame);

    // For Association
    void associate(const vector<Track> &multi_tracker, const vector<Detection> &detector_output);
    void updateTracks(cv::Mat& frame, vector<Detection>& detector_output);

private:
    /********Control Flags********/
    const int STATUS_FLAG = 1;
    // const int TIMER_FLAG = 1; // TODO TIMER TO TEST PERF
    const int DEBUG_FLAG = 0;

    /********User Input*****
     * ***/
    string path_video_input;
    string path_video_output;
    string path_class_input;
    string path_net_input;
    string tracker_name;

    /********Video Input Details********/
    double input_fps;
    int fw;
    int fh;

    /********Video and DNN Init********/
    cv::VideoCapture cap;
    cv::VideoWriter out;
    cv::dnn::Net net;

    /********Constants********/
    const double SCALE_FACTOR = 1.0 / 3.0;
    const float INPUT_WIDTH = 640.0;
    const float INPUT_HEIGHT = 640.0;

    // For Association
    const int MAX_AGE = 1;
    const int MIN_HITS = 3;
    const float IOU_THRES = 0.3; // IoU Thres to reject assocations
    const float REFRESH_IOU_THRES = 0.80; // IoU Thres to replace old tracker with new for current track
};

#endif
