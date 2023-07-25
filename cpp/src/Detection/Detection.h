///////////////////////////////////////////////////////////////////////////////
// Detection.h: Header file for Detection
//
// You can change thresholds and network input details below in Constants
//
// by Jeric,2023
// 

#ifndef DETECTION_H
#define DETECTION_H

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

#include "../common/Utils.h"
#include "../common/Init_NN.h"

using namespace std;

class ObjectDetector
{
public:
    ObjectDetector();
    void inputPaths(const string& directory_name, const string& source_path);
    void initDetector();
    int runObjectDetection();

    /********Data Storage********/
    // For NN
    vector<string> class_list;
    
    // For Detections
    vector<Detection> detector_output;

    /********Counters********/
    int total_frames = 0;

    /********Methods********/    
    // For Detection
    void detect(cv::Mat &image, cv::dnn::Net &net, vector<Detection> &output, const vector<string> &className);


private:
    /********Control Flags********/
    const int STATUS_FLAG = 1;
    const int DEBUG_FLAG = 0;

    /********Constants********/
    // For Detection
    const float INPUT_WIDTH = 640.0;
    const float INPUT_HEIGHT = 640.0;
    const float CLASS_CONF_THRES = 0.25;
    const float NMS_THRES= 0.4;
    const float DETECT_CONF_THRES= 0.4;

    /********User Input********/
    string path_video_input;
    string path_video_output;
    string path_class_input;
    string path_net_input;

    /********Video Input Details********/
    double input_fps;
    int fw;
    int fh;

    /********Video and DNN Init********/
    cv::VideoCapture cap;
    cv::VideoWriter out;
    cv::dnn::Net net;
};

#endif
