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

using namespace std;

class ObjectDetector
{
public:
    ObjectDetector();
    void inputPaths(const string& directory_name, const string& source_path);
    void initDetector();
    int runObjectDetection();

private:
    /********Control Flags********/
    const int STATUS_FLAG = 1;
    const int DEBUG_FLAG = 1;

    /********User Input********/
    string path_video_input;
    string path_video_output;
    string path_class_input;
    string path_net_input;

    /********Constants********/
    // For Drawing BBoxes
    const vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};

    // For Detection
    const float INPUT_WIDTH = 640.0;
    const float INPUT_HEIGHT = 640.0;
    const float CLASS_CONF_THRES = 0.25;
    const float NMS_THRES= 0.4;
    const float DETECT_CONF_THRES= 0.4;

    /********Data Structs********/
    struct Detection
    {
        int class_id;
        float confidence;
        cv::Rect bbox;
    };

    /********Data Storage********/
    // For Detections
    vector<Detection> detector_output;
    vector<string> class_list;

    /********Video Input Details********/
    double input_fps;
    int fw;
    int fh;
    
    /********BBox Drawing Details********/
    float line_width = 3.0;
    float font_scale = line_width / 3.0f;
    float line_thickness = max(line_width - 1.0f, 1.0f);

    /********Counters********/
    int total_frames = 0;

    /********Video and DNN Init********/
    cv::VideoCapture cap;
    cv::VideoWriter out;
    cv::dnn::Net net;

    /********Methods********/
    // For Init of ObjectDetector
    void loadClassList(vector<string> &class_list);
    void loadNet(cv::dnn::Net &net);
    void warmupNet(cv::dnn::Net& net);

    // For Detection
    void detect(cv::Mat &image, cv::dnn::Net &net, vector<Detection> &output, const vector<string> &className);
    
    // For Drawing BBoxes
    void drawBBox(cv::Mat &frame, vector<Detection> &output, const vector<string> &class_list);
};

#endif
