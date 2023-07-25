///////////////////////////////////////////////////////////////////////////////
// Utils.h: Header file containing utlitiy functions for Detection and Tracking
//
// You can change BBox drawing settings in Constants
//
// by Jeric,2023
// 

#ifndef UTILS_H
#define UTILS_H

#include <set>
#include <opencv2/opencv.hpp>

using namespace std;

/********Constants********/
// For Drawing BBoxes
const vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};
const float line_width = 3.0;
const float font_scale = line_width / 3.0f;
const float line_thickness = max(line_width - 1.0f, 1.0f);

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

/********Functions********/
cv::Mat formatYOLOv5(const cv::Mat& input_image);
cv::Rect scaleBBox(const cv::Rect& input_bbox, const double scale_factor);
double GetIOU(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt);
void drawBBox(cv::Mat &frame, vector<Detection>& detector_output, const vector<string>& class_list);
void drawBBox(cv::Mat &frame, vector<Track> &multi_tracker, const vector<string> &class_list, const int MIN_HITS);

#endif
