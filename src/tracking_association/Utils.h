#ifndef UTILS_H
#define UTILS_H

#include <set>
#include <opencv2/opencv.hpp>

using namespace std;

cv::Mat formatYOLOv5(const cv::Mat& input_image);
cv::Rect scaleBBox(const cv::Rect& input_bbox, const double scale_factor);
double GetIOU(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt);

#endif
