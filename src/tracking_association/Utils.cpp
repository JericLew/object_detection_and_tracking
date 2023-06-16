#include "Utils.h"

using namespace std;

cv::Mat formatYOLOv5(const cv::Mat& input_image)
{
    int col = input_image.cols;
    int row = input_image.rows;
    int _max = MAX(col, row);
    cv::Mat formatted_image = cv::Mat::zeros(_max, _max, CV_8UC3);
    input_image.copyTo(formatted_image(cv::Rect(0, 0, col, row)));
    return formatted_image;
}

cv::Rect scaleBBox(const cv::Rect& input_bbox, const double scale_factor)
{
    cv::Rect scaled_bbox;
    scaled_bbox.x = input_bbox.x * scale_factor;
    scaled_bbox.y = input_bbox.y * scale_factor;
    scaled_bbox.width = input_bbox.width * scale_factor;
    scaled_bbox.height = input_bbox.height * scale_factor;    

    return scaled_bbox;
}

// Computes IOU between two bounding bboxes
double GetIOU(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt)
{
	float in = (bb_test & bb_gt).area();
	float un = bb_test.area() + bb_gt.area() - in;

	if (un < DBL_EPSILON)
		return 0;

	return (double)(in / un);
}