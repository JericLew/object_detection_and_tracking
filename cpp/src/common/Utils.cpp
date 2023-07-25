///////////////////////////////////////////////////////////////////////////////
// Utils.cpp: Source file containing utlitiy functions for Detection and Tracking
//
// by Jeric,2023
// 

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

void drawBBox(cv::Mat &frame, vector<Detection>& detector_output, const vector<string>& class_list)
{
    cout << "Drawing BBox for detections...\n";
    int detections = detector_output.size();
    for (int i = 0; i < detections; ++i)
    {
        Detection& detection = detector_output[i];
        cv::Rect& bbox = detection.bbox;
        int class_id = detection.class_id;
        float confidence = detection.confidence;
        string class_name = class_list[class_id];
        cv::Scalar color = colors[class_id % colors.size()];
        std::string text_label = class_name + " " + std::to_string(static_cast<int>(confidence * 100)) + "%";

        cv::rectangle(frame, bbox, color, line_width, cv::LINE_AA);
        cv::rectangle(frame, cv::Point(bbox.x, bbox.y - 30), cv::Point(bbox.x + bbox.width, bbox.y), color, cv::FILLED);
        cv::putText(frame, text_label, cv::Point(bbox.x, bbox.y - 5), 0, font_scale, cv::Scalar(), line_thickness, cv::LINE_AA);
    }
}

void drawBBox(cv::Mat &frame, vector<Track> &multi_tracker, const vector<string> &class_list, const int MIN_HITS)
{
    cout << "Drawing BBox from trackers...\n";
    for (int i = 0; i < multi_tracker.size(); i++)
    {
        Track& track = multi_tracker[i];
        if (track.num_hit < MIN_HITS)
        {
            continue;
        }

        cv::Rect& bbox = track.bbox;
        int class_id = track.class_id;
        int track_id = track.track_id;
        float confidence = track.confidence;
        string class_name = class_list[class_id];
        cv::Scalar color = colors[class_id % colors.size()];
        std::string text_label = std::to_string(track_id)+ ": "  + class_name;
        cv::rectangle(frame, bbox, color, line_width, cv::LINE_AA);
        cv::rectangle(frame, cv::Point(bbox.x, bbox.y - 30), cv::Point(bbox.x + bbox.width, bbox.y), color, cv::FILLED);
        cv::putText(frame, text_label, cv::Point(bbox.x, bbox.y - 5), 0, font_scale, cv::Scalar(), line_thickness, cv::LINE_AA);
    }
}