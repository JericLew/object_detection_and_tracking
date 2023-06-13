#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>


// Add the implementation for format_yolov5, unwrap_detection, and drawBoundingBoxes functions
cv::Mat format_yolov5(const cv::Mat& input_image, cv::Mat& output_image) {
    cv::Mat resized_image;
    cv::resize(input_image, resized_image, cv::Size(640, 640));

    cv::Mat blob;
    cv::dnn::blobFromImage(resized_image, blob, 1.0/255.0, cv::Size(640, 640), cv::Scalar(0, 0, 0), true, false);

    output_image = resized_image.clone();

    return blob;
}

void unwrap_detection(const cv::Mat& input_image, const cv::Mat& output_data, float detect_conf_thres, float class_conf_thres, std::vector<cv::Rect>& bboxes, std::vector<float>& confidences, std::vector<int>& class_ids) {
    bboxes.clear();
    confidences.clear();
    class_ids.clear();

    int image_width = input_image.cols;
    int image_height = input_image.rows;
    float x_factor = static_cast<float>(image_width) / 640.0f;
    float y_factor = static_cast<float>(image_height) / 640.0f;

    cv::Mat valid_indices;
    cv::compare(output_data.col(4), detect_conf_thres, valid_indices, cv::CMP_GE);

    cv::Mat valid_classes;
    cv::Mat argmax_output = output_data.colRange(5, output_data.cols);
    cv::reduce(argmax_output, valid_classes, 1, cv::REDUCE_ARGMAX);

    cv::Mat valid_scores = argmax_output(cv::Range(0, argmax_output.rows), valid_classes);

    cv::Mat valid_mask = valid_scores > class_conf_thres;

    std::vector<int> valid_indices_vec;
    for (int i = 0; i < valid_mask.rows; ++i) {
        if (valid_mask.at<uchar>(i)) {
            valid_indices_vec.push_back(i);
        }
    }

    cv::Mat valid_rows = output_data(cv::Range(0, output_data.rows), cv::Range(0, 4));
    cv::Mat valid_boxes = valid_rows(cv::Range(0, valid_indices_vec.size()), cv::Range(0, 4));

    valid_boxes.col(0) = (valid_boxes.col(0) - 0.5 * valid_boxes.col(2));
    valid_boxes.col(1) = (valid_boxes.col(1) - 0.5 * valid_boxes.col(3));
    valid_boxes.convertTo(valid_boxes, CV_32S);

    valid_boxes.col(0) = valid_boxes.col(0) * x_factor;
    valid_boxes.col(1) = valid_boxes.col(1) * y_factor;
    valid_boxes.col(2) = valid_boxes.col(2) * x_factor;
    valid_boxes.col(3) = valid_boxes.col(3) * y_factor;

    for (int i = 0; i < valid_indices_vec.size(); ++i) {
        cv::Rect bbox(valid_boxes.at<int>(i, 0), valid_boxes.at<int>(i, 1), valid_boxes.at<int>(i, 2), valid_boxes.at<int>(i, 3));
        bboxes.push_back(bbox);
        confidences.push_back(valid_scores.at<float>(valid_indices_vec[i]));
        class_ids.push_back(valid_classes.at<int>(valid_indices_vec[i]));
    }
}

void nms(const std::vector<cv::Rect>& in_boxes, const std::vector<float>& in_confidences, const std::vector<int>& in_class_ids, float detect_conf_thres, float nms_thres, std::vector<cv::Rect>& result_boxes, std::vector<float>& result_confidences, std::vector<int>& result_class_ids) {
    result_boxes.clear();
    result_confidences.clear();
    result_class_ids.clear();

    std::vector<int> indexes;
    cv::dnn::NMSBoxes(in_boxes, in_confidences, detect_conf_thres, nms_thres, indexes);

    for (int i : indexes) {
        result_boxes.push_back(in_boxes[i]);
        result_confidences.push_back(in_confidences[i]);
        result_class_ids.push_back(in_class_ids[i]);
    }
}

void drawBoundingBoxes(cv::Mat& image, const std::vector<cv::Rect>& bboxes, const std::vector<int>& class_ids, const std::vector<float>& confidences) {
    for (size_t i = 0; i < bboxes.size(); ++i) {
        cv::Rect bbox = bboxes[i];
        int class_id = class_ids[i];
        float confidence = confidences[i];

        // Draw the bounding box
        cv::rectangle(image, bbox, cv::Scalar(0, 255, 0), 2);

        // Construct the label text
        std::string label = cv::format("Class: %d, Confidence: %.2f", class_id, confidence);

        // Get the label size
        int baseline;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);

        // Draw the label background rectangle
        cv::rectangle(image, cv::Point(bbox.x, bbox.y - label_size.height - 10), cv::Point(bbox.x + label_size.width, bbox.y - 5), cv::Scalar(0, 255, 0), cv::FILLED);

        // Draw the label text
        cv::putText(image, label, cv::Point(bbox.x, bbox.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
    }
}