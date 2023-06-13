#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

class ObjectDetector
{
public:
    ObjectDetector();

    void runObjectDetection();
    void streamStart();

private:
    // constants
    float detect_conf_thres;
    float class_conf_thres;
    float nms_thres;

    // video input details
    double fps;
    int fw;
    int fh;

    // variables
    int frame_count;
    std::vector<cv::Rect> detect_bboxes;
    std::vector<float> detect_conf;
    std::vector<int> detect_class_ids;

    // methods
    void detect(cv::Mat &current_frame);
    void unwrap_detection(const cv::Mat &input_image, const cv::Mat &output_data, float detect_conf_thres, float class_conf_thres, std::vector<cv::Rect> &bboxes, std::vector<float> &confidences, std::vector<int> &class_ids);
    void nms(const std::vector<cv::Rect> &in_boxes, const std::vector<float> &in_confidences, const std::vector<int> &in_class_ids, float detect_conf_thres, float nms_thres, std::vector<cv::Rect> &result_boxes, std::vector<float> &result_confidences, std::vector<int> &result_class_ids);
    cv::Mat format_yolov5(const cv::Mat &input_image, cv::Mat &output_image);
    void drawBoundingBoxes(cv::Mat &image, const std::vector<cv::Rect> &bboxes, const std::vector<int> &class_ids, const std::vector<float> &confidences);

    // init cap, out and net
    cv::VideoCapture cap = cv::VideoCapture("/home/jeric/tracking_ws/video_input/video1.avi");
    cv::VideoWriter out = cv::VideoWriter("/home/jeric/tracking_ws/video_output_c++/detect_output.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30, cv::Size(640, 480));
    cv::dnn::Net net = cv::dnn::readNetFromONNX("/home/jeric/yolov5/yolov5s.onnx");
};

ObjectDetector::ObjectDetector()
    : detect_conf_thres(0.4f), class_conf_thres(0.25f), nms_thres(0.4f), frame_count(0)
{
    // enable CUDA
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    fps = cap.get(cv::CAP_PROP_FPS);
    fw = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    fh = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::cout << "FPS: " << fps << ", Width: " << fw << ", Height: " << fh << std::endl;
}

void ObjectDetector::streamStart()
{
    if (!cap.isOpened())
    {
        std::cout << "Failed to open video file!" << std::endl;
        return;
    }
    cv::Mat frame;
    while (1)
    {
        cap.read(frame);

        if (frame.empty())
        {
            std::cout << "Failed to capture frame!" << std::endl;
            break;
        }

        cv::imshow("Video Stream", frame); // Display the frame

        if (cv::waitKey(1) == 27) // Break the loop when 'Esc' key is pressed
            break;
    }

    cap.release();
    cv::destroyAllWindows();
}

void ObjectDetector::runObjectDetection()
{
    cv::Mat frame;
    cv::Mat outputFrame;

    while (cap.read(frame))
    {
        // Detect objects in the current frame
        detect(frame);

        // Draw bounding boxes on the frame
        outputFrame = frame.clone();
        drawBoundingBoxes(outputFrame, detect_bboxes, detect_class_ids, detect_conf);

        // Display the output frame
        cv::imshow("Object Detection", outputFrame);
        cv::waitKey(1);

        // Write the output frame to the video writer
        out.write(outputFrame);
    }
}

struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
};

cv::Mat ObjectDetector::format_yolov5(const cv::Mat &source) {
  
    // put the image in a square big enough
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat resized = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(resized(cv::Rect(0, 0, col, row)));
    
    // resize to 640x640, normalize to [0,1[ and swap Red and Blue channels
    cv::Mat result;
    cv::dnn::blobFromImage(source, result, 1./255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
  
    return result;
}

void ObjectDetector::detect(cv::Mat &image, cv::dnn::Net &net, std::vector<Detection> &output, const std::vector<std::string> &className)
{
    cv::Mat blob;
    blob = format_yolov5(image);

    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    float *data = (float *)outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i)
    {

        float confidence = data[4];
        if (confidence >= CONFIDENCE_THRESHOLD)
        {

            float *classes_scores = data + 5;
            cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > SCORE_THRESHOLD)
            {

                confidences.push_back(confidence);

                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }

        data += 85;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    for (int i = 0; i < nms_result.size(); i++)
    {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }
}

int main()
{
    // // Create an instance of the ObjectDetector class
    ObjectDetector detector;

    detector.streamStart();

    // // Run the object detection
    // detector.runObjectDetection();

    return 0;
}
