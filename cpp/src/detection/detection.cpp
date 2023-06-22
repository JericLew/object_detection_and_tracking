#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

class ObjectDetector
{
public:
    ObjectDetector(const std::string& directoryName, const std::string& sourcePath);

    int runObjectDetection();

private:
    // program input parameters
    std::string directoryName_;
    std::string sourcePath_;

    // path to files
    std::string path_video_input;
    std::string path_video_output;
    std::string path_class_input;
    std::string path_net_input;
    // constants
    const std::vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};

    const float INPUT_WIDTH = 640.0;
    const float INPUT_HEIGHT = 640.0;
    const float SCORE_THRESHOLD = 0.2;
    const float NMS_THRESHOLD = 0.45;
    const float CONFIDENCE_THRESHOLD = 0.25;

    // structs
    struct Detection
    {
        int class_id;
        float confidence;
        cv::Rect box;
    };

    // video input details
    double input_fps;
    int fw;
    int fh;

    // variables
    std::vector<std::string> class_list;
    int total_frames = 0;

    // methods
    void load_class_list(std::vector<std::string>& class_list);

    void load_net(cv::dnn::Net& net);

    cv::Mat format_yolov5(const cv::Mat& source);

    void detect(cv::Mat& image, cv::dnn::Net& net, std::vector<Detection>& output,
        const std::vector<std::string>& className);

    void drawBBox(cv::Mat& frame, std::vector<Detection>& output, const std::vector<std::string>& class_list);

    // init capture, writer and network
    cv::VideoCapture cap;
    cv::VideoWriter out;
    cv::dnn::Net net;
};

void ObjectDetector::load_class_list(std::vector<std::string> &class_list)
{
    std::ifstream ifs(path_class_input);
    std::string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
}

void ObjectDetector::load_net(cv::dnn::Net &net)
{   
    cv::dnn::Net result = cv::dnn::readNetFromONNX(path_net_input);
    if (cv::cuda::getCudaEnabledDeviceCount())
    {
        std::cout << "Running with CUDA\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }
    else
    {
        std::cout << "Running on CPU\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    net = result;
}

ObjectDetector::ObjectDetector(const std::string& directoryName, const std::string& sourcePath)
    : directoryName_(directoryName), sourcePath_(sourcePath)
{
    // Concatenate the directory name with another string
    path_video_input = sourcePath;
    path_video_output = directoryName + "output/detect_cpp.mp4";
    path_class_input = directoryName + "classes/classes_train.txt";
    path_net_input = directoryName + "models/best_all.onnx";

    // Open video input
    cap.open(path_video_input);

    if (!cap.isOpened())
    {
        std::cerr << "Error opening video file\n";
    }

    // Prints video input info
    input_fps = cap.get(cv::CAP_PROP_FPS);
    fw = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    fh = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::cout << "FPS: " << input_fps << ", Width: " << fw << ", Height: " << fh << std::endl;

    // Open video output
    out.open(path_video_output, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), input_fps, cv::Size(fw, fh));

    if (!out.isOpened())
    {
        std::cerr << "Error creating VideoWriter\n";
    }

    // Load class list
    load_class_list(class_list);

    // Load net
    load_net(net);
}

// cv::Mat ObjectDetector::format_yolov5(const cv::Mat& source) {
//     int col = source.cols;
//     int row = source.rows;
//     int _max = MAX(col, row);
//     cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
//     source.copyTo(result(cv::Rect(0, 0, col, row)));
//     return result;
// }

void ObjectDetector::detect(cv::Mat& image, cv::dnn::Net& net, std::vector<Detection>& output, const std::vector<std::string>& className) {
    cv::Mat blob;

    // auto input_image = format_yolov5(image);
    cv::dnn::blobFromImage(image, blob, 1./255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);

    // forward pass into network
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    float x_factor = image.cols / INPUT_WIDTH;
    float y_factor = image.rows / INPUT_HEIGHT;
    
    float *data = (float *)outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;
    
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // unwrap detections
    for (int i = 0; i < rows; ++i) {
        float confidence = data[4]; // conf in 4th address
        if (confidence >= CONFIDENCE_THRESHOLD) {
            float * classes_scores = data + 5; // address of class scores start 5 addresses away
            cv::Mat scores(1, className.size(), CV_32FC1, classes_scores); // create mat for score per detection
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id); // find max score
            // if (max_class_score > SCORE_THRESHOLD) {

                confidences.push_back(confidence); // add conf to vector
                class_ids.push_back(class_id.x); // add class_id to vector
                
                // center coords of bbox
                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            // }
        }
        data += 85; // next detection (x,y,w,h,conf,80 class conf)
    }
    // nms
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    for (int i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result); // add to output
    }
}

void ObjectDetector::drawBBox(cv::Mat& frame, std::vector<Detection>& output, const std::vector<std::string>& class_list)
{
    int detections = output.size();

    for (int i = 0; i < detections; ++i)
    {
        auto detection = output[i];
        auto box = detection.box;
        auto classId = detection.class_id;
        const auto color = colors[classId % colors.size()];
        cv::rectangle(frame, box, color, 3);
        cv::rectangle(frame, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
        cv::putText(frame, class_list[classId].c_str() + std::to_string(detection.confidence), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
}
int ObjectDetector::runObjectDetection()
{ 
    cv::Mat frame;

    while (true)
    {
        auto start = std::chrono::high_resolution_clock::now();

        cap.read(frame);

        if (frame.empty())
        {
            std::cout << "End of stream\n";
            break;
        }

        std::vector<Detection> output;
        detect(frame, net, output, class_list);
        drawBBox(frame, output, class_list);
        total_frames++;
        cv::imshow("output", frame);
        out.write(frame);

        if (cv::waitKey(1) != -1)
        {
            cap.release();
            std::cout << "finished by user\n";
            break;
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Time taken for detection in ms: " << time_taken  << std::endl;
    }

    std::cout << "Total frames: " << total_frames << "\n";
    cap.release();
    out.release();
    cv::destroyAllWindows();
    return 0;
}

int main(int argc, char *argv[])
{
    // Check if all three arguments are provided
    if (argc < 3)
    {
        std::cout << "Please provide /path/to/tracking_ws/ and /path/to/source" << std::endl;
        return 1;
    }

    // Get the directory name, source path, and tracker name from the arguments
    std::string directoryName = argv[1];
    std::string sourcePath = argv[2];

    // Concatenate the directory name with another string
    std::string concatenatedString = directoryName + "/another_string";

    ObjectDetector detector(directoryName, sourcePath);

    detector.runObjectDetection();

    return 0;
}
