#include "Detection.h"

using namespace std;

ObjectDetector::ObjectDetector(){}

void ObjectDetector::inputPaths(const string& directory_name, const string& source_path)
{
    // Concatenate the directory name with another string
    path_video_input = source_path;
    path_video_output = directory_name + "output/detect_cpp.mp4";
    path_class_input = directory_name + "classes/classes_train.txt";
    path_net_input = directory_name + "models/best_all.onnx";    
}

void ObjectDetector::loadClassList(vector<string>& class_list)
{
    cout << "Loading Class List...\n";
    ifstream ifs(path_class_input);
    string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
}

void ObjectDetector::loadNet(cv::dnn::Net& net)
{   
    cv::dnn::Net result = cv::dnn::readNet(path_net_input);
    if (cv::cuda::getCudaEnabledDeviceCount())
    {
        cout << "Running Detection with CUDA...\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }
    else
    {
        cout << "Running Detection on CPU...\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    net = result;
}

void ObjectDetector::warmupNet(cv::dnn::Net& net)
{
    cout << "Warming Up Detector" << endl;
    int64 start_warmup = cv::getTickCount();

    cv::Mat dummy_image = cv::Mat::zeros(cv::Size(INPUT_WIDTH, INPUT_HEIGHT), CV_8UC3);  // Create a dummy black image

    // Preprocess the dummy image
    cv::Mat dummy_blob;
    cv::dnn::blobFromImage(dummy_image, dummy_blob, 1.0, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);

    // Warm up the network by performing a forward pass
    net.setInput(dummy_blob);
    vector<cv::Mat> dummy_outputs;
    net.forward(dummy_outputs, net.getUnconnectedOutLayersNames());

    int64 end_warmup = cv::getTickCount();
    double warmup_Time = (end_warmup - start_warmup) * 1000 / cv::getTickFrequency();
    cout << "Warm-Up Time: " << warmup_Time << " ms" <<endl; 
}

void ObjectDetector::initDetector()
{
    cout << "********Start Init ObjectDetector********\n";
    // Open video input
    cap.open(path_video_input);
    if (!cap.isOpened())
    {
        cerr << "Error opening video file\n";
    }

    // Prints video input info
    input_fps = cap.get(cv::CAP_PROP_FPS);
    fw = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    fh = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    cout << "FPS: " << input_fps << ", Width: " << fw << ", Height: " << fh << endl;

    // Open video output
    out.open(path_video_output, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), input_fps, cv::Size(fw, fh));
    if (!out.isOpened())
    {
        cerr << "Error creating VideoWriter\n";
    }

    loadClassList(class_list);
    loadNet(net);
    warmupNet(net);

    cv::namedWindow("Window", cv::WINDOW_NORMAL);
    cv::resizeWindow("Window", 1920, 1080);
    cout << "********End Init ObjectDetector********\n\n";
}

void ObjectDetector::detect(cv::Mat& input_image, cv::dnn::Net &net, vector<Detection>& detector_output, const vector<string>& class_list)
{
    cout << "Detecting...\n";

    cv::Mat blob;

    cv::Mat formatted_image = formatYOLOv5(input_image);

    cv::dnn::blobFromImage(formatted_image, blob, 1. / 255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);

    // forward pass into network
    net.setInput(blob);
    vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    float x_factor = formatted_image.cols / INPUT_WIDTH;
    float y_factor = formatted_image.rows / INPUT_HEIGHT;

    float *data = (float *)outputs[0].data;

    const int dimensions = 5 + class_list.size(); // x,y,w,h,conf + num of class conf
    const int rows = 25200;

    vector<int> class_ids;
    vector<float> confidences;
    vector<cv::Rect> bboxes;

    // unwrap detections
    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4]; // conf in 4th address
        if (confidence >= DETECT_CONF_THRES)
        {
            float *classes_scores = data + 5; // address of class scores start 5 addresses away
            cv::Mat scores(1, class_list.size(), CV_32FC1, classes_scores); // create mat for score per detection
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id); // find max score
            if (max_class_score > CLASS_CONF_THRES)
            {
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
                bboxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        data += dimensions; // next detection (x,y,w,h,conf,num of class conf)
    }
    // nms
    vector<int> nms_result;
    cv::dnn::NMSBoxes(bboxes, confidences, DETECT_CONF_THRES, NMS_THRES, nms_result);
    for (int i = 0; i < nms_result.size(); i++)
    {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.bbox = bboxes[idx];
        detector_output.push_back(result); // add to detector_output
    }
}

void ObjectDetector::drawBBox(cv::Mat &frame, vector<Detection>& detector_output, const vector<string>& class_list)
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

int ObjectDetector::runObjectDetection()
{
    cv::Mat frame;

    while (true)
    {
        int64 start_loop = cv::getTickCount();

        cap.read(frame);

        if (frame.empty())
        {
            cout << "********End of stream********\n";
            break;
        }

        detector_output.clear();
        detect(frame, net, detector_output, class_list);
        drawBBox(frame, detector_output, class_list);
        total_frames++;
        cv::imshow("Window", frame);
        out.write(frame);

        if (cv::waitKey(1) != -1)
        {
            cap.release();
            cout << "********Ended by User********\n";
            break;
        }

        int64 end_loop = cv::getTickCount();
        double elasped_time_loop = (end_loop - start_loop) * 1000 / cv::getTickFrequency();
        cout << "Elapsed Time: " << elasped_time_loop << " ms" << std::endl;
        cout << endl;
    }

    cout << "Total frames: " << total_frames << "\n";
    cap.release();
    out.release();
    cv::destroyAllWindows();
    return 0;
}
