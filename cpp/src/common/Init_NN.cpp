///////////////////////////////////////////////////////////////////////////////
// Init_NN.cpp: Source file containing functions to initialise object detection network
//
// by Jeric,2023
// 

#include "Init_NN.h"

using namespace std;

void loadClassList(vector<string>& class_list, const string &path_class_input)
{
    cout << "Loading Class List...\n";
    ifstream ifs(path_class_input);
    string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
}

void loadNet(cv::dnn::Net& net, const string &path_net_input)
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

void warmupNet(cv::dnn::Net& net, float INPUT_WIDTH, float INPUT_HEIGHT)
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
