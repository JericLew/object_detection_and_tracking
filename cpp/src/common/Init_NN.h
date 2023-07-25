///////////////////////////////////////////////////////////////////////////////
// Init_NN.h: Header file containing functions to initialise object detection network
//
// by Jeric,2023
// 

#ifndef INITNN_H
#define INITNN_H

#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace std;

/********Functions********/
void loadClassList(vector<string>& class_list, const string &path_class_input);
void loadNet(cv::dnn::Net& net, const string &path_net_input);
void warmupNet(cv::dnn::Net& net, float INPUT_WIDTH, float INPUT_HEIGHT);


#endif
