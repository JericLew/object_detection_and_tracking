#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace std;

int main(int argc, char *argv[])
{
    int64 start = cv::getTickCount();

    // program input parameters
    string video_path;
    string output_path;

    string video_name;
    size_t last_slash_pos;
    size_t last_dot_pos;
    string video_name_no_ext;

    // Check if all three arguments are provided
    if (argc != 3)
    {
        cout << "Please provide /path/to/video/file/video.xxx and /path/to/output/folder/" << endl;
        return 1;
    }

    // Get the video path from args
    video_path = argv[1];
    output_path = argv[2];
    
    // Find the position of the last occurrence of '/'
    last_slash_pos = video_path.find_last_of('/');
    
    // Extract the substring after the last '/'
    video_name = video_path.substr(last_slash_pos + 1);
    
    // Find the position of the last occurrence of '.'
    last_dot_pos = video_name.find_last_of('.');
    
    // Extract the substring before the last '.'
    video_name_no_ext = video_name.substr(0, last_dot_pos);


    cv::VideoCapture cap(video_path);
    
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file" << std::endl;
        return 1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    
    int frameCount = 0;
    cv::Mat frame;
    
    while (cap.read(frame)) {
        std::string frame_name = output_path + video_name_no_ext + "_frame_" + std::to_string(frameCount) + ".jpg";
        cv::imwrite(frame_name, frame);
        
        frameCount++;
    }
    
    cap.release();
    
    std::cout << "Total frames extracted: " << frameCount << std::endl;

    int64 end = cv::getTickCount();
    double elapsedTime = (end - start) / cv::getTickFrequency();
    cout << "Total Elapsed Time: " << elapsedTime << " s" << std::endl;

    return 0;
}