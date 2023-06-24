#include "new_detect.h"

int main(int argc, char *argv[])
{
    int64 start = cv::getTickCount();

    // program input parameters
    string directory_name;
    string source_path;

    // Check if all three arguments are provided
    if (argc < 3)
    {
        cout << "Please provide /path/to/tracking_ws/and  /path/to/source" << endl;
        return 1;
    }

    // Get the directory name and source path from the arguments
    directory_name = argv[1];
    source_path = argv[2];

    ObjectDetector detector;
    detector.inputPaths(directory_name, source_path);
    detector.initDetector();
    detector.runObjectDetection();

    int64 end = cv::getTickCount();
    double elasped_time = (end - start) / cv::getTickFrequency();
    cout << "Total Elapsed Time: " << elasped_time << " s" << std::endl;
    cout << endl;

    return 0;
}
