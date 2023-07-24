#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>

int main() {
    std::ifstream infile("/home/jeric/MOT15/train/ADL-Rundle-6/det/det.txt"); // Replace "input.txt" with the actual path to your input file
    
    // Map to store the lines grouped by frame value
    std::map<int, std::vector<std::vector<float>>> frameMap;
    
    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        
        int frame;
        char comma;
        float id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z;
        
        // Parse the line into respective variables
        if (!(iss >> frame >> comma >> id >> comma >> bb_left >> comma >> bb_top >> comma >> bb_width >> comma >> bb_height
              >> comma >> conf >> comma >> x >> comma >> y >> comma >> z)) {
            // Error parsing the line, skip it
            continue;
        }
        
        // Create the bounding box vector
        std::vector<float> boundingBox = { bb_left, bb_top, bb_width, bb_height, conf };
        
        // Add the bounding box vector to the frameMap
        frameMap[frame].push_back(boundingBox);
    }
    
    // Iterate over the frameMap and print the matrix for each frame
    for (const auto& framePair : frameMap) {
        int frame = framePair.first;
        const std::vector<std::vector<float>>& boundingBoxes = framePair.second;
        
        // Print the frame number
        std::cout << "Frame " << frame << ":" << std::endl;
        
        // Print the matrix for the current frame
        for (const auto& box : boundingBoxes) {
            std::cout << "[ ";
            for (const auto& value : box) {
                std::cout << value << ", ";
            }
            std::cout << "]" << std::endl;
        }
    }
    
    return 0;
}
