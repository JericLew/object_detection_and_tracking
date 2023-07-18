#include "EvalTest.h"

using namespace std;

MOTCorrelationTracker::MOTCorrelationTracker(){}

void MOTCorrelationTracker::inputPaths(const string& directory_name, const string& source_path, const string& tracker_name)
{
    // Concatenate the directory name with another string
    path_video_input = source_path;
    path_video_output = directory_name + "output/track_ass_cpp.mp4";
    path_class_input = directory_name + "classes/classes_train.txt";
    path_net_input = directory_name + "models/best_all.onnx";
    MOTCorrelationTracker::tracker_name = tracker_name;
    if (tracker_name == "KCF") //KCF BUGGY
        cout << "KCF results in old detections, please use MOSSE";
    else if (tracker_name == "TLD")
        cout << "TLD result in negative value in bbox, please use MOSSE";
    else if (tracker_name == "BOOSTING")
        cout << "BOOSTING result in negative value in bbox, please use MOSSE";
    else if (tracker_name == "MEDIAN_FLOW")
        cout << "Using MIL";
    else if (tracker_name == "MIL")
        cout << "MIL is slow, please use MOSSE";
    else if (tracker_name == "GOTURN")
        cout << "GOTURN not supported, please use MOSSE";
    else if (tracker_name == "MOSSE")
        cout << "Using MOSSE";
    else if (tracker_name == "CSRT")
        cout << "CSRT is slow, please use MOSSE"; 
}

void MOTCorrelationTracker::loadClassList(vector<string>& class_list)
{
    cout << "Loading Class List...\n";
    ifstream ifs(path_class_input);
    string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
}

void MOTCorrelationTracker::loadNet(cv::dnn::Net& net)
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

void MOTCorrelationTracker::warmupNet(cv::dnn::Net& net)
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

void MOTCorrelationTracker::initTracker()
{
    cout << "********Start Init MOTCorrelationTracker********\n";
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
    cout << "********End Init MOTCorrelationTracker********\n\n";
}

void MOTCorrelationTracker::createTracker(cv::Mat& shrunk_frame, Detection& detection, cv::Ptr<cv::Tracker>& new_tracker)
{
    /*
    https://github.com/opencv/opencv_contrib/blob/master/modules/tracking/samples/samples_utility.hpp 
    */
    if (tracker_name == "KCF") //KCF BUGGY
        new_tracker = cv::TrackerKCF::create();
    else if (tracker_name == "TLD")
        new_tracker = cv::legacy::upgradeTrackingAPI(cv::legacy::TrackerTLD::create());
    else if (tracker_name == "BOOSTING")
        new_tracker = cv::legacy::upgradeTrackingAPI(cv::legacy::TrackerBoosting::create());
    else if (tracker_name == "MEDIAN_FLOW")
        new_tracker = cv::legacy::upgradeTrackingAPI(cv::legacy::TrackerMedianFlow::create());
    else if (tracker_name == "MIL")
        new_tracker = cv::TrackerMIL::create();
    else if (tracker_name == "GOTURN")
        new_tracker = cv::TrackerGOTURN::create();
    else if (tracker_name == "MOSSE")
        new_tracker = cv::legacy::upgradeTrackingAPI(cv::legacy::TrackerMOSSE::create());
    else if (tracker_name == "CSRT")
        new_tracker = cv::TrackerCSRT::create();

    // Shrink detection bbox
    cv::Rect scaled_bbox = scaleBBox(detection.bbox, SCALE_FACTOR);

    new_tracker->init(shrunk_frame, scaled_bbox);    
}

void MOTCorrelationTracker::createTrack(cv::Mat& shrunk_frame, Detection& detection)
{
    cout << "Creating Track" << "ID:" << track_count << "...\n";

    cv::Ptr<cv::Tracker> new_tracker;
    createTracker(shrunk_frame, detection, new_tracker);

    Track new_track;
    new_track.track_id = track_count;
    new_track.tracker = new_tracker;
    new_track.class_id = detection.class_id;
    new_track.confidence = detection.confidence;    
    new_track.bbox = detection.bbox;
    new_track.num_hit = 1;
    new_track.num_miss = 0;
    multi_tracker.push_back(new_track);
    track_count++;
}

void MOTCorrelationTracker::getTrackersPred(cv::Mat& shrunk_frame)
{
    cout << "Getting Trackers Predictions...\n";
    for (int i = 0; i < multi_tracker.size(); i++)
    {
        Track& track = multi_tracker[i];
        bool track_status = track.tracker->update(shrunk_frame, track.bbox);
        if (!track_status) // if track update failed, bbox = 0
        {
            track.bbox.x = 0;
            track.bbox.y = 0;
            track.bbox.width = 0;
            track.bbox.height = 0;
        } 

        if (DEBUG_FLAG)
        {
            cout << "Track Status: " << (track_status ? "Found" : "Lost") << endl;
            cout << "tlwh: " << track.bbox.x << ' ' << track.bbox.y << ' ' << track.bbox.width << ' ' << track.bbox.height << endl;
        }

        track.bbox = scaleBBox(track.bbox, 1.0 / SCALE_FACTOR); // Enlarge shrunked bbox
    }
}

void MOTCorrelationTracker::drawBBox(cv::Mat &frame, vector<Detection>& detector_output, const vector<string>& class_list)
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

void MOTCorrelationTracker::drawBBox(cv::Mat &frame, vector<Track> &multi_tracker, const vector<string> &class_list)
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

void MOTCorrelationTracker::associate()
{
    cout << "Associating...\n";
    unsigned int num_of_tracks = 0;
    unsigned int num_of_detections = 0;

    num_of_tracks = multi_tracker.size();
    num_of_detections = detector_output.size();

    if (num_of_tracks == 0)
    {
        for (int i = 0; i < num_of_detections; i++)
        {
            unmatched_detections.insert(i);
        }
        return;
    }

    iou_matrix.clear();
    iou_matrix.resize(num_of_tracks, vector<double>(num_of_detections, 0));

    // compute iou matrix as a distance matrix
    for (unsigned int i = 0; i < num_of_tracks; i++) 
    {
        for (unsigned int j = 0; j < num_of_detections; j++)
        {
            // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
            iou_matrix[i][j] = 1 - GetIOU(multi_tracker[i].bbox, detector_output[j].bbox);
        }
    }

    // solve the assignment problem using hungarian algorithm.
    // the resulting assignment is [track(prediction) : detection], with len=preNum
    HungarianAlgorithm HungAlgo;
    assignment.clear();
    HungAlgo.Solve(iou_matrix, assignment);

    if (DEBUG_FLAG)
    {
        cout << "assignment: ";
        for (const auto& i : assignment) {
            cout << i << " ";
        }
        cout << endl;        
    }

    unmatched_tracks.clear();
    unmatched_detections.clear();
    all_detections.clear();
    matched_detections.clear();
    
    if (num_of_detections > num_of_tracks) //	there are unmatched detections
    {
        for (unsigned int n = 0; n < num_of_detections; n++)
            all_detections.insert(n);

        for (unsigned int i = 0; i < num_of_tracks; ++i)
            matched_detections.insert(assignment[i]);

        // insert the set difference between all detections and all matched items
        // leaves unmatched detection id in unmatched_detections set
        set_difference(all_detections.begin(), all_detections.end(),
            matched_detections.begin(), matched_detections.end(),
            insert_iterator<set<int>>(unmatched_detections, unmatched_detections.begin()));
    }
    else if (num_of_detections < num_of_tracks) // there are unmatched tracks
    {
        for (unsigned int i = 0; i < num_of_tracks; ++i)
            if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
                unmatched_tracks.insert(i);
    }

    // filter out matches with low IOU
    matched_pairs.clear();
    for (unsigned int i = 0; i < num_of_tracks; ++i)
    {
        if (assignment[i] == -1) // pass over invalid values
            continue;
        if (1 - iou_matrix[i][assignment[i]] < IOU_THRES)
        {
            unmatched_tracks.insert(i);
            unmatched_detections.insert(assignment[i]);
        }
        else
            matched_pairs.push_back(cv::Point(i, assignment[i]));
    }
}

void MOTCorrelationTracker::updateTracks(cv::Mat& shrunk_frame, vector<Detection>& detector_output)
{
    cout << "Updating Tracks...\n";
    // update matched trackers with assigned detections.
    // each prediction is corresponding to a tracker

    if (DEBUG_FLAG)
    {
        cout << "unmatched_tracks: ";
        for (const auto& i : unmatched_tracks) {
            cout << i << " ";
        }
        cout << endl;
        cout << "unmatched_detections: ";
        for (const auto& i : unmatched_detections) {
            cout << i << " ";
        }
        cout << endl;
        cout << "matched_pairs: ";
        for (const auto& i : matched_pairs) {
            cout << i << " ";
        }
        cout << endl;    
    }

    int detect_idx, track_idx; // for matched pairs TODO refresh track is wonky and can cause lost in track
    for (unsigned int i = 0; i < matched_pairs.size(); i++)
    {
        track_idx = matched_pairs[i].x;
        detect_idx = matched_pairs[i].y;
        double iou_score = GetIOU(detector_output[detect_idx].bbox, multi_tracker[track_idx].bbox);
        if (iou_score < REFRESH_IOU_THRES)
        {
            cout << "refresh track ID: " << multi_tracker[track_idx].track_id << endl;
            cv::Ptr<cv::Tracker> new_tracker;
            createTracker(shrunk_frame, detector_output[detect_idx], new_tracker);
            multi_tracker[track_idx].tracker = new_tracker;
        }

        multi_tracker[track_idx].num_hit++; 
        multi_tracker[track_idx].num_miss = 0;
    }

    // create and initialise new trackers for unmatched detections
    for (int unmatched_id : unmatched_detections)
    {
        createTrack(shrunk_frame, detector_output[unmatched_id]);
    }

    // num_miss++ for unmatched tracks
    for (int unmatched_id : unmatched_tracks)
    {
        multi_tracker[unmatched_id].num_miss++;
    }

    if (DEBUG_FLAG)
    {
        cout << "Num Miss: ";
        for (int i = 0; i < multi_tracker.size(); i++)
        {
            Track& track = multi_tracker[i];
            cout << track.num_miss << " ";
        }
        cout << endl;           
    }
 
    // delete dead tracks
    for (int i = 0; i < multi_tracker.size(); i++)
    {
        if (multi_tracker[i].num_miss > MAX_AGE)
        {
            multi_tracker.erase(multi_tracker.begin() + i);
        }
    }
}

void MOTCorrelationTracker::extract_detections(){
    std::ifstream infile("/home/jeric/MOT15/train/KITTI-13/det/det.txt"); // Replace "input.txt" with the actual path to your input file
    
    // Map to store the lines grouped by frame value
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
        Detection detection;
        detection.class_id = 0;
        detection.confidence = conf/100;
        detection.bbox = cv::Rect(bb_left, bb_top, bb_width, bb_height);
        
        // Add the bounding box vector to the frameMap
        frameMap[frame].push_back(detection);
    }
    
    // Iterate over the frameMap and print the matrix for each frame
    for (const auto& framePair : frameMap) {
        int frame = framePair.first;
        const std::vector<Detection>& detections = framePair.second;
        
        // Print the frame number
        std::cout << "Frame " << frame << ":" << std::endl;
        
        // Print the matrix for the current frame
        for (const Detection& detection : detections) {
            cout << detection.confidence << endl;
        }
    }
}

int MOTCorrelationTracker::runObjectTracking()
{
    cv::Mat frame;

    extract_detections();
    auto lastFrameIterator = frameMap.rbegin();  // Get an iterator to the last added element
    int last_frame = lastFrameIterator->first;  // Retrieve the frame number of the last added element

    cout << last_frame << endl;
    string folderPath = "/home/jeric/MOT15/train/KITTI-13/img1/";
    string fileExtension = ".jpg";
    
    for (int i = 1; i <= last_frame; i++)
    {
        std::stringstream ss;
        ss << std::setw(6) << std::setfill('0') << i;
        std::string fileName = ss.str() + ".jpg";  // Assuming the images are in JPEG format, adjust the extension if necessary

        // Construct the full image file path
        std::string imagePath = folderPath + fileName;

        // Read the image using cv::imread()
        cv::Mat frame = cv::imread(imagePath);

        int64 start_loop = cv::getTickCount();

        // // resize for track
        // cv::Mat shrunk_frame;
        // cv::resize(frame, shrunk_frame, cv::Size(), SCALE_FACTOR, SCALE_FACTOR, cv::INTER_AREA);


        detector_output.clear();
        getTrackersPred(frame);
        detector_output =  frameMap[i-1];
        associate();
        updateTracks(frame, detector_output);


        // if (total_frames == 10)
        // {
        //     detect(frame, net, detector_output, class_list);
        //     int detections = detector_output.size();
        //     for (int i = 0; i < detections; ++i)
        //     {
        //         createTrack(shrunk_frame, detector_output[i]);
        //     }
        // }

        // else if (total_frames > 10 && total_frames%10== 0)
        // {
        //     detector_output.clear();
        //     getTrackersPred(shrunk_frame);
        //     detect(frame, net, detector_output, class_list);
        //     associate();
        //     updateTracks(shrunk_frame, detector_output);
        // }

        // else
        // {
        //     int64 start_track = cv::getTickCount();
        //     getTrackersPred(shrunk_frame);
        //     int64 end_track = cv::getTickCount();
        //     double elasped_time_track = (end_track - start_track) * 1000 / cv::getTickFrequency();
        //     cout << "Track Time: " << elasped_time_track << " ms" << std::endl;
        // }

        // drawBBox(frame, detector_output, class_list);
        drawBBox(frame, multi_tracker, class_list);
        total_frames++;
        cv::imshow("Window", frame);
        out.write(frame);

        if (DEBUG_FLAG)
        {
            cout << "Tracker Contents: ";
            for (int i = 0; i < multi_tracker.size(); i++)
            {
                Track& track = multi_tracker[i];
                cout << track.track_id << " ";
            }
            cout << endl;            
        }

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
