#include "MOTCorrelationTracker.h"
#include "Hungarian.h"

using namespace std;

MOTCorrelationTracker::MOTCorrelationTracker(){}

void MOTCorrelationTracker::inputPaths(const string& directory_name, const string& source_path, const string& tracker_name)
{
    // Concatenate the directory name with another string
    path_video_input = source_path;
    path_video_output = directory_name + "video_output_c++/track_ass_output.mp4";
    path_class_input = directory_name + "classes/classes.txt";
    path_net_input = directory_name + "models/yolov5s.onnx";
    MOTCorrelationTracker::tracker_name = tracker_name;
    
}

void MOTCorrelationTracker::load_class_list(vector<string>& class_list)
{
    cout << "Loading Class List\n";
    ifstream ifs(path_class_input);
    string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
}

void MOTCorrelationTracker::load_net(cv::dnn::Net& net)
{   
    cv::dnn::Net result = cv::dnn::readNet(path_net_input);
    if (cv::cuda::getCudaEnabledDeviceCount())
    {
        cout << "Running with CUDA\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }
    else
    {
        cout << "Running on CPU\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    net = result;
}

void MOTCorrelationTracker::initTracker()
{
    cout << "Init MOTCorrelationTracker\n";
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

    // Load class list
    load_class_list(class_list);

    // Load net
    load_net(net);

    cv::namedWindow("Window", cv::WINDOW_NORMAL);
    cv::resizeWindow("Resized_Window", 1920, 1080);
}

cv::Mat MOTCorrelationTracker::format_yolov5(const cv::Mat &source)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

void MOTCorrelationTracker::detect(cv::Mat &image, cv::dnn::Net &net, vector<Detection>& detector_output, const vector<string> &className)
{
    cout << "Detecting\n";
    cv::Mat blob;

    auto input_image = format_yolov5(image);

    cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);

    // forward pass into network
    net.setInput(blob);
    vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    float *data = (float *)outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;

    vector<int> class_ids;
    vector<float> confidences;
    vector<cv::Rect> boxes;

    // unwrap detections
    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4]; // conf in 4th address
        if (confidence >= CONFIDENCE_THRESHOLD)
        {
            float *classes_scores = data + 5;                              // address of class scores start 5 addresses away
            cv::Mat scores(1, className.size(), CV_32FC1, classes_scores); // create mat for score per detection
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id); // find max score
            if (max_class_score > SCORE_THRESHOLD)
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
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        data += 85; // next detection (x,y,w,h,conf,80 class conf)
    }
    // nms
    vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    for (int i = 0; i < nms_result.size(); i++)
    {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        detector_output.push_back(result); // add to detector_output
    }
}

void MOTCorrelationTracker::createTracker(cv::Mat &frame, Detection& detection)
{
    cout << "Creating Tracker" << "ID:" << track_count << "\n";
    /* https://github.com/opencv/opencv_contrib/blob/master/modules/tracking/samples/samples_utility.hpp */
    cv::Ptr<cv::Tracker> new_tracker;
    if (tracker_name == "MOSSE")
        new_tracker = cv::legacy::upgradeTrackingAPI(cv::legacy::TrackerMOSSE::create());
    else if (tracker_name=="KCF")
        new_tracker = cv::TrackerKCF::create();
    new_tracker->init(frame, detection.box);

    Track new_track;
    new_track.track_id = track_count;
    new_track.tracker = new_tracker;
    new_track.class_id = detection.class_id;
    new_track.confidence = detection.confidence;
    new_track.box = detection.box;
    new_track.num_hit = 1;
    new_track.num_miss = 0;
    multi_tracker.push_back(new_track);
    track_count++;
}

void MOTCorrelationTracker::getTrackersPred(cv::Mat &frame)
{
    cout << "Getting Trackers Predictions\n";
    for (Track &track : multi_tracker)
    {
        bool isTracking = track.tracker->update(frame, track.box);
    }
}

void MOTCorrelationTracker::drawBBox(cv::Mat &frame, vector<Detection>& detector_output, const vector<string> &class_list)
{
    cout << "Drawing BBox for detections\n";
    int detections = detector_output.size();
    for (int i = 0; i < detections; ++i)
    {
        auto detection = detector_output[i];
        auto box = detection.box;
        auto classId = detection.class_id;
        const auto color = colors[classId % colors.size()];
        cv::rectangle(frame, box, color, 3);
        cv::rectangle(frame, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
        cv::putText(frame, class_list[classId].c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
}

void MOTCorrelationTracker::drawBBox(cv::Mat &frame, vector<Track> &multi_tracker, const vector<string> &class_list)
{
    cout << "Drawing BBox from trackers\n";
    for (Track& track : multi_tracker)
    {
        if (track.num_hit < min_hits)
        {
            continue;
        }
        const auto color = colors[track.class_id % colors.size()];
        cv::rectangle(frame, track.box, color, 3);
        cv::rectangle(frame, cv::Point(track.box.x, track.box.y - 20), cv::Point(track.box.x + track.box.width, track.box.y), color, cv::FILLED);
        cv::putText(frame, class_list[track.class_id].c_str() + to_string(track.track_id), cv::Point(track.box.x, track.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
}

// Computes IOU between two bounding boxes
double GetIOU(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt)
{
	float in = (bb_test & bb_gt).area();
	float un = bb_test.area() + bb_gt.area() - in;

	if (un < DBL_EPSILON)
		return 0;

	return (double)(in / un);
}

void MOTCorrelationTracker::associate()
{
    cout << "Associating tracks and detections\n";
    unsigned int num_of_tracks = 0;
    unsigned int num_of_detections = 0;

    num_of_tracks = multi_tracker.size();
    num_of_detections = detector_output.size();

    iouMatrix.clear();
    iouMatrix.resize(num_of_tracks, vector<double>(num_of_detections, 0));

    // compute iou matrix as a distance matrix
    for (unsigned int i = 0; i < num_of_tracks; i++) 
    {
        for (unsigned int j = 0; j < num_of_detections; j++)
        {
            // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
            iouMatrix[i][j] = 1 - GetIOU(multi_tracker[i].box, detector_output[j].box);
        }
    }

    // solve the assignment problem using hungarian algorithm.
    // the resulting assignment is [track(prediction) : detection], with len=preNum
    HungarianAlgorithm HungAlgo;
    assignment.clear();
    HungAlgo.Solve(iouMatrix, assignment);

    // cout << "Vector contents: ";
    // for (const auto& i : assignment) {
    //     cout << i << " ";
    // }
    // cout << endl;

    unmatchedTracks.clear();
    unmatchedDetections.clear();
    allItems.clear();
    matchedItems.clear();
    
    if (num_of_detections > num_of_tracks) //	there are unmatched detections
    {
        for (unsigned int n = 0; n < num_of_detections; n++)
            allItems.insert(n);

        for (unsigned int i = 0; i < num_of_tracks; ++i)
            matchedItems.insert(assignment[i]);

        // insert the set difference between all detections and all matched items
        // leaves unmatched detection id in unmatchedDetections set
        set_difference(allItems.begin(), allItems.end(),
            matchedItems.begin(), matchedItems.end(),
            insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
    }
    else if (num_of_detections < num_of_tracks) // there are unmatched tracks
    {
        for (unsigned int i = 0; i < num_of_tracks; ++i)
            if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
                unmatchedTracks.insert(i);
    }

    // filter out matches with low IOU
    matchedPairs.clear();
    for (unsigned int i = 0; i < num_of_tracks; ++i)
    {
        if (assignment[i] == -1) // pass over invalid values
            continue;
        if (1 - iouMatrix[i][assignment[i]] < iouThreshold)
        {
            unmatchedTracks.insert(i);
            unmatchedDetections.insert(assignment[i]);
        }
        else
            matchedPairs.push_back(cv::Point(i, assignment[i]));
    }
}

void MOTCorrelationTracker::updateTrackers(cv::Mat &frame, vector<Detection>& detector_output)
{
    cout << "Updating Trackers\n";
    // update matched trackers with assigned detections.
    // each prediction is corresponding to a tracker
    int detIdx, trkIdx;
    for (unsigned int i = 0; i < matchedPairs.size(); i++)
    {
        trkIdx = matchedPairs[i].x;
        detIdx = matchedPairs[i].y;
        multi_tracker[trkIdx].num_hit++; // TODO change this to refresh track
        multi_tracker[trkIdx].num_miss = 0;
    }

    // create and initialise new trackers for unmatched detections
    for (int unmatched_id : unmatchedDetections)
    {
        createTracker(frame, detector_output[unmatched_id]);
    }

    // num_miss++ for unmatched tracks
    for (int unmatched_id : unmatchedTracks)
    {
        multi_tracker[unmatched_id].num_miss++;
    }

    // delete dead tracks
    for (auto it = multi_tracker.begin(); it != multi_tracker.end();)
    {
        if (it != multi_tracker.end() && (*it).num_miss > max_age)
        {
            it = multi_tracker.erase(it);
        }
        it++;
    }
}

int MOTCorrelationTracker::runObjectTracking()
{
    cv::Mat frame;

    while (true)
    {
        int64 start = cv::getTickCount();

        cap.read(frame);

        if (frame.empty())
        {
            cout << "End of stream\n";
            break;
        }

        if (total_frames == 10)
        {
            detect(frame, net, detector_output, class_list);
            int detections = detector_output.size();
            for (int i = 0; i < detections; ++i)
            {
                createTracker(frame, detector_output[i]);
            }
        }

        else if (total_frames > 10 && total_frames%10 == 0)
        {
            detector_output.clear();
            getTrackersPred(frame);
            detect(frame, net, detector_output, class_list);
            associate();
            updateTrackers(frame, detector_output);
        }

        else
        {
            getTrackersPred(frame);
        }

        drawBBox(frame, multi_tracker, class_list);
        total_frames++;
        cv::imshow("Window", frame);
        out.write(frame);


        std::cout << "Tracker contents: ";
        for (const Track& track : multi_tracker) {
            std::cout << track.track_id << " ";
        }
        std::cout << std::endl;


        if (cv::waitKey(1) != -1)
        {
            cap.release();
            cout << "finished by user\n";
            break;
        }

        int64 end = cv::getTickCount();
        double elapsedTime = (end - start) * 1000 / cv::getTickFrequency();
        cout << "Elapsed Time: " << elapsedTime << " ms" << std::endl;
        cout << endl;
    }

    cout << "Total frames: " << total_frames << "\n";
    cap.release();
    out.release();
    cv::destroyAllWindows();
    return 0;
}
