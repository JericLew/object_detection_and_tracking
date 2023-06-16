#include "MOTCorrelationTracker.h"

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

void MOTCorrelationTracker::initTracker()
{
    cout << "********Init MOTCorrelationTracker********\n";
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
    loadClassList(class_list);

    // Load net
    loadNet(net);

    cv::namedWindow("Window", cv::WINDOW_NORMAL);
    cv::resizeWindow("Resized_Window", 1920, 1080);
}

void MOTCorrelationTracker::detect(cv::Mat& input_image, cv::dnn::Net &net, vector<Detection>& detector_output, const vector<string>& class_list)
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

    const int dimensions = 85; // x,y,w,h,conf + num of class conf
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
        data += 85; // next detection (x,y,w,h,conf,80 class conf)
    }
    // nms
    vector<int> nms_result;
    cv::dnn::NMSBoxes(bboxes, confidences, CLASS_CONF_THRES, NMS_THRES, nms_result);
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

void MOTCorrelationTracker::createTracker(cv::Mat& shrunk_frame, Detection& detection)
{
    cout << "Creating Tracker" << "ID:" << track_count << "...\n";
    /* https://github.com/opencv/opencv_contrib/blob/master/modules/tracking/samples/samples_utility.hpp */
    cv::Ptr<cv::Tracker> new_tracker;
    if (tracker_name == "MOSSE")
        new_tracker = cv::legacy::upgradeTrackingAPI(cv::legacy::TrackerMOSSE::create());
    else if (tracker_name=="KCF")
        new_tracker = cv::TrackerKCF::create();

    // Shrink detection bbox
    cv::Rect scaled_bbox = scaleBBox(detection.bbox, SCALE_FACTOR);

    cout << scaled_bbox.x << ' ' << scaled_bbox.y << ' ' << scaled_bbox.width << ' ' << scaled_bbox.height << endl;

    new_tracker->init(shrunk_frame, scaled_bbox);

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
    for (Track &track : multi_tracker)
    {
        bool isTracking = track.tracker->update(shrunk_frame, track.bbox);
        cout << track.bbox.x << ' ' << track.bbox.y << ' ' << track.bbox.width << ' ' << track.bbox.height << endl;
        track.bbox = scaleBBox(track.bbox, 1.0 / SCALE_FACTOR); // Enlarge shrunked bbox
    }
}

void MOTCorrelationTracker::drawBBox(cv::Mat &frame, vector<Detection>& detector_output, const vector<string> &class_list)
{
    cout << "Drawing BBox for detections...\n";
    int detections = detector_output.size();
    for (int i = 0; i < detections; ++i)
    {
        auto detection = detector_output[i];
        auto bbox = detection.bbox;
        auto classId = detection.class_id;
        const auto color = colors[classId % colors.size()];
        cv::rectangle(frame, bbox, color, 3);
        cv::rectangle(frame, cv::Point(bbox.x, bbox.y - 20), cv::Point(bbox.x + bbox.width, bbox.y), color, cv::FILLED);
        cv::putText(frame, class_list[classId].c_str(), cv::Point(bbox.x, bbox.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
}

void MOTCorrelationTracker::drawBBox(cv::Mat &frame, vector<Track> &multi_tracker, const vector<string> &class_list)
{
    cout << "Drawing BBox from trackers...\n";
    for (Track& track : multi_tracker)
    {
        if (track.num_hit < MIN_HITS)
        {
            continue;
        }
        const auto color = colors[track.class_id % colors.size()];
        cv::rectangle(frame, track.bbox, color, 3);
        cv::rectangle(frame, cv::Point(track.bbox.x, track.bbox.y - 20), cv::Point(track.bbox.x + track.bbox.width, track.bbox.y), color, cv::FILLED);
        cv::putText(frame, class_list[track.class_id].c_str() + to_string(track.track_id), cv::Point(track.bbox.x, track.bbox.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
}

void MOTCorrelationTracker::associate()
{
    cout << "Associating...\n";
    unsigned int num_of_tracks = 0;
    unsigned int num_of_detections = 0;

    num_of_tracks = multi_tracker.size();
    num_of_detections = detector_output.size();

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
        cout << "Assignment contents: ";
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

void MOTCorrelationTracker::updateTrackers(cv::Mat& shrunk_frame, vector<Detection>& detector_output)
{
    cout << "Updating Trackers...\n";
    // update matched trackers with assigned detections.
    // each prediction is corresponding to a tracker

    if (DEBUG_FLAG)
    {
        cout << "unmatched_tracks contents: ";
        for (const auto& i : unmatched_tracks) {
            cout << i << " ";
        }
        cout << endl;
        cout << "unmatched_detections contents: ";
        for (const auto& i : unmatched_detections) {
            cout << i << " ";
        }
        cout << endl;
        cout << "matched_pairs contents: ";
        for (const auto& i : matched_pairs) {
            cout << i << " ";
        }
        cout << endl;    
    }

    int detect_idx, track_idx;
    for (unsigned int i = 0; i < matched_pairs.size(); i++)
    {
        track_idx = matched_pairs[i].x;
        detect_idx = matched_pairs[i].y;
        multi_tracker[track_idx].num_hit++; // TODO change this to refresh track
        multi_tracker[track_idx].num_miss = 0;
    }

    // create and initialise new trackers for unmatched detections
    for (int unmatched_id : unmatched_detections)
    {
        createTracker(shrunk_frame, detector_output[unmatched_id]);
    }

    // num_miss++ for unmatched tracks
    for (int unmatched_id : unmatched_tracks)
    {
        multi_tracker[unmatched_id].num_miss++;
    }


    // delete dead tracks
    if (DEBUG_FLAG)
    {
        cout << "num miss: ";
        for (Track& track: multi_tracker) {
            cout << track.num_miss << " ";
        }
        cout << endl;           
    }
 

    for (int i = 0; i < multi_tracker.size(); i++)
    {
        if (multi_tracker[i].num_miss > MAX_AGE)
        {
            multi_tracker.erase(multi_tracker.begin() + i);
        }
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
            cout << "********End of stream********\n";
            break;
        }

        // resize for track
        cv::Mat shrunk_frame;
        cv::resize(frame, shrunk_frame, cv::Size(), SCALE_FACTOR, SCALE_FACTOR, cv::INTER_AREA);

        if (total_frames == 10)
        {
            detect(frame, net, detector_output, class_list);
            int detections = detector_output.size();
            for (int i = 0; i < detections; ++i)
            {
                createTracker(shrunk_frame, detector_output[i]);
            }
        }

        else if (total_frames > 10 && total_frames%10 == 0)
        {
            detector_output.clear();
            getTrackersPred(shrunk_frame);
            detect(frame, net, detector_output, class_list);
            associate();
            updateTrackers(shrunk_frame, detector_output);
        }

        else
        {
            getTrackersPred(shrunk_frame);
        }

        drawBBox(frame, multi_tracker, class_list);
        total_frames++;
        cv::imshow("Window", frame);
        out.write(frame);

        if (DEBUG_FLAG)
        {
            cout << "Tracker contents: ";
            for (const Track& track : multi_tracker) {
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
