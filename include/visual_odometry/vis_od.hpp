#pragma once

#include <iostream>
#include <ctime>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include "feature.hpp"
// #include "rclcpp/rclcpp.hpp"
// #include "sensor_msgs/msg/image.hpp"
// #include "std_msgs/msg/header.hpp"
// #include "geometry_msgs/msg/pose.hpp"
// #include <cv_bridge/cv_bridge.h>
#include <chrono>
#include <fstream>
#include <sstream>
#include "matplotlibcpp.h"

calib_data projMat1;
calib_data projMat2;
Mat img0,img1,imgt0;
Mat left_frame_new;
Mat right_frame_new;
Mat left_frame;
Mat right_frame;
Mat left_frame1;
Mat right_frame1;
Mat dist_gray_frame, dist_gray_frame2, dist_gray_frame3;
Mat gray_frame, gray_frame2, gray_frame3;
VideoCapture cap;
Mat match_img, match_img2;
Mat R,t;
Mat trans_flat(1,12,CV_64F);
Mat trans_mat(4,4,CV_64F);
Mat R_trans(3,3,CV_64F);
Mat t_trans(3,1,CV_64F);
Mat proj_(3,4,CV_64F);

vector<cv::String> dir_vec;
vector<cv::String> dir_vec1;
int count_ =0;
int count_var;

vector<KeyPoint> kp;

vector<KeyPoint> kp2;

Mat points4d;
vector<Point3d> points;
vector<DMatch> good_matches;

void match_image(Mat& , Mat&, Mat&, calib_data& , calib_data&, Mat&, String&);
void augment(Mat&, Mat&, Mat&);
void comp_path(Mat&, Mat&, Mat&);
void get_config(vector<double>&, Mat&);
