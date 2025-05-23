#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <yaml-cpp/yaml.h>
#include <cmath>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Geometry>




using namespace std;
using namespace cv;
using namespace Eigen;

struct calib_data{
	Mat cam, dist, proj;
	double b;

	calib_data() : 	cam(Mat::zeros(3, 3, CV_64F)),
					dist(Mat::zeros(1, 5, CV_64F)),
					proj(Mat::zeros(3, 4, CV_64F)) {}
	
};

void fdetectMatch(Mat&, Mat&, Mat&, calib_data&, Mat&, Mat&, Mat&);

void depthcomp(Mat&, Mat&, Mat&);
calib_data read_yaml2(const YAML::Node& , const YAML::Node& , const YAML::Node& );
calib_data read_yaml_kitti(const YAML::Node&);
void filter_matches(vector<DMatch>&, vector<DMatch>&);
void comp_depth(Mat&, Point2f&, Point3f&, double, Mat);
void disparity(Mat&, Mat&, Mat&);
void DrawTrajectory(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>>);


