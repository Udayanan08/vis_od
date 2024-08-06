#include "../include/visual_odometry/vis_od.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;




void match_image(Mat& left_frame, Mat& right_frame, Mat& left_frame1, calib_data& projMat1, calib_data& projMat2, Mat& match_img2, String& file_name){
	// cv::cvtColor(left_frame, dist_gray_frame, cv::COLOR_BGR2GRAY);
	// cv::cvtColor(right_frame, dist_gray_frame2, cv::COLOR_BGR2GRAY);
	// cv::cvtColor(left_frame1, dist_gray_frame3, cv::COLOR_BGR2GRAY);

	// cv::undistort(left_frame, gray_frame, projMat1.cam, projMat1.dist, noArray());
	// cv::undistort(right_frame, gray_frame2, projMat2.cam, projMat2.dist, noArray());
	// cv::undistort(left_frame1, gray_frame3, projMat1.cam, projMat1.dist, noArray());

	vector<DMatch> good_matches;
	fdetectMatch(left_frame, right_frame, left_frame1, projMat1, R, t, match_img2);
	comp_path(R,t,trans_mat);
	ofstream file(file_name, std::ios::app);
	if(file.is_open()){
		trans_flat = trans_mat.reshape(1,1);
		for(int i=0; i<12;i++){
			file<<trans_flat.at<double>(0,i)<<" ";
		}
		file<<endl;
		file.close();
	}
	// cout<<"Transformation matrix : "<<endl<<trans_mat<<endl;
	// cout<<"Rotation matarix : "<<endl<<R<<endl;
	// cout<<"Translational matrix : "<<endl<<trans_mat<<endl;
	R.release();
	t.release();
}

void comp_path(Mat& R, Mat& t, Mat& trans_mat){
	Mat trans_new, trans_new_inv;
	trans_new = Mat::zeros(4,4,CV_64F);
	augment(R, t, trans_new);
	invert(trans_new,trans_new_inv);
	trans_mat = trans_mat*trans_new_inv;
}

void augment(Mat& R, Mat& t, Mat& trans_mat){
	trans_mat.at<double>(0,0)=R.at<double>(0,0);
	trans_mat.at<double>(0,1)=R.at<double>(0,1);
	trans_mat.at<double>(0,2)=R.at<double>(0,2);
	trans_mat.at<double>(0,3)=t.at<double>(0,0);
	trans_mat.at<double>(1,0)=R.at<double>(1,0);
	trans_mat.at<double>(1,1)=R.at<double>(1,1);
	trans_mat.at<double>(1,2)=R.at<double>(1,2);
	trans_mat.at<double>(1,3)=t.at<double>(0,1);
	trans_mat.at<double>(2,0)=R.at<double>(2,0);
	trans_mat.at<double>(2,1)=R.at<double>(2,1);
	trans_mat.at<double>(2,2)=R.at<double>(2,2);
	trans_mat.at<double>(2,3)=t.at<double>(0,2);
	trans_mat.at<double>(3,0)=0.0;
	trans_mat.at<double>(3,1)=0.0;
	trans_mat.at<double>(3,2)=0.0;
	trans_mat.at<double>(3,3)=1.0;

}


int main(int argc, char * argv[]){

	cv::String f_n = argv[1];
	cv::String file_name = f_n + ".txt";
	proj_ = Mat::zeros(3,4,CV_64F);
	string line;
	string item;
	ifstream config("/home/ud/ccodes/datasets/data_odometry_gray/dataset/sequences/00/calib.txt");
	vector<double> elements;

	while(std::getline(config, line)){
		stringstream ss(line);
		ss>>item;
		while(ss>>item){
			elements.push_back(stod(item));
		}
		
		proj_.at<double>(0,0)=elements[0];
		proj_.at<double>(0,1)=elements[1];
		proj_.at<double>(0,2)=elements[2];
		proj_.at<double>(0,3)=elements[3];
		proj_.at<double>(1,0)=elements[4];
		proj_.at<double>(1,1)=elements[5];
		proj_.at<double>(1,2)=elements[6];
		proj_.at<double>(1,3)=elements[7];
		proj_.at<double>(2,0)=elements[8];
		proj_.at<double>(2,1)=elements[9];
		proj_.at<double>(2,2)=elements[10];
		proj_.at<double>(2,3)=elements[11];
	}


	cout<<proj_<<endl<<endl<<endl<<endl;
	// Reading the projection matrix
	YAML::Node config1 = YAML::LoadFile("../config/kitti_gray.yaml");
	YAML::Node config2 = YAML::LoadFile("../config/kitti_gray.yaml");
	projMat1 = read_yaml_kitti(config1["P0"]);
	projMat2 = read_yaml_kitti(config2["P1"]);

	//read image from the folder
	const cv::String dir = "/home/ud/ccodes/datasets/data_odometry_gray/dataset/sequences/"+f_n+"/image_0/";
	const cv::String dir1 = "/home/ud/ccodes/datasets/data_odometry_gray/dataset/sequences/"+f_n+"/image_1/";

	cv::utils::fs::glob(dir,"*.png", dir_vec,false,false);
    cv::utils::fs::glob(dir1,"*.png", dir_vec1,false,false);

	imgt0 = cv::imread(dir_vec[count_], cv::IMREAD_GRAYSCALE);
	count_var = dir_vec.size()-1;
	trans_flat = Mat::zeros(1,12,CV_64F);
	R_trans = Mat::eye(3,3,CV_64F);
	t_trans = Mat::zeros(3,1,CV_64F);
	trans_mat = Mat::eye(4,4,CV_64F);

	//writing the first line of output file with origin
	ofstream file(file_name);
	if(file.is_open()){
		trans_flat = trans_mat.reshape(1,1);
		for(int i=0; i<12;i++){
			file<<trans_flat.at<double>(0,i)<<" ";
		}
		file<<endl;
		file.close();
	}
	// Decomposing the projection matrix to get cam matrix and baseline
	Mat k,R_,t_;
	decomposeProjectionMatrix(projMat1.proj,k,R_,t_);
	double b1 = t_.at<double>(0,0)/t_.at<double>(3,0);
	projMat1.cam = k;
	decomposeProjectionMatrix(projMat2.proj,k,R_,t_);
	double b2 = t_.at<double>(0,0)/t_.at<double>(3,0);
	projMat2.cam = k;
	projMat1.b = b2-b1;

	while(count_<count_var){
		img0 = imgt0.clone();
		imgt0 = cv::imread(dir_vec[count_+1], cv::IMREAD_GRAYSCALE);
		img1 = cv::imread(dir_vec1[count_], cv::IMREAD_GRAYSCALE);
		auto start = std::chrono::high_resolution_clock::now();
		match_image(img0, img1, imgt0, projMat1, projMat2, match_img2,file_name);
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		std::cout << "\33[A";
		std::cout << "\33[2K";
		cout<<"\rcount : "<<"\033[0;32m"<<count_<<"\033[0m"<<flush;
		cout << "\nTook " << "\033[0;32m"<<duration.count()<<"\033[0m" << " milliseconds to compute "<<flush;
		count_++;
	}
	cout<<endl;
	return 0;	
}


