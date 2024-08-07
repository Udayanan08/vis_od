#include "../include/visual_odometry/feature.hpp"

void fdetectMatch(Mat& limg, Mat& rimg, Mat& limgt, calib_data& calib, Mat& R, Mat& t, Mat& match_img2){
    
    Mat ldesc, rdesc, ldesct;
    //Mat limg, rimg, limgt;
    // Mat disp;
    Mat disp(cv::Mat::zeros(limg.size().width,limg.size().height,CV_32F));
    vector<KeyPoint> kpl, kpr, kplt;

    // Ptr<CLAHE> clahe = createCLAHE();
    // clahe->setClipLimit(4);
    // clahe->setTilesGridSize(Size(4,4));
    
    // clahe->apply(limg1, limg);
    // clahe->apply(rimg1, rimg);
    // clahe->apply(limg2, limgt);
      
    //FEATURE DETECTOR
    disparity(limg, rimg, disp);
    Ptr<ORB> orb = ORB::create(2000, 1.2, 16, 20, 2, 2, ORB::HARRIS_SCORE, 20, 15);//
    // 
    orb->detectAndCompute(limg, noArray(), kpl, ldesc);

    orb->detectAndCompute(limgt, noArray(), kplt, ldesct);

    // ldesc.convertTo(ldesc, CV_32F);
    // rdesc.convertTo(rdesc, CV_32F);
    // ldesct.convertTo(ldesct, CV_32F);

    //FEATURE MATCHER
    vector<DMatch> matches, matchest;
    vector<DMatch> good_matches, good_matchest;
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
    matcher->match(ldesc, ldesct, matchest);
    
    filter_matches(matchest, good_matchest);
    
    vector<Point3f> points3d;
    vector<Point2f> points2d;
    for(DMatch i:good_matchest){
        Point3f d; 
        comp_depth(disp,kpl[i.queryIdx].pt, d, calib.b, calib.cam);
        if(d.x){
            points2d.push_back(kplt[i.trainIdx].pt);
            points3d.push_back(d);
        }
    }
    
    if(points3d.size()>5){
        Mat r;
        solvePnPRansac(points3d, points2d, calib.cam, noArray(), r, t, false,100);
        Rodrigues(r,R);
    }
    
    drawMatches(limg, kpl, rimg, kpr, good_matches, match_img2, Scalar::all(-1),
 	Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
}

void disparity(Mat& limg, Mat& rimg, Mat& disp){
    static Ptr<StereoSGBM> stereo = StereoSGBM::create(0,96,7,8*7*7,32*7*7); //(0,4*16,7,0,0,0,0,0,0,0,cv::StereoSGBM::MODE_SGBM_3WAY);
    Mat disp_sgbm;
    stereo->compute(limg,rimg, disp_sgbm);
    disp_sgbm.convertTo(disp, CV_32F, 1.0/16.0f);    
}
void comp_depth(Mat& disp,Point2f& kp1, Point3f& point3d, double b, Mat k){
    float d = disp.at<float>(kp1.y,kp1.x);
    double d1 = (k.at<double>(0,0)*b)/d;
    if(d1>=10){
        double fx = k.at<double>(0,0);
        double fy = k.at<double>(1,1);
        double cx = k.at<double>(0,2);
        double cy = k.at<double>(1,2);
        point3d.x = d1*(kp1.x - cx)/fx;
        point3d.y = d1*(kp1.y - cy)/fy;
        point3d.z = d1;
    }else{
        point3d.x = 0;
        point3d.y = 0;
        point3d.z = 0;
    }

}


void filter_matches(vector<DMatch>& matches, vector<DMatch>& good_matches){
    auto min_max = minmax_element(matches.begin(), matches.end(),[](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
    // cout<<"min_dist - "<<min_max.first->distance<<endl;
    // cout<<"max_dist - "<<min_max.second->distance<<endl;
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;
    for(unsigned int i=0;i<matches.size();i++){
        if(matches[i].distance <= max(2*min_dist,20.0)){
            good_matches.push_back(matches[i]);
        }
    }
}

calib_data read_yaml2(const YAML::Node& node1, const YAML::Node& node2, const YAML::Node& node3){
	int rows1 = node1["rows"].as<int>();
    int cols1 = node1["cols"].as<int>();
	int rows2 = node2["rows"].as<int>();
    int cols2 = node2["cols"].as<int>();
	int rows3 = node3["rows"].as<int>();
    int cols3 = node3["cols"].as<int>();

	
	std::vector<double> data1 = node1["data"].as<std::vector<double>>();
	std::vector<double> data2 = node2["data"].as<std::vector<double>>();
	std::vector<double> data3 = node3["data"].as<std::vector<double>>();


	calib_data calb;
	for (int i = 0; i < rows1; ++i) {
        for (int j = 0; j < cols1; ++j) {
            calb.cam.at<double>(i, j) = data1[i * cols1 + j];
        }
    }
	for (int i = 0; i < rows2; ++i) {
        for (int j = 0; j < cols2; ++j) {
            calb.dist.at<double>(i, j) = data2[i * cols2 + j];
        }
    }
	for (int i = 0; i < rows3; ++i) {
        for (int j = 0; j < cols3; ++j) {
            calb.proj.at<double>(i, j) = data3[i * cols3 + j];
        }
    }
	return calb;
}
calib_data read_yaml_kitti(const YAML::Node& node1){
	std::vector<double> data1 = node1["data"].as<std::vector<double>>();

    calib_data calib;
    
    calib.cam.at<double>(0,0)=data1[0];
    calib.cam.at<double>(0,1)=data1[1];
    calib.cam.at<double>(0,2)=data1[2];
    calib.cam.at<double>(1,0)=data1[4];
    calib.cam.at<double>(1,1)=data1[5];
    calib.cam.at<double>(1,2)=data1[6];
    calib.cam.at<double>(2,0)=data1[8];
    calib.cam.at<double>(2,1)=data1[9];
    calib.cam.at<double>(2,2)=data1[10];

    calib.proj.at<double>(0,0)=data1[0];
    calib.proj.at<double>(0,1)=data1[1];
    calib.proj.at<double>(0,2)=data1[2];
    calib.proj.at<double>(0,3)=data1[3];
    calib.proj.at<double>(1,0)=data1[4];
    calib.proj.at<double>(1,1)=data1[5];
    calib.proj.at<double>(1,2)=data1[6];
    calib.proj.at<double>(1,3)=data1[7];
    calib.proj.at<double>(2,0)=data1[8];
    calib.proj.at<double>(2,1)=data1[9];
    calib.proj.at<double>(2,2)=data1[10];
    calib.proj.at<double>(2,3)=data1[11];

	return calib;
}

