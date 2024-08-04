#include "../include/visual_odometry/feature.hpp"

void posecomp(vector<KeyPoint>& kp1, vector<KeyPoint>& kp2, vector<DMatch>& g_matches, calib_data& calib1, calib_data& calib2, Mat& R, Mat& t){
    vector<Point2f> point1, point2;

    for(DMatch i:g_matches){
        point1.push_back(kp1[i.queryIdx].pt);
        point2.push_back(kp2[i.trainIdx].pt);
    }
    
    Mat E;
    //Mat E = findEssentialMat(point1, point2, calib1.cam, cv::RANSAC, 0.999, 1.0, 1000, noArray());
    cout<<"match size - "<<g_matches.size()<<endl;
    //cout<<"Essential Matrix - "<<E.size()<<endl;
    recoverPose(point1, point2, calib1.cam, calib1.dist, calib2.cam, calib2.dist, E, R, t,cv::RANSAC,0.999, 1.0, noArray());

    
}

void fdetectMatch(Mat& limg1, Mat& rimg1, Mat& limg2, Mat& k, Mat& R, Mat& t, Mat& match_img2){
    
    Mat ldesc, rdesc, ldesct;
    Mat limg, rimg, limgt;
    vector<KeyPoint> kpl, kpr, kplt;

    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(4);
    clahe->setTilesGridSize(Size(4,4));
    
    clahe->apply(limg1, limg);
    clahe->apply(rimg1, rimg);
    clahe->apply(limg2, limgt);
      
    //FEATURE DETECTOR

    Ptr<ORB> orb = ORB::create(1000, 1.2, 16, 20, 2, 2, ORB::HARRIS_SCORE, 20, 15);//
    auto start = std::chrono::high_resolution_clock::now();
    orb->detectAndCompute(limg, noArray(), kpl, ldesc);
    auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	cout << "The function took " << duration.count() << " milliseconds\n";
	orb->detectAndCompute(rimg, noArray(), kpr, rdesc);
    orb->detectAndCompute(limgt, noArray(), kplt, ldesct);

    // ldesc.convertTo(ldesc, CV_32F);
    // rdesc.convertTo(rdesc, CV_32F);
    // ldesct.convertTo(ldesct, CV_32F);

    //FEATURE MATCHER
    cout<<"---------------"<<endl;
    vector<DMatch> matches, matchest;
    vector<DMatch> good_matches, good_matchest;
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
	matcher->match(ldesc, rdesc, matches);
    matcher->match(ldesc, ldesct, matchest);
    
    filter_matches(matches, good_matches);
    filter_matches(matchest, good_matchest);
    
    // cout<<"good_matches : "<<good_matches.size()<<endl;
    // cout<<"good_matches_t : "<<good_matchest.size()<<endl;
    vector<Point3f> points3d;
    vector<Point2f> points2d;
    for(DMatch i:good_matches){
        for(DMatch j:good_matchest){
            if(i.queryIdx==j.queryIdx){
                Point3f d;
                comp_depth(kpl[i.queryIdx].pt, kpr[i.trainIdx].pt, d, 0.5707, k);
                if(d.x){
                    points3d.push_back(d);
                    points2d.push_back(kplt[j.trainIdx].pt);    
                }

            }
        }        
    }
    // cout<<"points3d : "<<points3d<<endl<<"points2d : "<<points2d<<endl;
    // cout<<"points 3d : "<<points3d.size()<<" points2d : "<<points2d.size()<<endl;
    if(points3d.size()>5){
        Mat r;
        solvePnPRansac(points3d, points2d, k, noArray(), r, t, false,100);
        Rodrigues(r,R);
    }
    
    // match_img2 = rimg.clone();
    drawMatches(limg, kpl, rimg, kpr, good_matches, match_img2, Scalar::all(-1),
 	Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
}

void filter_matches(vector<DMatch>& matches, vector<DMatch>& good_matches){
    auto min_max = minmax_element(matches.begin(), matches.end(),[](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
    // cout<<"min_dist - "<<min_max.first->distance<<endl;
    // cout<<"max_dist - "<<min_max.second->distance<<endl;
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;
    for(int i=0;i<matches.size();i++){
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

void comp_depth(Point2f& kp1, Point2f& kp2, Point3f& point3d, double b, Mat k){
        double d = abs(kp1.x - kp2.x);
        if(d!=0&&abs(kp1.y-kp2.y)<1){
            double d1 = (k.at<double>(0,0)*b)/d;        
            point3d.x = (d1*(kp1.x - k.at<double>(0,2))/k.at<double>(0,0));
            point3d.y = (d1*(kp1.y - k.at<double>(1,2))/k.at<double>(1,1));
            point3d.z = d1;
        }
}



