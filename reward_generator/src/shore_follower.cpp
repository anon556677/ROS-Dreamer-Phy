#include <math.h>
#include <iostream>
#include <cstdlib>
#include <iterator>
#include <set>
#include <numeric>
#include <functional>
#include "reward_generator/shore_follower.h"

//#define DEBUG
//#define SHOW_IMG

ShoreFollower::ShoreFollower() : nh_("~"), it_(nh_) {
    // from launch
  	std::string laser_frame("front_laser");
  	nh_.param("dist_to_shore", dist_to_shore_, 10.0f);
  	nh_.param("laser_frame", lzr_frame_, laser_frame);
  	nh_.param("local_grid_front_size", local_grid_front_size_, 10.0f);
  	nh_.param("safety_area_cost", safety_cost_, 1000);
  	nh_.param("max_gap", max_gap_, 0.5f);
  	nh_.param("publish_occupancy_grid", publish_occupancy_grid_, true);
  	nh_.param("blur_kernel_size", blur_kernel_size_, 2.0f);
  	nh_.param("pose_coeff", pose_coeff_, 2.5f);
  	nh_.param("vel_ceoff", vel_coeff_, 1.2f);
  	nh_.param("vel_target_", vel_target_, 1.0f);
  	nh_.param("backward_coeff", backward_coeff_, -0.625f);
  	nh_.param("rate", rate_, 12);

    //Define map parameters
    safe_to_shore_ = std::max(dist_to_shore_*0.4, 2.0);
    local_grid_front_size_ = std::min(dist_to_shore_*1.2, 10.0);
    local_grid_rear_size_ = std::min(dist_to_shore_/5.0, 1.0);
    local_grid_side_size_ = dist_to_shore_*1.2;
    local_grid_res_ = 1.0/dist_to_shore_;
	
    //convert parameters in meters to pixels
    //image size
    int hsize = (int) (local_grid_front_size_+local_grid_rear_size_)/local_grid_res_;
    int wsize = (int) (local_grid_side_size_)*2/local_grid_res_;
    size_ = std::max(hsize, wsize);
    //"crop" the ranges for the costmap
    int minx = std::max(0,(int)(size_/2-local_grid_rear_size_/local_grid_res_));
    int maxx = std::min((int)(size_/2 + local_grid_front_size_/local_grid_res_), size_);
    int miny = std::max(0,(int)(size_/2-local_grid_side_size_/local_grid_res_));
    int maxy = std::min((int)(size_/2+local_grid_side_size_/local_grid_res_), size_);
   
    croph_ = cv::Range(minx, maxx);
    cropw_ = cv::Range(miny, maxy);
    x_offset_ = (int) local_grid_side_size_ / local_grid_res_;
    y_offset_ = (int) local_grid_rear_size_ / local_grid_res_;
    
    //initialize the constants used to calculate the costs	
    kernel_size_ = (int) dist_to_shore_*2/local_grid_res_;
  	safe_kernel_size_ = (int) safe_to_shore_*2/local_grid_res_;
    max_gap_ = max_gap_/local_grid_res_;
    
    //initialize the size of the Gaussian kernel size (to smooth the map)
    int k{(blur_kernel_size_/local_grid_res_)};
  	if((k & 1) == 0){//make it odd
		  k ++;
	  }
    blur_ksize_ = cv::Size(k,k);
	
    initializeMap();
	
    //create subscribers and publishers
    lzr_sub_ = nh_.subscribe("laser",1,&ShoreFollower::LZRCallback,this);
    refresh_sub_ = nh_.subscribe("refresh",1,&ShoreFollower::refreshCallback,this);
    done_sub_ = nh_.subscribe("done",1,&ShoreFollower::doneCallback,this);
    odom_sub_ = nh_.subscribe("odom",1,&ShoreFollower::odomCallback,this);
    reward_pub_ = nh_.advertise<std_msgs::Float32>("reward",1,true);
    cp_reward_pub_ = nh_.advertise<reward_generator::follower_reward>("composite_reward",1,true);
    image_pub_ = it_.advertise("DreamersView", 1); 
}

void ShoreFollower::LZRCallback(const sensor_msgs::LaserScanConstPtr& data) {
    if (compute_) {
        refreshMap();
        buffer_.clear();
        inner_shores_.clear();
        points_.clear();
        addToBuffer(data);
        filterPoints(buffer_,points_);
        genContourMap();
        genCostMap();
#ifdef SHOW_IMG
        try {
            cv::imshow("cost_map", cost_map_);
            cv::imshow("thin_map", thin_map_);
            cv::imshow("n_cost_map", n_cost_map_);
            cv::imshow("map", map_);
            cv::waitKey(5);
        } catch (...) {
            std::cout << "error in imshow\n";
        }
#endif
    }
}

void ShoreFollower::odomCallback(const nav_msgs::OdometryConstPtr& data) {
    // Quaternion Projection
    float siny_cosp = 2 * (data->pose.pose.orientation.w * data->pose.pose.orientation.z + data->pose.pose.orientation.x * data->pose.pose.orientation.y);
    float cosy_cosp = 1 - 2 * (data->pose.pose.orientation.y * data->pose.pose.orientation.y + data->pose.pose.orientation.z * data->pose.pose.orientation.z);
    float yaw = std::atan2(siny_cosp, cosy_cosp);
    // Velocity in the robot frame
    float vlin = std::cos(yaw)*data->twist.twist.linear.x + std::sin(yaw)*data->twist.twist.linear.y;
#ifdef SHOW_REWARD_INFO
    ROS_INFO("vlin: %.3f",vlin);
#endif
    if (vlin > -0.25) {
        vel_reward_ = 1 - fabs(vel_target_ - vlin)/vel_target_;
    } else {
        vel_reward_ = backward_coeff_;
    }
}

void ShoreFollower::doneCallback(const std_msgs::BoolConstPtr& data) {
    compute_ = false;
    ROS_INFO("Pausing laser processing");
}

void ShoreFollower::refreshCallback(const rl_server::EpisodeConstPtr& data) {
    compute_ = true;
    ROS_INFO("Resuming laser processing...");
}

void ShoreFollower::initializeMap() {
    // Instantiate maps
  	if((kernel_size_ & 1) == 0){
  		kernel_size_ ++;
  	}
  	if((safe_kernel_size_ & 1) == 0){
  		safe_kernel_size_ ++;
  	}
  	dreamers_map_ = cv::Mat::zeros(size_, size_, CV_8UC3);      // The map filled with circles
    dreamers_view_ = cv::Mat::zeros(64, 64, CV_8UC3);
  	blue_map_ = cv::Mat::zeros(size_, size_, CV_8UC3); // A blue map for dreamer to see
    blue_map_.setTo(cv::Scalar(255,0,0));             
  	
    map_ = cv::Mat::zeros(size_, size_, CV_8UC1);      // The map filled with circles
  	zero_map_ = cv::Mat::zeros(size_, size_, CV_8UC1); // A map full of zeros
  	one_map_ = cv::Mat::ones(size_, size_, CV_8UC1);   // A map full of ones
  	thin_map_ = cv::Mat::ones(size_, size_, CV_8UC1);  // The contours of map_
  
    // A matrix with a circle inside
    // kernel_ is used to create the low cost path
    kernel_ = cv::Mat::zeros(kernel_size_, kernel_size_, CV_8UC1); 
  	hskernel_ = (int) kernel_size_/2.0;
  	cv::Point center_coords((int) kernel_size_/2.0 + 1, (int) kernel_size_/2.0 + 1);
  	cv::circle(kernel_, center_coords, (int) dist_to_shore_/local_grid_res_, cv::Scalar(255), -1);
    //safety_ is used to create the high cost safety area
  	safety_ = cv::Mat::zeros(safe_kernel_size_, safe_kernel_size_, CV_8UC1); 
  	hssafety_ = (int) safe_kernel_size_/2.0;
  	center_coords = cv::Point((int) safe_kernel_size_/2.0 + 1, (int) safe_kernel_size_/2.0 + 1);
  	cv::circle(safety_, center_coords, (int) safe_to_shore_/local_grid_res_, cv::Scalar(255), -1);
}

void ShoreFollower::refreshMap() {
    // Refresh the maps as a new laser comes in
	  zero_map_.copyTo(map_);
	  blue_map_.copyTo(dreamers_map_);
	  one_map_.copyTo(thin_map_);
}

void ShoreFollower::genContourMap(){
    // Computes the circle map
    for (const std::array<int, 2>& point : points_) {
  	  	setMapBlockValue(point[0],point[1],kernel_, kernel_size_, map_, cv::Scalar(255));
  	}
    // Compute the contours of the cicle map
  	std::vector<std::vector<cv::Point> > contours;
  	std::vector<cv::Vec4i> hierarchy;
  	cv::findContours(map_, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
  	cv::drawContours(thin_map_, contours, -1, cv::Scalar(0), 5);
  	cv::drawContours(dreamers_map_, contours, -1, cv::Scalar(0,0,0), 7);
  	thin_map_(cv::Range(0,5), cv::Range(0,size_)) = 255;
  	thin_map_(cv::Range(size_-5,size_), cv::Range(0,size_)) = 255;
  	thin_map_(cv::Range(0,size_), cv::Range(0,5)) = 255;
  	thin_map_(cv::Range(0,size_), cv::Range(size_-5,size_)) = 255;
}
                    
void ShoreFollower::setMapBlockValue(int cx, int cy, cv::Mat& the_kernel, int ksize, cv::Mat& the_map, cv::Scalar val) {
    /*
    Makes a circle inside the map at coordinate cx cy.
    Instead of using cv::circle which is slow we use a custom function
    which copies the pixels of a small image containing a circle at the
    right spot. It checks and corrects for all edge cases. 
    */
    // It looks ugly it is ugly, but its fast
    //TODO where does this 0,0 point comes from ??
    if ((cx == 0) && (cy == 0)) return;
    int bxl;
    int bxh;
    int byl;
    int byh;
    int kxl;
    int kxh;
    int kyl;
    int kyh;
  	int hsk{ksize /2};
    
    bxl = cx-hsk-1;
  	kxl = std::max(0, -bxl);
  	bxl = std::max(bxl, 0);
  	bxl = std::min(bxl, size_-1);
  	bxh = std::max(cx+hsk,0);
  	bxh = std::min(bxh, size_-1);
  	byl = cy-hsk-1;
  	kyl = std::max(0, - byl);
  	byl = std::max(byl,0);
  	byl = std::min(byl, size_-1);
  	byh = std::min(cy+hsk, size_-1);
  	byh = std::max(byh, 0);
  
  	kxl = std::min(ksize, kxl);
  	kxh = bxh - bxl + kxl;
  	kxh = std::min(kxh,ksize);
  	kyl = std::min(ksize, kyl);
  	kyh = byh - byl + kyl;
  	kyh = std::min(kyh,ksize);
    the_map(cv::Range(bxl, bxh),cv::Range(byl, byh)).setTo(val, the_kernel(cv::Range(kxl,kxh),cv::Range(kyl,kyh)));
}

void ShoreFollower::genCostMap(){
	  cv::distanceTransform(thin_map_, // source image
		cost_map_,              // distance image
		cv::DIST_L2,            // distance type
		cv::DIST_MASK_3);       // mask size
     
    //and now, add the safety area (around the detected shores):
    for (const std::array<int, 2>& point : points_) {
		    setMapBlockValue(point[0],point[1],safety_, safe_kernel_size_,cost_map_,cv::Scalar(safety_cost_));
		    setMapBlockValue(point[0],point[1],safety_, safe_kernel_size_,dreamers_map_,cv::Scalar(0,0,255));
	  }
    //blur the image
    cv::GaussianBlur(cost_map_, smoothed_map_, blur_ksize_, 0, 0, cv::BORDER_DEFAULT);
    //crop it
    smoothed_map_(croph_,cropw_).copyTo(cropped_map_);
    cropped_map_.setTo(0,cropped_map_ < 5);
    cv::resize(dreamers_map_(croph_,cropw_), dreamers_view_, cv::Size(64,64), 0, 0, cv::INTER_NEAREST);
}

// Publish occupancy grid for rviz
void ShoreFollower::publishOccupancyGrid() {
  	occupancy_grid_.header.stamp = ros::Time::now();
    cv::Mat_<float> tmp;
    cv::threshold(cropped_map_, tmp, 100, 100, cv::THRESH_TRUNC);
    tmp.convertTo(n_cost_map_, CV_8U, 1.0);
    occupancy_grid_.data = n_cost_map_.reshape(1,1);
	  cmap_pub_.publish(occupancy_grid_);
}

//publish the custom LaserCostMap
void ShoreFollower::publishReward() {
    reward_.data = pose_reward_ + vel_reward_;
#ifdef SHOW_REWARD_INFO
    ROS_INFO("pose reward %.3f", pose_reward_);
    ROS_INFO("vel reward %.3f", vel_reward_);
#endif
    composite_reward_.distance_reward  = pose_reward_;
    composite_reward_.velocity_reward = vel_reward_;
    reward_pub_.publish(reward_);
    cp_reward_pub_.publish(composite_reward_);
}    


void ShoreFollower::addToBuffer(const sensor_msgs::LaserScanConstPtr & data) {
    // Takes the laser outputs and store them into a buffer.
    float xf, yf;
	  int x, y;
    unsigned int N = static_cast<int>((size_/2)/safe_kernel_size_*2);
    float step = safe_to_shore_;

    float min_dist;
    std::vector<float>::iterator iter;
    std::vector<float> tmp;
    tmp = data->ranges;
    iter = std::min_element(tmp.begin(), tmp.end());
    //iter = std::min_element(data->ranges.begin(), data->ranges.end());
    min_dist =  fabs(dist_to_shore_ - *iter);
    pose_reward_ = std::max(-20.0, (1. - min_dist*min_dist*0.5)*pose_coeff_);
#ifdef SHOW_REWARD_INFO
    ROS_INFO("pose dist %.3f", *iter);
#endif
    //calculate the coordinates of the current laser points in the boat referential, 
    //in the image space
    for (unsigned int i=0; i < data->ranges.size(); i ++) {
        //Exclude laser points associated with no measurements
        //if ((data->ranges[i] < data->range_max) && ((data->ranges[i] < size_) && (data->ranges[i] > 0.5))) {
        if ((data->ranges[i] < data->range_max) && (data->ranges[i] > 0.5)) {
            xf = std::cos(data->angle_min + i*(data->angle_increment))*(data->ranges[i]);
            yf = std::sin(data->angle_min + i*(data->angle_increment))*(data->ranges[i]);
            // Compensates for image format.
            /* Originaly           Compensation
                +------             -------
                |      |           |       |
                |      |           |   +   |
                |      |           |       |
                 ------             -------
            Boat coordinates marked as :   +
            Note it may not make sense to center over y as we don't want to go backward (x for opencv).
            */
            buffer_.push_back(std::array<int,2>{
                    static_cast<int>(xf/local_grid_res_ + 0.5*size_),
                    static_cast<int>(yf/local_grid_res_ + 0.5*size_)});
        }
	}
}

void ShoreFollower::filterPoints(PointsVector& in_v, PointsVector& out_v) {
//remove duplicates from one vector of points and store the result in another
#ifdef DEBUG
    std::cout <<"start filtering\n"
        << "in size is: " << in_v.size() << std::endl;
#endif
    std::set<std::array<int,2>, std::less<std::array<int,2>>> buff_set{};
    for (const auto& elem : in_v) {
        buff_set.insert(elem);
    }
    out_v.resize(in_v.size());
    std::copy(buff_set.cbegin(), buff_set.cend(),out_v.begin());
#ifdef DEBUG
    std::cout << "data is filtered, filtered size is: " << out_v.size() << std::endl;
#endif
}


void ShoreFollower::interpolatePoints(PointsVector& in_v, PointsVector& out_v) {
    /*
    Calculate the distance between one point and the following
         if the distance is too large, add points in between
         (this might not make sense on real data)
    */
    int N;
    int xdiff, ydiff;
    double xy_dist;
    int xtmp, ytmp;
    std::array<int,2> tmp_point;
    float xstep, ystep;
    size_t i{0};
#ifdef DEBUG
    std::cout <<"start interpolating\n"
        << "in size is: " << in_v.size() << std::endl;
#endif
    
    out_v.clear();
    while (i < (in_v.size()-1)) {
        N = 0;
        xy_dist = calcDist(in_v, i+1, i);
        if (xy_dist < max_gap_) {
            out_v.push_back(in_v.at(i));
            ++i;
        } else {
            N = static_cast<int>((xy_dist - max_gap_)/max_gap_);
#ifdef DEBUG
            std::cout << "interpolating " << N << " points between "
                << "xj: " << in_v.at(i+1)[0] << ",yj:" << in_v.at(i+1)[1] 
                << "  xi: " << in_v.at(i)[0] << ",yi:" << in_v.at(i)[1] << std::endl;                ;
#endif
            xdiff = in_v.at(i+1)[0] - in_v.at(i)[0];
            ydiff = in_v.at(i+1)[1] - in_v.at(i)[1];
            xstep = static_cast<float>(xdiff)/N;
            ystep = static_cast<float>(ydiff)/N;
            for (unsigned int j{1}; j <= N; j++) {
                xtmp = static_cast<int>(in_v.at(i)[0] + xstep*j);     
                ytmp = static_cast<int>(in_v.at(i)[1] + ystep*j);
                tmp_point = {xtmp, ytmp};
#ifdef DEBUG
                std::cout << "  Add "<< xtmp << ", " << ytmp << std::endl;
#endif
                out_v.push_back(tmp_point);
            }
            ++i;
        }
    }
#ifdef DEBUG
    std::cout << "before interpolation, in_v size: " << in_v.size() 
        << " after interpolation, out_v size: " << out_v.size() << std::endl;
#endif
}

//utility function to calculate the distance between two points
double ShoreFollower::calcDist(const PointsVector& pts, size_t i, size_t j){
    int xdiff{pts.at(j)[0] - pts.at(i)[0]};
    int ydiff{pts.at(j)[1] - pts.at(i)[1]};
    return std::sqrt(xdiff*xdiff + ydiff*ydiff);
}

void ShoreFollower::run() {
    ros::Rate r(rate_);
#ifdef SHOW_REWARD_INFO
    cv::namedWindow("Dreamer's View 64x64",cv::WINDOW_NORMAL);
    cv::resizeWindow("Dreamer's View 64x64", 600,600);
#endif
    r.sleep();
    std_msgs::Header h;

    while (ros::ok()) {
        ros::spinOnce();
        if (compute_) {
            h.stamp = ros::Time::now();
            sensor_msgs::ImagePtr msg = cv_bridge::CvImage(h, "bgr8", dreamers_view_).toImageMsg();
            publishReward();
            image_pub_.publish(msg);
#ifdef SHOW_REWARD_INFO
            cv::imshow("Dreamer's View 64x64", img_net_);
            cv::waitKey(5);
#endif
        }
        r.sleep();
    }
}


int main(int argc, char * argv[]) {
	ros::init(argc,argv,"reward_generator");
	ShoreFollower SH;
    SH.run();
}

