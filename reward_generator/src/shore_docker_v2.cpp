#include <math.h>
#include <iostream>
#include <cstdlib>
#include <iterator>
#include <set>
#include <numeric>
#include <functional>
#include "reward_generator/shore_docker_v2.h"

//#define DEBUG
//#define SHOW_IMG
//#define SHOW_REWARD_INFO

ShoreDocker::ShoreDocker() : nh_("~"), it_(nh_) {
    // from launch
    std::string laser_frame("front_laser");
    nh_.param("frame_size", frame_size_, 64);
    nh_.param("frame_padding", frame_padding_, 15);
    nh_.param("laser_frame", lzr_frame_, laser_frame);
    nh_.param("dist_to_boey", dist_to_boey, 14.0f);
    nh_.param("front_dist", front_dist_, 14.0f);
    nh_.param("rear_dist", rear_dist_, 2.0f);
    nh_.param("side_dist", side_dist_, 8.0f);
    nh_.param("dist_to_boey", dist_to_boey_, 2.0f);
    nh_.param("local_grid_res", grid_res_, 0.25f);
    nh_.param("px_coeff", px_coeff_, 1.f);
    nh_.param("py_ceoff", py_coeff_, 1.f);
    nh_.param("yaw_coeff", yaw_coeff_, 1.f);
    nh_.param("proximity_coeff", proximity_coeff_, 10.0f);

    nh_.param("rate", rate_, 12);
	
    //convert parameters in meters to pixels
    grid_res_ = (front_dist_ + rear_dist_)/frame_size_;
    
    size_ = frame_size_ + frame_padding_*2;

    y_offset_ = (int) side_dist_ / grid_res_ + frame_padding_;
    x_offset_ = (int) rear_dist_ / grid_res_ + frame_padding_;
    initializeFrame();
    croph_ = cv::Range(frame_padding_, frame_padding_+frame_size_);
    cropw_ = cv::Range(frame_padding_, frame_padding_+frame_size_);

    RED = cv::Vec3b(0, 0, 255);
    GREEN = cv::Vec3b(0, 255, 0);

    //instantiate boey pose to 0
    boey_yaw = 0;
    boey_px = 0;
    boey_py = 0;

	compute_ = true;

    //create subscribers and publishersx_offset
    lzr_sub_ = nh_.subscribe("laser",1,&ShoreDocker::LZRCallback,this);
    refresh_sub_ = nh_.subscribe("refresh",1,&ShoreDocker::refreshCallback,this);
    done_sub_ = nh_.subscribe("done",1,&ShoreDocker::doneCallback,this);
    odom_robot_sub_ = nh_.subscribe("odom_robot",1,&ShoreDocker::odomRobotCallback,this);
    odom_boey_sub_ = nh_.subscribe("odom_boey",1,&ShoreDocker::odomBoeyCallback,this);
    reward_pub_ = nh_.advertise<std_msgs::Float32>("reward",1,true);
    cp_reward_pub_ = nh_.advertise<reward_generator::docker_reward>("composite_reward",1,true);
    image_pub_ = it_.advertise("DreamersView", 1);
}

void ShoreDocker::LZRCallback(const sensor_msgs::LaserScanConstPtr& data) {
    if (compute_) {
        refreshFrame();
        addToBuffer(data);
        filterPoints(buffer_,points_);
        drawShore();
        drawBoey();
#ifdef SHOW_IMG
        try {
            cv::imshow("frame", frame_);
            cv::waitKey(5);
        } catch (...) {
            std::cout << "error in imshow\n";
        }
#endif
    }
}

void ShoreDocker::odomBoeyCallback(const nav_msgs::OdometryConstPtr& data) {
    float siny_cosp = 2 * (data->pose.pose.orientation.w * data->pose.pose.orientation.z + data->pose.pose.orientation.x * data->pose.pose.orientation.y);
    float cosy_cosp = 1 - 2 * (data->pose.pose.orientation.y * data->pose.pose.orientation.y + data->pose.pose.orientation.z * data->pose.pose.orientation.z);
    boey_yaw = std::atan2(siny_cosp, cosy_cosp);
    boey_px = data->pose.pose.position.x;
    boey_py = data->pose.pose.position.y;
}

void ShoreDocker::odomRobotCallback(const nav_msgs::OdometryConstPtr& data) {
    // Quaternion Projection
    float siny_cosp = 2 * (data->pose.pose.orientation.w * data->pose.pose.orientation.z + data->pose.pose.orientation.x * data->pose.pose.orientation.y);
    float cosy_cosp = 1 - 2 * (data->pose.pose.orientation.y * data->pose.pose.orientation.y + data->pose.pose.orientation.z * data->pose.pose.orientation.z);
    // Reward Calculation
    float yaw = std::atan2(siny_cosp, cosy_cosp);
    float px = data->pose.pose.position.x;
    float py = data->pose.pose.position.y;
    // Delta X Y: project the USV px and py in the boey frame
    float px_bf = px - boey_px;
    float py_bf = py - boey_py;
    float dx = cos(-boey_yaw)*px_bf - sin(-boey_yaw)*py_bf;
    float dy = sin(-boey_yaw)*px_bf + cos(-boey_yaw)*py_bf;
    pose_x_reward_ = - fabs(dx - dist_to_boey);
    pose_y_reward_ = - fabs(dy);

    // Delta theta: boey yaw + pi
    float yaw2boey = std::atan2(py_bf,px_bf);
    yaw_reward_ = - fabs(remainder(yaw2boey - yaw, 2*M_PI));

    // Projection of the boey in the USV frame
    float delta_x = boey_px - px;
    float delta_y = boey_py - py;
    boey_px_uf = delta_x*std::cos(-yaw) - delta_y*std::sin(-yaw);
    boey_py_uf = delta_x*std::sin(-yaw) + delta_y*std::cos(-yaw);
    boey_yaw_uf = boey_yaw - yaw; 
}

void ShoreDocker::doneCallback(const std_msgs::BoolConstPtr& data) {
    compute_ = false;
    ROS_INFO("Pausing laser processing");
}

void ShoreDocker::refreshCallback(const rl_server::EpisodeConstPtr& data) {
    compute_ = true;
    ROS_INFO("Resuming laser processing...");
}

void ShoreDocker::precomputeGaussian(cv::Mat_<uint8_t>& img, double sigma, int size){
    int hs = (size - 1)/2;
    double factor,r,s = 2.0*sigma*sigma;
    for (int x = -hs; x <= hs; x++) {
        for (int y = -hs; y <= hs; y++){
            r = std::sqrt(x*x + y*y);
            factor = (std::exp(-(r*r)/s));
            img(x+hs,y+hs) = int(factor*255);
        }
    }
}

void ShoreDocker::drawGaussian(cv::Mat_<cv::Vec3b>& img, double px, double py, double sigma, int size, int channel){
    int hs = (size - 1)/2;
    double factor,r,s = 2.0*sigma*sigma;
    int start_x = int (px - hs);
    int start_y = int (py - hs);
    for (int x = start_x; x <= start_x + size; x++) {
        if ((x >= 0) && (x < size_)) {
            for (int y = start_y; y <= start_y + size; y++){
                if ((y >= 0) && (y < size_)) {
                    r = std::sqrt((x-px)*(x-px) + (y-py)*(y-py));
                    factor = (std::exp(-(r*r)/s));
                    img(x,y)[channel] = int(factor*255);
                }
            }
        }
    }
}

void ShoreDocker::initializeFrame() {
    double kernel_size_ = 11;
    // Instantiate frames
    empty_frame_ = cv::Mat::zeros(size_, size_, CV_8UC1); // Reset
    frame_sc_ = cv::Mat::zeros(size_, size_, CV_8UC1); // A black frame for dreamer to see
    frame_ = cv::Mat::zeros(size_, size_, CV_8UC3); // A black frame for dreamer to see
    dreamers_view_ = cv::Mat::zeros(frame_size_, frame_size_, CV_8UC3); //cropped
    
    // Instantiate kernels
    //kernel_ = cv::Mat::zeros(kernel_size_, kernel_size_, CV_8UC3);
    gaussian_kernel_ = cv::Mat::zeros(kernel_size_, kernel_size_, CV_8UC1);
    precomputeGaussian(gaussian_kernel_, 2.8, kernel_size_);
  	//cv::Point center_coords((int) kernel_size_/2.0 + 1, (int) kernel_size_/2.0 + 1);
  	//cv::circle(kernel_, center_coords, 3, cv::Scalar(255,255,255), -1);
}

void ShoreDocker::refreshFrame() {
    // Refresh the maps as a new laser comes in
    frame_sc_ = 0;
    buffer_.clear();
    points_.clear();
}

void ShoreDocker::publishReward() {
#ifdef SHOW_REWARD_INFO
    ROS_INFO("pose x reward %.3f", pose_x_reward_);
    ROS_INFO("pose y reward %.3f", pose_y_reward_);
    ROS_INFO("yaw reward %.3f", yaw_reward_);
    ROS_INFO("proximity penalty %.3f", proximity_penalty_);
#endif
    reward_.data = pose_x_reward_*px_coeff_ +
                   pose_y_reward_*py_coeff_ +
                   yaw_reward_*yaw_coeff_ -
                   proximity_penalty_*proximity_coeff_;
    composite_reward_.delta_x_reward = pose_x_reward_*px_coeff_;
    composite_reward_.delta_y_reward = pose_y_reward_*py_coeff_;
    composite_reward_.delta_yaw_reward = yaw_reward_*yaw_coeff_; 
    composite_reward_.proximity_penalty = proximity_penalty_*proximity_coeff_;
    reward_pub_.publish(reward_);
    cp_reward_pub_.publish(composite_reward_);
}    

void ShoreDocker::addToBuffer(const sensor_msgs::LaserScanConstPtr & data) {
    // Takes the laser outputs and store them into a buffer.
    float xf, yf, ref;
    //calculate the coordinates of the current laser points in the boat referential 
    proximity_penalty_ = 0;
    for (unsigned int i=0; i < data->ranges.size(); i ++) {
        //Exclude laser points associated with no measurements
        //if ((data->ranges[i] < data->range_max) && ((data->ranges[i] < size_) && (data->ranges[i] > 0.5))) {
        if ((data->ranges[i] < data->range_max) && (data->ranges[i] > 0.5)) {
            if (data->ranges[i] < 0.75) {
                proximity_penalty_ = 1;
            }
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
            buffer_.push_back(std::array<int,3>{
                    static_cast<int>(xf/grid_res_ + x_offset_),
                    static_cast<int>(-yf/grid_res_ + y_offset_),
                    static_cast<int>(data->intensities[i]/200)});
        }
    }
}

void ShoreDocker::setMaxBlockValue(int cx, int cy, cv::Mat_<uint8_t>& the_kernel, int ksize, cv::Mat_<uint8_t>& the_map) {
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
    the_map(cv::Range(bxl, bxh),cv::Range(byl, byh)) = cv::max(the_kernel(cv::Range(kxl,kxh),cv::Range(kyl,kyh)),the_map(cv::Range(bxl, bxh),cv::Range(byl, byh)));
}

void ShoreDocker::filterPoints(PointsVector& in_v, PointsVector& out_v) {
//remove duplicates from one vector of points and store the result in another
#ifdef DEBUG
    std::cout <<"start filtering\n"
        << "in size is: " << in_v.size() << std::endl;
#endif
    std::set<std::array<int,3>, std::less<std::array<int,3>>> buff_set{};
    for (const auto& elem : in_v) {
        buff_set.insert(elem);
    }
    out_v.resize(in_v.size());
    std::copy(buff_set.cbegin(), buff_set.cend(),out_v.begin());
#ifdef DEBUG
    std::cout << "data is filtered, filtered size is: " << out_v.size() << std::endl;
#endif
}

void ShoreDocker::drawShore() {
    for (unsigned int i=0; i < points_.size(); i ++) {
        if ((points_[i][0] < size_) && (points_[i][0] > 0)) {
            if ((points_[i][1] < size_) && (points_[i][1] > 0)) {
                if (points_[i][2] == 0){
                     setMaxBlockValue(points_[i][1], points_[i][0], gaussian_kernel_, 11, frame_sc_);
                }
            }
        }
    }
    rgb_[0] = frame_sc_;
    rgb_[1] = frame_sc_;
    rgb_[2] = frame_sc_;
    cv::merge(rgb_,3, frame_);
}

void ShoreDocker::drawBoey() {
    double bc_px = boey_px_uf/grid_res_ + x_offset_;
    double bc_py = -boey_py_uf/grid_res_ + y_offset_;
    double bl_px = (boey_px_uf + std::cos(boey_yaw_uf + M_PI/2)*0.5)/grid_res_ + x_offset_;
    double bl_py = -(boey_py_uf + std::sin(boey_yaw_uf + M_PI/2)*0.5)/grid_res_ + y_offset_;
    double br_px = (boey_px_uf + std::cos(boey_yaw_uf - M_PI/2)*0.5)/grid_res_ + x_offset_;
    double br_py = -(boey_py_uf + std::sin(boey_yaw_uf - M_PI/2)*0.5)/grid_res_ + y_offset_;
    double sp_px = (boey_px_uf + std::cos(boey_yaw_uf)*dist_to_boey_)/grid_res_ + x_offset_;
    double sp_py = -(boey_py_uf + std::sin(boey_yaw_uf)*dist_to_boey_)/grid_res_ + y_offset_;

    //left
    drawGaussian(frame_, bl_py, bl_px, 2, 7, 2);
    //right
    drawGaussian(frame_, br_py, br_px, 2, 7, 1);
    //center
    drawGaussian(frame_, bc_py, bc_px, 2, 7, 0);
    //USV
    drawGaussian(frame_, y_offset_, x_offset_, 1.0, 5, 2);
    //stop spot
    drawGaussian(frame_, sp_py, sp_px, 1.2, 5, 0);
    frame_(croph_,cropw_).copyTo(dreamers_view_);

}

void ShoreDocker::run() {
    ros::Rate r(rate_);
#ifdef SHOW_IMG
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
#ifdef SHOW_IMG
            cv::imshow("Dreamer's View 64x64", dreamers_view_);
            cv::waitKey(5);
#endif
        }
        r.sleep();
    }
}

int main(int argc, char * argv[]) {
	ros::init(argc,argv,"reward_generator");
	ShoreDocker SD;
    SD.run();
}

