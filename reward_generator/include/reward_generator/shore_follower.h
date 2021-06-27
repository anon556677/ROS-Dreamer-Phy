#ifndef ShoreFollower_H
#define ShoreFollower_H

#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/OccupancyGrid.h>
#include <std_msgs/Float32.h>
#include <nav_msgs/Odometry.h>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <rl_server/Episode.h>
#include <std_msgs/Bool.h>
#include <vector>
#include <reward_generator/follower_reward.h>

class ShoreFollower {
    /*
    Generates all that is need for the RL agent to solve its task:
      + Reward
      + Projection of the laser-scanner
    */
    
    private:
        // ROS components
        ros::NodeHandle nh_;
        image_transport::ImageTransport it_;
        image_transport::Publisher image_pub_;
        ros::Subscriber lzr_sub_;
        ros::Subscriber odom_sub_;
        ros::Subscriber done_sub_;
        ros::Subscriber refresh_sub_;
        ros::Publisher cmap_pub_;
        ros::Publisher reward_pub_;
        ros::Publisher cp_reward_pub_;
        
        // settings
        float local_grid_res_;
        float local_grid_front_size_;
        float local_grid_rear_size_;
        float local_grid_side_size_;
        float dist_to_shore_;
        float safe_to_shore_;
        float vel_coeff_;
        float backward_coeff_;
        float pose_coeff_;
        float vel_target_;
        int rate_;
        float max_gap_;
        std::string lzr_frame_;
        bool publish_occupancy_grid_;
        int size_;
        int kernel_size_;
        int safe_kernel_size_;
        int hskernel_;
        int hssafety_;
        int x_offset_;
        int y_offset_;
        int safety_cost_;
        float blur_kernel_size_;
        bool compute_;
        cv::Size blur_ksize_;
        
        // ROS msgs
        nav_msgs::OccupancyGrid occupancy_grid_;
        std_msgs::Float32 reward_;
        reward_generator::follower_reward composite_reward_;

        // Reward
        float pose_reward_;
        float vel_reward_;

        // Vectors used to draw the map from the laser points
        typedef std::vector< std::array<int,2> > PointsVector; 
        PointsVector buffer_;
        PointsVector inner_shores_;
        PointsVector points_;

        // Map
        cv::Mat_<uint8_t> map_;
        cv::Mat_<uint8_t> zero_map_;
        cv::Mat_<uint8_t> one_map_;
        cv::Mat_<uint8_t> kernel_;
        cv::Mat_<uint8_t> safety_;
        cv::Mat_<uint8_t> thin_map_;
        cv::Mat_<int> label_map_;
        cv::Mat_<float> cost_map_;
        cv::Mat_<float> cropped_map_;
        cv::Mat_<float> smoothed_map_;
        cv::Mat_<uint8_t> n_cost_map_;
        cv::Mat_<cv::Vec3b> dreamers_view_;
        cv::Mat_<cv::Vec3b> dreamers_map_;
        cv::Mat_<cv::Vec3b> blue_map_;

        cv::Range cropw_;
        cv::Range croph_;
        
        // Laser -to-> map functions
        void initializeMap();
        void refreshMap();
        void setMapBlockValue(int, int, cv::Mat&, int, cv::Mat&, cv::Scalar);
        void genContourMap();
        void genCostMap();
        void addToBuffer(const sensor_msgs::LaserScanConstPtr &);
        void filterPoints(PointsVector&, PointsVector&);
        void interpolatePoints(PointsVector&, PointsVector&);
        double calcDist(const PointsVector&, size_t, size_t);

        // Callbacks
        void doneCallback(const std_msgs::BoolConstPtr&);
        void refreshCallback(const rl_server::EpisodeConstPtr&);
        void LZRCallback(const sensor_msgs::LaserScanConstPtr&);
        void odomCallback(const nav_msgs::OdometryConstPtr&);
        void publishOccupancyGrid();
        void publishReward();
    public:
        ShoreFollower();
        void run();
};

#endif