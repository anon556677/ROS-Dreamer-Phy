#ifndef ShoreDocker_H
#define ShoreDocker_H

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
#include <reward_generator/docker_reward.h>

class ShoreDocker {
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
        ros::Subscriber odom_robot_sub_;
        ros::Subscriber odom_boey_sub_;
        ros::Subscriber done_sub_;
        ros::Subscriber refresh_sub_;
        ros::Publisher cmap_pub_;
        ros::Publisher reward_pub_;
        ros::Publisher cp_reward_pub_;
          
        // settings
        int frame_size_;
        int frame_padding_;
        float grid_res_;
        float front_dist_;
        float rear_dist_;
        float dist_to_boey;
        float side_dist_;
        float dist_to_boey_;
        float yaw_coeff_;
        float py_coeff_;
        float px_coeff_;
        float proximity_coeff_;
        int rate_;
        std::string lzr_frame_;
        bool publish_occupancy_grid_;
        int size_;
        int x_offset_;
        int y_offset_;
        bool compute_;
          
        // ROS msgs
        std_msgs::Float32 reward_;
        reward_generator::docker_reward composite_reward_;

        // Reward
        float pose_x_reward_;
        float pose_y_reward_;
        float yaw_reward_;
        float proximity_penalty_;

        // Boey position
        float boey_yaw;
        float boey_px;
        float boey_py;
        // Boey projection in the usv frame
        float boey_yaw_uf;
        float boey_px_uf;
        float boey_py_uf;

        // Vectors used to draw the map from the laser points
        typedef std::vector< std::array<int,3> > PointsVector; 
        PointsVector buffer_;
        PointsVector points_;
        // Colors
        cv::Vec3b RED;
        cv::Vec3b GREEN;
        // Map
        cv::Mat_<cv::Vec3b> dreamers_view_;
        cv::Mat_<cv::Vec3b> empty_frame_;
        cv::Mat_<cv::Vec3b> frame_;

        cv::Range cropw_;
        cv::Range croph_;
          
        // Laser -to-> map functions
        void initializeFrame();
        void refreshFrame();
        void addToBuffer(const sensor_msgs::LaserScanConstPtr &);
        void filterPoints(PointsVector&, PointsVector&);
        void drawShore();
        void drawBoey();
          
        // Callbacks
        void doneCallback(const std_msgs::BoolConstPtr&);
        void refreshCallback(const rl_server::EpisodeConstPtr&);
        void LZRCallback(const sensor_msgs::LaserScanConstPtr&);
        void odomRobotCallback(const nav_msgs::OdometryConstPtr&);
        void odomBoeyCallback(const nav_msgs::OdometryConstPtr&);
        void publishOccupancyGrid();
        void publishReward();
    public:
        ShoreDocker();
        void run();
};

#endif
