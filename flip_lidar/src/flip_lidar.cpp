#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <vector>
#include <algorithm>

//This class takes a LaserScan as an input, and flips it
class FlipLidar {
    protected:
        ros::Subscriber pc_sub_;
        ros::Publisher flipped_pc_pub_;

        ros::NodeHandle nh_;
        std::string base_frame_;

    protected:

        void pc_callback(const sensor_msgs::LaserScanPtr msg) {
            std::vector<float> ranges = msg->ranges;
            std::vector<float> intensities = msg->intensities;
            std::reverse(ranges.begin(), ranges.end());
            std::reverse(intensities.begin(), intensities.end());

            sensor_msgs::LaserScan output;
            output.header.frame_id = base_frame_;
            output.header.stamp = ros::Time::now();
            output.angle_min = msg->angle_min;
            output.angle_max = msg->angle_max;
            output.angle_increment = msg->angle_increment;
            output.time_increment = msg->time_increment;
            output.scan_time = msg->scan_time;
            output.range_min = msg->range_min;
            output.range_max = msg->range_max;
            output.ranges = ranges;
            output.intensities = intensities;
            
            flipped_pc_pub_.publish(output);

        }

    public:
        FlipLidar() : nh_("~") {

            nh_.param("base_frame",base_frame_,std::string("/base_link"));

            pc_sub_ = nh_.subscribe("input_scan",1,&FlipLidar::pc_callback,this);
            flipped_pc_pub_ = nh_.advertise<sensor_msgs::LaserScan>("output_scan",1);
        }

};

int main(int argc, char * argv[]) 
{
    ros::init(argc,argv,"flip_lidar");
    FlipLidar of;
    ros::spin();
}

